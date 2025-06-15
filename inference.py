import cv2
import torch
import numpy as np
from utils.general import get_model, WIDTH, HEIGHT, to_img
from test import predict_location
import argparse
import os
import csv
import time
from typing import Optional, Tuple, List


def generate_background_image(
    cap: cv2.VideoCapture, max_frames: int = 100
) -> Optional[torch.Tensor]:
    """Generate a background image as the median of max_frames frames, uniformly sampled from the entire video.
    If the video is longer than 2 minutes, skip the first minute for sampling.

    Args:
        cap: OpenCV VideoCapture object for the video.
        max_frames: Maximum number of frames to sample.

    Returns:
        Torch tensor of the background image, or None if generation fails.
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total_frames <= 0 or fps <= 0:
        print("Warning: Could not determine total frame count or FPS.")
        return None

    # Calculate video duration in seconds
    duration = total_frames / fps
    start_frame = 0
    if duration > 120:  # If longer than 2 minutes, skip first minute
        start_frame = int(fps * 60)
        print(
            f"Video duration {duration:.2f}s > 120s, skipping first minute ({start_frame} frames)"
        )

    max_frames = max(
        10, min(max_frames, int((total_frames - start_frame) * 0.01) or max_frames)
    )
    step = max(1, (total_frames - start_frame) // max_frames)
    frame_indices = [
        start_frame + int(i * step)
        for i in range(max_frames)
        if start_frame + i * step < total_frames
    ]

    frames = []
    cnt = 0
    original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (WIDTH, HEIGHT))
        frame_normalized = frame_resized / 255.0
        frames.append(frame_normalized)
        cnt += 1
        if cnt % 10 == 0:
            print(f"Read {cnt}/{max_frames} frames for background generation")
    cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)

    if not frames:
        print("Warning: No frames were read for background generation.")
        return None

    median_frame = np.median(frames, axis=0)
    return torch.from_numpy(median_frame).permute(2, 0, 1).float().cuda()


def initialize_model(model_path: str) -> Tuple[Optional[torch.nn.Module], int, str]:
    """Initialize the TrackNet model from a checkpoint.

    Args:
        model_path: Path to the TrackNet model checkpoint.

    Returns:
        Tuple of (model, sequence_length, background_mode), or (None, 0, '') on failure.
    """
    try:
        tracknet_ckpt = torch.load(model_path)
        seq_len = tracknet_ckpt["param_dict"]["seq_len"]
        bg_mode = tracknet_ckpt["param_dict"]["bg_mode"]
        print(f"Checkpoint: seq_len={seq_len}, bg_mode={bg_mode}")
        model = get_model("TrackNet", seq_len, bg_mode).cuda()
        model.load_state_dict(tracknet_ckpt["model"])
        model.eval()
        return model, seq_len, bg_mode
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, 0, ""


def initialize_video(
    video_path: str, output_file: Optional[str]
) -> Tuple[Optional[cv2.VideoCapture], Optional[cv2.VideoWriter], float, float, float]:
    """Initialize video capture and optional video writer.

    Args:
        video_path: Path to the input MP4 video file.
        output_file: Path to save output video, or None.

    Returns:
        Tuple of (capture, video_writer, fps, width_scaler, height_scaler), or (None, None, 0, 0, 0) on failure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return None, None, 0, 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_scaler = w / WIDTH
    h_scaler = h / HEIGHT

    video_writer = None
    if output_file:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, (w, h))
        if not video_writer.isOpened():
            print("Error initializing video writer")
            cap.release()
            return None, None, 0, 0, 0

    return cap, video_writer, fps, w_scaler, h_scaler


def process_frame(
    frame: np.ndarray,
    frame_buffer: List[torch.Tensor],
    tracknet: torch.nn.Module,
    seq_len: int,
    bg_mode: str,
    background_tensor: Optional[torch.Tensor],
    w_scaler: float,
    h_scaler: float,
    t: int,
) -> Tuple[int, int, int]:
    """Process a single video frame and predict ball location.

    Args:
        frame: Input frame in BGR format.
        frame_buffer: Buffer of recent frame tensors.
        tracknet: TrackNet model.
        seq_len: Sequence length for the model.
        bg_mode: Background mode ('concat' or other).
        background_tensor: Background image tensor, or None.
        w_scaler: Width scaling factor.
        h_scaler: Height scaling factor.
        t: Current frame number.

    Returns:
        Tuple of (center_x, center_y, visibility).
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (WIDTH, HEIGHT))
    frame_normalized = frame_resized / 255.0
    frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).float().cuda()

    frame_buffer.append(frame_tensor)
    if len(frame_buffer) > seq_len:
        frame_buffer.pop(0)

    if len(frame_buffer) == seq_len:
        x = torch.stack(frame_buffer, dim=0)  # (seq_len, C, H, W)
        x = x.view(1, -1, HEIGHT, WIDTH)  # (1, C * seq_len, H, W)
        if bg_mode == "concat" and background_tensor is not None:
            x = torch.cat([background_tensor.unsqueeze(0), x], dim=1)
        #print(f"Input shape to model: {x.shape}")
        with torch.no_grad():
            y_pred = tracknet(x).detach().cpu()
        y_p = y_pred[0, -1].numpy() > 0.5
        bbox = predict_location(to_img(y_p))
        if bbox is not None:
            cx = int(bbox[0] + bbox[2] / 2)
            cy = int(bbox[1] + bbox[3] / 2)
            cx = int(cx * w_scaler)
            cy = int(cy * h_scaler)
            vis = 1
        else:
            cx, cy = 0, 0
            vis = 0
    else:
        cx, cy, vis = 0, 0, 0

    return cx, cy, vis


def display_frame(
    frame: np.ndarray, pred_history: List[Tuple[int, int, int, int]], traj_len: int
) -> np.ndarray:
    """Draw ball predictions on the frame.

    Args:
        frame: Input frame in BGR format.
        pred_history: List of past predictions (frame, x, y, visibility).
        traj_len: Number of past predictions to draw.

    Returns:
        Frame with drawn predictions.
    """
    display_frame = frame.copy()
    num_to_draw = min(traj_len, len(pred_history))
    for pred in pred_history[-num_to_draw:]:
        _, x, y, v = pred
        if v == 1:
            cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
    return display_frame


def write_csv_incrementally(
    csv_writer: Optional[csv.writer], csv_file: Optional, data: List[int], frame: int
) -> None:
    """Write a single row of prediction data to the CSV file.

    Args:
        csv_writer: CSV writer object, or None.
        csv_file: File object for CSV, or None.
        data: List of [frame, visibility, x, y].
        frame: Current frame number.
    """
    if csv_writer and csv_file:
        csv_writer.writerow(data)
        csv_file.flush()  # Ensure data is written immediately
        if frame % 100 == 0:  # Log every 100 frames to confirm writing
            print(f"Wrote frame {frame} to CSV")


def report_fps(start_time: float, frame_count: int, last_report_time: float) -> float:
    """Report processing FPS every 10 seconds.

    Args:
        start_time: Start time of processing (seconds since epoch).
        frame_count: Number of frames processed.
        last_report_time: Time of the last FPS report.

    Returns:
        Updated last_report_time if a report was made, else unchanged.
    """
    current_time = time.time()
    if current_time - last_report_time >= 10:
        elapsed_time = current_time - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time
            print(f"Processing speed: {fps:.2f} FPS (based on {frame_count} frames)")
        return current_time
    return last_report_time


def main():
    parser = argparse.ArgumentParser(
        description="Track a ball in a video using TrackNet."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the TrackNet model checkpoint",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to the MP4 video file",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save output video file",
    )
    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="Save ball prediction data to CSV file next to input video",
    )
    args = parser.parse_args()

    # Initialize model
    tracknet, seq_len, bg_mode = initialize_model(args.model_path)
    if tracknet is None:
        return

    # Initialize video
    cap, video_writer, fps, w_scaler, h_scaler = initialize_video(
        args.video_path, args.output_file
    )
    if cap is None:
        return

    # Initialize CSV writer (next to input video_path)
    csv_file = None
    csv_writer = None
    if args.save_csv:
        csv_path = os.path.join(
            os.path.dirname(args.video_path),
            os.path.splitext(os.path.basename(args.video_path))[0]
            + "_ball_predict.csv",
        )
        try:
            csv_file = open(csv_path, "w", newline="")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Frame", "Visibility", "X", "Y"])
            print(f"Initialized CSV file at {csv_path}")
        except Exception as e:
            print(f"Error initializing CSV file: {e}")
            cap.release()
            if video_writer:
                video_writer.release()
            return

    # Generate and display background image if needed
    background_tensor = None
    if bg_mode == "concat":
        print(f"Generating background image with 30 frames...")
        background_tensor = generate_background_image(cap, max_frames=30)
        if background_tensor is None:
            print("Error: Could not generate background image for bg_mode='concat'.")
            cap.release()
            if video_writer:
                video_writer.release()
            if csv_file:
                csv_file.close()
            return
        print(f"Background tensor shape: {background_tensor.shape}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Initialize buffers and timing
    frame_buffer = []
    pred_history = []
    traj_len = 8
    t = 0
    start_time = time.time()
    last_report_time = start_time

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        cx, cy, vis = process_frame(
            frame,
            frame_buffer,
            tracknet,
            seq_len,
            bg_mode,
            background_tensor,
            w_scaler,
            h_scaler,
            t,
        )
        pred_history.append((t, cx, cy, vis))
        if args.save_csv:
            write_csv_incrementally(csv_writer, csv_file, [t, vis, cx, cy], t)

        # Display frame
        # display_frame_result = display_frame(frame, pred_history, traj_len)
        # if video_writer:
        #     video_writer.write(display_frame_result)
        # cv2.imshow("Prediction", display_frame_result)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

        # Report FPS
        last_report_time = report_fps(start_time, t + 1, last_report_time)
        t += 1

    # Cleanup
    cap.release()
    if video_writer:
        video_writer.release()
    if csv_file:
        csv_file.close()
        print(f"Completed writing CSV file at {csv_path}")
    cv2.destroyAllWindows()
    print(f"Processed {t} frames in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
