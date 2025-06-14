import cv2
import torch
import numpy as np
from utils.general import get_model, WIDTH, HEIGHT, to_img
from test import predict_location
import argparse
import os
import csv


def generate_background_image(cap, max_frames=100):
    """Generate a background image as the median of max_frames frames, uniformly sampled from the entire video."""
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("Warning: Could not determine total frame count.")
        return None

    # Automatically determine max_frames (1% of total frames, min 10, max 200)
    max_frames = max(10, min(max_frames, int(total_frames * 0.01) or max_frames))
    step = max(1, total_frames // max_frames)  # Step size for uniform sampling
    frame_indices = [
        int(i * step) for i in range(max_frames) if i * step < total_frames
    ]

    frames = []
    cnt = 0
    original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)  # Save current position
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
    cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)  # Restore position

    if not frames:
        print("Warning: No frames were read for background generation.")
        return None

    # Compute median across frames
    median_frame = np.median(frames, axis=0)
    return torch.from_numpy(median_frame).permute(2, 0, 1).float().cuda()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the TrackNet model checkpoint",
    )
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to the MP4 video file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save output video file (if specified, also saves *_ball_predict.csv)",
    )
    args = parser.parse_args()

    # Load model
    tracknet_ckpt = torch.load(args.model_path)
    tracknet_seq_len = tracknet_ckpt["param_dict"]["seq_len"]
    bg_mode = tracknet_ckpt["param_dict"]["bg_mode"]
    print(f"Checkpoint: seq_len={tracknet_seq_len}, bg_mode={bg_mode}")

    tracknet = get_model("TrackNet", tracknet_seq_len, bg_mode).cuda()
    tracknet.load_state_dict(tracknet_ckpt["model"])
    tracknet.eval()
    max_frames = 30  # Default max frames for background generation
    # Initialize video capture
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_scaler = w / WIDTH
    h_scaler = h / HEIGHT

    # Initialize video writer if output_file is specified
    video_writer = None
    csv_data = []
    if args.output_file:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(args.output_file, fourcc, fps, (w, h))
        if not video_writer.isOpened():
            print("Error initializing video writer")
            cap.release()
            return

    # Generate background image if needed
    background_tensor = None
    if bg_mode == "concat":
        print(f"Generating background image with {max_frames} frames...")
        background_tensor = generate_background_image(cap, max_frames)
        if background_tensor is None:
            print(
                "Error: Could not generate background image for bg_mode='concat'. Exiting."
            )
            cap.release()
            if video_writer:
                video_writer.release()
            return
        print(f"Background tensor shape: {background_tensor.shape}")
        cap.set(
            cv2.CAP_PROP_POS_FRAMES, 0
        )  # Reset to start after background generation

    # Initialize buffers
    frame_buffer = []
    pred_history = []
    traj_len = 8  # Number of past predictions to draw
    t = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (WIDTH, HEIGHT))
        frame_normalized = frame_resized / 255.0
        frame_tensor = (
            torch.from_numpy(frame_normalized).permute(2, 0, 1).float().cuda()
        )

        frame_buffer.append(frame_tensor)
        if len(frame_buffer) > tracknet_seq_len:
            frame_buffer.pop(0)

        if len(frame_buffer) == tracknet_seq_len:
            # Stack frames and reshape to (1, C * seq_len, H, W)
            x = torch.stack(frame_buffer, dim=0)  # (seq_len, C, H, W)
            x = x.view(1, -1, HEIGHT, WIDTH)  # (1, C * seq_len, H, W)
            # Add background channels for concat mode
            if bg_mode == "concat" and background_tensor is not None:
                x = torch.cat(
                    [background_tensor.unsqueeze(0), x], dim=1
                )  # (1, 3 + C * seq_len, H, W)
            print(f"Input shape to model: {x.shape}")  # Debug
            with torch.no_grad():
                y_pred = tracknet(x).detach().cpu()  # (1, seq_len, H, W)
            y_p = y_pred[0, -1].numpy() > 0.5  # for last frame
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
            pred_history.append((t, cx, cy, vis))
            if args.output_file:
                csv_data.append([t, vis, cx, cy])
        else:
            # For frames before seq_len, assume no prediction
            pred_history.append((t, 0, 0, 0))
            if args.output_file:
                csv_data.append([t, 0, 0, 0])

        # Display
        display_frame = frame.copy()
        if len(pred_history) > 0:
            num_to_draw = min(traj_len, len(pred_history))
            for pred in pred_history[-num_to_draw:]:
                f, x, y, v = pred
                if v == 1:
                    cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)

        # Write to video file if specified
        if video_writer:
            video_writer.write(display_frame)

        cv2.imshow("Prediction", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        t += 1

    # Save CSV file if output_file is specified
    if args.output_file and csv_data:
        csv_path = os.path.splitext(args.output_file)[0] + "_ball_predict.csv"
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Frame", "Visibility", "X", "Y"])
            writer.writerows(csv_data)
        print(f"Saved CSV file to {csv_path}")

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
