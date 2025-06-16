import torch
from utils.general import get_model  # Adjust import based on your project structure
import argparse
from utils.general import get_model, WIDTH, HEIGHT, to_img
import onnx

def export_to_onnx(model_path, output_onnx_path, seq_len, bg_mode):
    # Load the model
    tracknet_ckpt = torch.load(model_path, map_location="cpu")
    seq_len = tracknet_ckpt["param_dict"]["seq_len"]
    bg_mode = tracknet_ckpt["param_dict"]["bg_mode"]
    print(f"Exporting model with seq_len={seq_len}, bg_mode={bg_mode}")

    model = get_model("TrackNet", seq_len, bg_mode)
    model.load_state_dict(tracknet_ckpt["model"])
    model.eval()

    # Create a dummy input based on model requirements
    C, H, W = 3, HEIGHT, WIDTH  # Adjust HEIGHT and WIDTH as per utils.general
    input_channels = C * seq_len if bg_mode != "concat" else C * (seq_len + 1)
    dummy_input = torch.randn(1, input_channels, H, W)  # Shape: (1, C * seq_len, H, W)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
    #    opset_version=11,  # Use a compatible opset
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"Model exported to {output_onnx_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pt model")
    parser.add_argument("--output_onnx", type=str, required=True, help="Path to save ONNX model")
    args = parser.parse_args()
    export_to_onnx(args.model_path, args.output_onnx, seq_len=8, bg_mode="concat")

    # Example usage:
    model = onnx.load(args.output_onnx)
    onnx.checker.check_model(model)
    print("ONNX model is valid!")
