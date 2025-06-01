# convert_to_onnx.py

import torch
from pytorch_model import Classifier, BasicBlock

def convert_model(weights_path="pytorch_model_weights.pth", output_path="model.onnx"):
    print("Loading model...")
    model = Classifier(BasicBlock, [2, 2, 2, 2])
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    print("Exporting to ONNX...")
    torch.onnx.export(
        model, dummy_input, output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11
    )

    print(f"Model successfully converted and saved to {output_path}")

if __name__ == "__main__":
    convert_model()
