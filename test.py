# test.py

import argparse
from model import ImagePreprocessor, OnnxModel


def run_test(model_path: str, image_path: str):
    print("Preprocessing image...")
    preprocessor = ImagePreprocessor()
    input_tensor = preprocessor.preprocess(image_path)

    print("Running inference...")
    model = OnnxModel(model_path)
    predicted_class = model.predict(input_tensor)

    print(f" Predicted Class ID: {predicted_class}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model.onnx", help="Path to ONNX model")
    parser.add_argument("--image", required=True, help="Path to image to classify")
    args = parser.parse_args()

    run_test(args.model, args.image)
