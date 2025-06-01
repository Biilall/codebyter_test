# test_server.py

import requests
import base64
import argparse
import os
from model import ImagePreprocessor
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def run_inference(api_url, api_key, image_path):
    preprocessor = ImagePreprocessor()
    input_tensor = preprocessor.preprocess(image_path)

    payload = {
        "input": input_tensor.tolist()
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    print(" Sending request to Cerebrium...")
    # print("payload", payload)
    print("api url", api_url)
    response = requests.post(api_url, json=payload, headers=headers)

    if response.status_code == 200:
        print(" Prediction Result:", response.json())
    else:
        print(f" Error [{response.status_code}]: {response.text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    # parser.add_argument("--api-url", default=os.getenv("CEREBRIUM_API_URL"), help="Cerebrium endpoint URL")
    # parser.add_argument("--api-key", default=os.getenv("CEREBRIUM_API_KEY"), help="Cerebrium API key")
    args = parser.parse_args()

    # if not args.api_url or not args.api_key:
    #     print(" Missing API URL or API Key. Provide via arguments or environment variables.")
    #     exit(1)

    # run_inference(args.api_url, args.api_key, args.image)
    print("api url",os.getenv("CEREBRIUM_API_URL"))
    run_inference(os.getenv("CEREBRIUM_API_URL"), os.getenv("CEREBRIUM_API_KEY"), args.image)
    
