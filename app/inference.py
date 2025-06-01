# app/inference.py

import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def predict(params):
    input_array = np.array(params["input"], dtype=np.float32)

    output = session.run([output_name], {input_name: input_array})
    predicted_class = int(np.argmax(output[0]))
    
    return {"class_id": predicted_class}
