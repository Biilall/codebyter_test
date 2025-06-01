# 🔍 ImageNet ONNX Classifier - Deployment Ready

This project contains a deep learning image classifier trained on the ImageNet dataset, implemented in PyTorch and converted to ONNX for production deployment on [Cerebrium](https://cerebrium.ai/). The model takes a 224x224 image and predicts one of 1000 ImageNet classes.

---

##  Project Structure
├── pytorch_model.py 
├── convert_to_onnx.py 
├── model.py
├── test.py # Local model test
├── test_server.py # Inference test against deployed Cerebrium API
├── resnet18-f37072fd.pth # (Download manually - weights)
├── n01440764_tench.JPEG # Test image (class ID 0)
├── n01667114_mud_turtle.JPEG # Test image (class ID 35)
├── model.onnx 
├── requirements.txt
└── README.md


---

## 🧰 Requirements

```bash
pip install -r requirements.txt

```


## file convert

```bash
python convert_to_onnx.py
```

- you will get a file

```bash
model.onnx
```


## test locally
```bash
python test.py --image images/n01440764_tench.JPEG
```

- Expected output:
```bash
Predicted Class ID: 0
```


