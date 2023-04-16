import torch

# Load the pre-trained model from the .pt file
model = torch.load("runs/train/yolov7-800-AdamW/weights/last.pt")

arch = model['model']
n_p = sum(x.numel() for x in arch.parameters())

print(f"There are {n_p} Trainable Parameters")