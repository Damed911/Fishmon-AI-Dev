import torch

# Load the pre-trained model from the .pt file
model = torch.load("G:\\Hasil training\\Dataset Kapal\\Object Detection\\yolov7-800\\weights\\best.pt")

# Calculate the number of trainable parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"The model has {num_params:,} trainable parameters.")
