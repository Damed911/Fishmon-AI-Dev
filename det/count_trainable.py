import torch
from torchvision import models
from torchinfo import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Load the pre-trained model from the .pt file
model = torch.load("runs/train/yolov7-800/weights/best.pt", map_location=device)["model"]

# print(model)
# n_p = sum(x.numel() for x in arch.parameters())

# print(f"There are {n_p} Trainable Parameters")

summary(model.to(torch.float), input_size=(4,3,800,800))