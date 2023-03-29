import cv2
import numpy as np
import os

data_dir = "C:\\Users\\Daffa Ajiputra\\Documents\\Skripsi\\Dataset Kapal bb\\combine"

# Load the image and its corresponding label file
img = cv2.imread(os.path.join(data_dir, '00a72b4f-hiv00031-06912.jpeg'))
img = cv2.resize(img, (1920, 1080))
label_file = open(os.path.join(data_dir, '00a72b4f-hiv00031-06912.txt'), 'r')

# Loop through each line of the label file
for line in label_file:
    # Parse the label values
    values = line.strip().split(' ')
    class_id = int(values[0])
    x_center = float(values[1])
    y_center = float(values[2])
    width = float(values[3])
    height = float(values[4])

    # Calculate the bounding box coordinates
    left = int((x_center - width / 2) * img.shape[1])
    top = int((y_center - height / 2) * img.shape[0])
    right = int((x_center + width / 2) * img.shape[1])
    bottom = int((y_center + height / 2) * img.shape[0])

    # Draw the bounding box on the image
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

# Show the resulting image
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()