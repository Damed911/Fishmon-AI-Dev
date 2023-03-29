import os

# Path to the directory containing images and annotations
data_dir = "E:\\Dataset Kapal bb\\combine"

for img_file in os.listdir(data_dir):
    img_file_id = '.'.join(img_file.split('.')[:-1])
        # Load instance segmentation labels
    seg_path = os.path.join(data_dir, img_file_id + ".txt")
    if not os.path.exists(seg_path):
        continue
    else:
        with open(seg_path, "r") as f:
            labels = f.readlines()

    new_labels = []
    for label in labels:
        label = label.strip().split()
        object_class = label[0]
        vertices = label[1:]

        x_values = [float(vertices[i]) for i in range(0, len(vertices), 2)]
        y_values = [float(vertices[i+1]) for i in range(0, len(vertices), 2)]
        
        # Calculate the minimum bounding box that encompasses the polygon vertices
        xmin = min(x_values)
        xmax = max(x_values)
        ymin = min(y_values)
        ymax = max(y_values)

        x, y, w, h = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0, xmax - xmin, ymax - ymin
        
        new_labels.append(f"{object_class} {x} {y} {w} {h}")

    save_dir = os.path.join(data_dir, "result")
    with open(os.path.join(save_dir, img_file_id + ".txt"), 'w') as f:
        f.write('\n'.join(new_labels))