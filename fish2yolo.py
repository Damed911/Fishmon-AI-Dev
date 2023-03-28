import xml.etree.ElementTree as ET
from tqdm import tqdm
import os
import shutil
import argparse

import numpy as np
import cv2

import yaml

from sklearn.model_selection import train_test_split

FISH_NAMES = ['Black Sea Sprat', 'Gilt-Head Bream', 'Hourse Mackerel', 'Red Mullet', 
              'Red Sea Bream', 'Sea Bass', 'Shrimp', 'Striped Red Mullet', 'Trout']

RANDOM_STATE = 42

CONFIG = {
    "train": "",
    "val": "",
    "test": "",
    "is_coco": False,
    "nc": len(FISH_NAMES),
    "names": FISH_NAMES
}

def convert_box(size, box):
    dw, dh = 1. / size[0], 1. / size[1]
    x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh

def gen_data(X_path, y_label, output_path, split):

    images_path = os.path.join(output_path, 'images', split)
    labels_path = os.path.join(output_path, 'labels', split)
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    for (class_name_path, filename, label), label_id in tqdm(zip(X_path, y_label)):
        id = '.'.join(filename.split('.')[:-1])
        label_name = f'{id}.txt'

        ori_image_path = os.path.join(class_name_path, filename)
        ori_label_path = f'{class_name_path} Label/{label_name}'

        new_image_path = os.path.join(images_path, f'{label_id}_{label}_{filename}')
        new_label_path = os.path.join(labels_path, f'{label_id}_{label}_{label_name}')

        shutil.copy(ori_image_path, new_image_path)
        shutil.copy(ori_label_path, new_label_path)
    
    CONFIG[split] = images_path


def split_data(data_path, output_path=None):
    print("Splitting data ...")

    x_images_path = []
    y_labels = []

    for i, label in enumerate(FISH_NAMES):
        class_name_path = os.path.join(data_path, label, label)
        for filename in os.listdir(class_name_path):
            x_images_path.append((class_name_path, filename, label))
            y_labels.append(i)

    X_train, X_test, y_train, y_test = train_test_split(
                                            x_images_path, 
                                            y_labels, 
                                            test_size=0.2,
                                            stratify=y_labels,
                                            random_state=RANDOM_STATE
                                        )

    X_train, X_val, y_train, y_val = train_test_split(
                                            X_train, 
                                            y_train, 
                                            test_size=0.125, 
                                            stratify=y_train,
                                            random_state=RANDOM_STATE
                                        )

    print("Split data success!")
    print(f'Total data : {len(x_images_path)}')
    print(f'Train size : {len(X_train)} | {len(X_train)/len(x_images_path)*100}%')
    print(f'Test size  : {len(X_test)} | {len(X_test)/len(x_images_path)*100}%')
    print(f'Val size   : {len(X_val)} | {len(X_val)/len(x_images_path)*100}%')

    if output_path is None:
        output_path = data_path

    print("Copying data ...")
    gen_data(X_train, y_train, output_path, 'train')
    gen_data(X_test, y_test, output_path, 'test')
    gen_data(X_val, y_val, output_path, 'val')
    print("Copying data success!")


def main(args):
    data_path = args.data_path

    # FISH_NAMES = [p for p in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, p))]
    # FISH_NAMES.sort()
    # print(FISH_NAMES)

    for label in FISH_NAMES:
        image_path = os.path.join(data_path, label, label)

        gt_path = f'{image_path} GT'
        label_path = f'{image_path} Label'
        # output_path = f'{ori_path} Test'

        os.makedirs(label_path, exist_ok=True)

        for filename in tqdm(os.listdir(gt_path), desc=label):

            id = '.'.join(filename.split('.')[:-1])

            try:
                image = cv2.imread(os.path.join(gt_path, filename), cv2.IMREAD_GRAYSCALE)

                h, w = image.shape
                y_coord, x_coord = image.nonzero()

                xmin = x_coord.min()
                xmax = x_coord.max()
                ymin = y_coord.min()
                ymax = y_coord.max()

                bb = convert_box((w, h), [xmin, xmax, ymin, ymax])

                with open(os.path.join(label_path, f'{id}.txt'), 'w') as f:
                    f.write(" ".join([str(a) for a in (FISH_NAMES.index(label), *bb)]) + '\n')

            except Exception as e:
                print(f'[Warning]: {e} Convert image with name {filename} is failed!')

    split_data(data_path, args.output_path)

    with open(r'./data/fish.yaml', 'w') as file:
        conf = yaml.dump(CONFIG, file)
        print(conf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--data_path', type=str, help='Data path to images', required=True)
    parser.add_argument('-o', '--output_path', type=str, help='Output data path to images')

    args = parser.parse_args()
    print(args)

    main(args)
