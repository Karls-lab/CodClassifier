"""
A script to train the YOLO model.
"""
import albumentations as A
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import cv2
import os 
import pandas as pd
import sys
import shutil
import random
from resizeImages import resize_images


cur_dir = os.path.dirname(__file__)
folderName = 'images'
dest_folder = 'NormalizedImages'
path = os.path.join(cur_dir, folderName)
print(path)

# get the path names of the images
image_path_names = []
text_path_names = []
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".jpg"):
            image_path_names.append(os.path.join(file))
        elif file.endswith(".txt"):
            text_path_names.append(os.path.join(file))



# sys.exit()

resize_images(image_path_names, folderName, (512, 512))

# sys.exit()


# split the images into training and testing sets
random.shuffle(image_path_names)
train_test_split = 0.8
train_size = int(len(image_path_names) * train_test_split)
train_image_path_names = image_path_names[:train_size]
test_image_path_names = image_path_names[train_size:]

# write the image path names to the train and test list files
with open('train.txt', 'w') as f:
    for image_name in train_image_path_names:
        f.write(f"{dest_folder}/{image_name}\n")

with open('valid.txt', 'w') as f:
    for image_name in test_image_path_names:
        f.write(f"{dest_folder}/{image_name}\n")


"""
Star the Augmentation Process for each image in the
training set
"""
# Declare an augmentation pipeline
transform = A.Compose([
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0, scale_limit=0.2, rotate_limit=45, p=1),
    A.HorizontalFlip(p=1),
    A.Transpose(p=.5),
    A.VerticalFlip(p=1),
    A.RandomGamma(p=.2),
    A.RGBShift(p=.5),
    A.CLAHE(p=.5),
    A.HueSaturationValue(p=.2),
    A.ChannelShuffle(p=.5),
    A.GaussNoise(p=0.2),
    A.Perspective(p=1),
    A.Resize(width=512, height=512),  # Resize images
], bbox_params=A.BboxParams(format="yolo", label_fields=['class_labels']))


# Now for each image in the training set, augment it 2 times
train_image_path_names = []
with open('train.txt', 'r') as f:
    for line in f:
        train_image_path_names.append(line.strip())

"""
Augment every Image in the training.txt file twice
"""
print(f"train_image_path_names: {train_image_path_names}")
for image_path in train_image_path_names:
    for i in range(2):
        box_path = image_path[:-4] + ".txt"
        aug_image_name = image_path.split('/')[-1].split('.')[0] # no .txt or .jpg
        aug_image_path = f"{dest_folder}/{aug_image_name}_aug{i}.jpg"
        aug_box_path = f"{dest_folder}/{aug_image_name}_aug{i}.txt"
        jpgImageCopy = shutil.copy(image_path, aug_image_path)
        bounding_box = shutil.copy(f"{box_path}", f"{aug_box_path}")

        # Read the bounding box file
        # Bounding box format: [class_id, x_center, y_center, width, height]
        bboxes = []
        with open(bounding_box, "r") as f:
            lines = f.readlines()
            # print(f"lines: {lines}")
            for line in lines:
                # print(f"line: {line}")
                bounding_box = [float(x) for x in line.strip().split(" ")]
                bboxes.append(bounding_box)

        print(f"bounding_box: {bboxes}")
        modified_boxes = [[float(box[1]), float(box[2]), float(box[3]), float(box[4])] for box in bboxes]
        print(f"modified_boxes: {modified_boxes}")

        # Read an image with OpenCV and convert it to the RGB colorspace
        image = cv2.imread(aug_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed = transform(image=image, bboxes=modified_boxes, class_labels=[0]*len(modified_boxes))

        transformed_image = transformed["image"]
        transformed_bboxes = transformed["bboxes"]

        print(f"transformed_bboxes: {transformed_bboxes}")



        # Print out the bounding box coords to check if everything is correct
        print(f"yolo bboxes: {bboxes}")

        # Now save the augmented image and bounding boxes to a file
        cv2.imwrite(aug_image_path, transformed_image)
        with open(aug_box_path, "w") as f:
            for box in transformed_bboxes:
                f.write("0 " + " ".join([str(x) for x in box]) + "\n")


        # Append the augmented image path to the train.txt file
        with open('train.txt', 'a') as f:
            f.write(f"\n{aug_image_path}")



