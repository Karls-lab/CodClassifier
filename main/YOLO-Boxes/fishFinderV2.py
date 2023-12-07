"""
This python module will find the fish in the image, crop it, and give save it as a file
Version 2 will iterates over the folders of images in the training set, and save the cropped images 
into cropped_CLASSNAME folders
"""

# Example using OpenCV (assuming darknet is compiled with OpenCV support)
import glob
import cv2
import numpy as np
import os
import random
import sys

# Load YOLO
home_dir = os.path.expanduser("~")
net = cv2.dnn.readNet("backup/yolov4_final.weights", "Setup/yolov3.cfg")  # Update the paths to your downloaded weights and configuration file
classes = ['fish']

# Function to get output layers from YOLO
layer_names = net.getLayerNames()
unconnected_layers = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in unconnected_layers]

def resizeToBlack(folderPath):
    print(folderPath)
    folderName = folderPath.split("/")[-1]
    output_directory = folderPath.replace(folderName, "cropped_" + folderName)
    jpg_files = glob.glob(os.path.join(folderPath, "*.jpg"))
    print(folderName)
    print(f"Number of images in folder: {len(jpg_files)}")
    for imagePath in jpg_files:
        image = cv2.imread(imagePath)
        image = np.zeros(image.shape, np.uint8)
        image = cv2.resize(image, (299, 299))

        image_name = imagePath.split("/")[-1]
        output_path = os.path.join(output_directory, f"{image_name}")

        # create a new folder for the cropped images if it doesn't exist 
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        print(f"Cropped image saved to: {output_path}")
        try:
            cv2.imwrite(output_path, image)
        except:
            print(f"Error: Cropped image is empty. ")
            continue



def cropImages(folderPath, keepNoF=False):
    folderName = folderPath.split("/")[-1]
    output_directory = folderPath.replace(folderName, "cropped_" + folderName)
    # Load all images from the class folder
    jpg_files = glob.glob(os.path.join(folderPath, "*.jpg"))
    print(f"\nFolder path: {folderPath}")
    print(f"Number of images in folder: {len(jpg_files)}")

    # Loop over the images in the folder
    for image in jpg_files:
        img = cv2.imread(image)
        img = cv2.resize(img, None, fx=1, fy=1)  
        height, width, _ = img.shape  

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (608, 608), (0, 0, 0), False, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        # print(f"Image shape: {img.shape}")
        # print(f"Sample pixel values: {img[0, 0]}")


        # Showing information on the screen
        class_ids = []
        confidences = []
        boxes = []
        max_confidence_index = -1
        max_confidence = 0.0

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.70:  # Adjust confidence threshold as needed
                    # Object detected
                    # print(f"Detected fish with prob: {confidence}")
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Check if the current detection has a higher confidence
                    if confidence > max_confidence:
                        max_confidence = confidence
                        max_confidence_index = len(boxes)  # Index in the 'boxes' list

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)


        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Output directory for saving cropped images
        os.makedirs(output_directory, exist_ok=True)

        # Loop over the detected objects and save cropped images
        x, y, w, h = 0, 0, 0, 0

        # if there was no detection, skip or keep for stg1 and 2
        if len(boxes) == 0:
            if keepNoF:
                x, y, w, h = 0, 0, width, height
            else:
                continue
        else: # else, crop the image based on the bounding box coordinates
            x, y, w, h = boxes[0]

        # Crop the image based on the bounding box coordinates
        cropped_img = img[y:y + h, x:x + w]

        # Resize every image to 299, 299, 3
        if cropped_img.size != 0:
            cropped_img = cv2.resize(cropped_img, (299, 299))
        else:
            print(f"Error: Cropped image is empty. ")
            continue

        # Save the cropped image to a file
        image_name = image.split("/")[-1]
        output_path = os.path.join(output_directory, f"{image_name}")

        # create a new folder for the cropped images if it doesn't exist 
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Save the cropped image to a file
        print(f"Cropped image saved to: {output_path}")
        cv2.imwrite(output_path, cropped_img)


# Get the images from each class folder in the training set
training_set_path = "../../data/raw_train"
folders = os.listdir(training_set_path)
folders = [os.path.join(training_set_path, folder) for folder in folders]

# NoF = folders[folders.index("../../data/raw_train/NoF")]
# folders.remove(NoF)
# resizeToBlack(NoF)

# print(folders)
# for folder in folders:
#     cropImages(folder)
#     print(f"Finished cropping images in folder: {folder}")

stg1_crop = "../../data/test_stg2"
cropImages(stg1_crop, True)

