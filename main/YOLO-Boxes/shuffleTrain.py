import random
import os
import cv2
import numpy as np


def shuffleTrain():
    # Read the original train.txt file
    with open('train.txt', 'r') as f:
        lines = f.readlines()
    random.shuffle(lines)
    with open('train.txt', 'w') as f:
        f.writelines(lines)


# For every image in the dest_folder, normalize the pixel values 
# to be between 0 and 1
def normalizePixels(folder):
    for file in os.listdir(folder):
        if file[-1:-4] != ".jpg": continue
        image_path = os.path.join(folder, file)
        image = cv2.imread(image_path)
        image = image / 255.0
        image = np.clip(image, 0, 1) # Clip the values between 0 and 1 
        cv2.imwrite(image_path, (image * 255).astype(np.uint8))  # Convert back to uint8 before saving


shuffleTrain()
normalizePixels('NormalizedImages')