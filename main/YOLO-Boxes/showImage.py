import matplotlib.pyplot as plt
import cv2
import random
import os


def show_image(image, bboxes):
    plt.imshow(image)
    print(f'Showing bboxes: {bboxes}')
    label, x_center, y_center, width, height = bboxes[0], bboxes[1], bboxes[2], bboxes[3], bboxes[4]
    x_min = int((x_center - width / 2) * image.shape[1])
    y_min = int((y_center - height / 2) * image.shape[0])
    x_max = int((x_center + width / 2) * image.shape[1])
    y_max = int((y_center + height / 2) * image.shape[0])
    rect = plt.Rectangle((x_min, y_min), (x_max - x_min), (y_max - y_min), linewidth=2, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
    plt.show()

# get random image from testImages folder
directory_path = 'NormalizedImages'
all_files = os.listdir(directory_path)
jpg_files = [file for file in all_files if file.endswith('.jpg')]
print(f"jpg_files: {jpg_files}")
random_file = random.choice(jpg_files)
print(f"random_file: {random_file}")

# get bounding box from text file
bounding_box = random_file[:-4] + '.txt'
bboxes = [] # Lists of Lists of bounding boxes
with open(f'{directory_path}/{bounding_box}', "r") as f:
    lines = f.readlines()
    # print(f"lines: {lines}")
    for line in lines:
        # print(f"line: {line}")
        bounding_box = [float(x) for x in line.strip().split(" ")]
        bboxes.append(bounding_box)

print(f"bboxes: {bboxes}")

img = cv2.imread(f"{directory_path}/{random_file}")
boxes = bboxes[0]
show_image(img, boxes)