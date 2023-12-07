import cv2 
import sys
"""
Resize every image in the training.txt file to 512x512
"""
def resize_bounding_boxes(bboxes):
    # YOLO format: [label, x_center, y_center, width, height]
    yolo_bboxes = []
    for box in bboxes:
        label, x, y, w, h = box[0], box[1], box[2], box[3], box[4]
        x_center = x
        y_center = y
        width = w 
        height = h 
        yolo_bboxes.append([label, x_center, y_center, width, height])
    return yolo_bboxes

def resize_image(image_path, target_size):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, target_size)
    return resized_image

def resize_images(image_paths, folderName, target_size):
    for image_path in image_paths:
        img = cv2.imread(f"{folderName}/{image_path}")
        if type(img) == type(None): 
            print(f"image_path: {image_path} is None")
            continue
        if img.shape[0] == 512 or img.shape[1] == 512:
            print('image already resized')
            continue
        resized_img = resize_image(f"{folderName}/{image_path}", (512, 512))
        
        bounding_box = f"{folderName}/{image_path[:-4]}.txt"
        with open(bounding_box, "r") as f:
            bboxes = [] # Lists of Lists of bounding boxes
            lines = f.readlines()
            # print(f"lines: {lines}")
            for line in lines:
                # print(f"line: {line}")
                bounding_box = [float(x) for x in line.strip().split(" ")]
                bboxes.append(bounding_box)
        resized_bounding_box = resize_bounding_boxes(bboxes)

        # Save the resized image and bounding box 
        image_path = image_path
        print(f"image_path: {image_path}")
        cv2.imwrite(f"{'NormalizedImages'}/{image_path}", resized_img)
        with open(f"{'NormalizedImages'}/{image_path[:-4]}.txt", "w") as f:
            for bbox in resized_bounding_box:
                f.write(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}\n")

        # display image 
        # print("HERE")
        # show_image(resized_img, resized_bounding_box)

