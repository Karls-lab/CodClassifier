import os
import random
import cv2
"""
selects a random image that has 'aug' in the name in the folder images,
and displays the bounding box and the image 
"""

folder_path = "images"
file_list = [file for file in os.listdir(folder_path) if 'aug' in file]

if len(file_list) > 0:
    random_image = random.choice(file_list)
    image_path = os.path.join(folder_path, random_image)

    # Display the image
    image = cv2.imread(image_path)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Display the bounding box (assuming you have the bounding box coordinates)
    # bounding_box = ...
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imshow("Bounding Box", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
else:
    print("No images found with 'aug' in the name.")
