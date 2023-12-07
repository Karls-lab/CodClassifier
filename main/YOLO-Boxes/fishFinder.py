"""
This python module will find the fish in the image, crop it, and give save it as a file

install open cv: pip3 install opencv-python
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
# net = cv2.dnn.readNet("backup/fishFinderV1.weights", "Setup/yolov3.cfg")  # Update the paths to your downloaded weights and configuration file
net = cv2.dnn.readNet("backup/yolov3_3000.weights", "Setup/yolov3.cfg")  # Update the paths to your downloaded weights and configuration file
classes = ['fish']

# Function to get output layers from YOLO
layer_names = net.getLayerNames()
unconnected_layers = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in unconnected_layers]

# Load random jpg image from the images folder
jpg_files = glob.glob(os.path.join("NormalizedImages", "*.jpg"))
random_file = random.choice(jpg_files)
img = cv2.imread(random_file)
img = cv2.resize(img, None, fx=1, fy=1)  # Adjust the resizing factor as needed
height, width, _ = img.shape  # The third value in the shape represents the number of color channels (e.g., RGB has 3 channels)

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (512, 512), (0, 0, 0), False, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)
print(f"Image shape: {img.shape}")
print(f"Sample pixel values: {img[0, 0]}")

# show the image
import matplotlib.pyplot as plt
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


# Define your three classes
classes = ['fish']

# Initialize lists to store information about detected objects
confidences = []
boxes = []
max_confidence_index = -1
max_confidence = 0.0

# Iterate through the output layers and detections
for out in outs:
    for detection in out:
        confidence = detection[5:]
        if confidence > 0:  # Adjust confidence threshold as needed
            center_x = int(detection[1] * width)
            center_y = int(detection[2] * height)
            w = int(detection[3] * width)
            h = int(detection[4] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Check if the current detection has a higher confidence
            if confidence > max_confidence:
                max_confidence = confidence
                max_confidence_index = len(boxes)  # Index in the 'boxes' list

            # Append information about the detected object to the lists
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))

# Apply non-maximum suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.4)

# Output directory for saving cropped images
output_directory = "output"
os.makedirs(output_directory, exist_ok=True)

# Loop over the detected objects and save cropped images
print(f"Number of objects detected: {len(indexes)}")
if len(indexes) == 0:
    print("No objects detected.")
    sys.exit(0)

# Displaying the results
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(classes), 3))

for i in indexes.flatten():
    x, y, w, h = boxes[i]
    label = 'fish'
    confidence = str(round(confidences[i], 2))
    color = (0, 0, 255)  # You can assign different colors for each class if needed
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, color, 2)

# Show the final image
import matplotlib.pyplot as plt
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
