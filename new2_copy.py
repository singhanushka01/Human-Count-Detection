import cv2
import numpy as np

# Load the pre-trained YOLOv8 model
net = cv2.dnn.readNet('yolov8.weights', 'yolov8.cfg')

# Load COCO class labels (YOLOv8 has 80 classes)
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Load image
image = cv2.imread('image2.jpeg')

# Get image dimensions
height, width = image.shape[:2]

# Preprocess image for YOLO
blob = cv2.dnn.blobFromImage(image, 1/255.0, (608, 608), swapRB=True, crop=False)
net.setInput(blob)

# Get output layer names
output_layer_names = net.getUnconnectedOutLayersNames()

# Run forward pass and get output
outputs = net.forward(output_layer_names)

# Initialize count for people
people_count = 0

# Process each output layer
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # Check if the detected object is a person (class_id for person is 0)
        if confidence > 0.5 and class_id == 0:
            # Increment people count
            people_count += 1

            # Get bounding box coordinates
            center_x, center_y, w, h = (detection[:4] * np.array([width, height, width, height])).astype('int')
            x, y = int(center_x - w/2), int(center_y - h/2)

            # Draw bounding box around the person
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the result
cv2.imshow('Person Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the result
print(f'Number of people detected: {people_count}')
