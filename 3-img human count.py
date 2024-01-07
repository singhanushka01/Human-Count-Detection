import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getUnconnectedOutLayersNames()

# Load COCO names file (contains class names for YOLO)
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Path to the directory containing images and subdirectories
img_dir = 'D:\human count'

# Get the list of image files in the directory and its subdirectories
image_files = [os.path.join(root, f) for root, dirs, files in os.walk(img_dir) for f in files if f.endswith(('.jpg', '.jpeg', '.png'))]

for image_path in image_files:
    # Read the image
    frame = cv2.imread(image_path)

    # Get image shape
    height, width, _ = frame.shape

    # Create blob from the image
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Run forward pass to get output from YOLO model
    detections = net.forward(layer_names)

    # Process YOLO detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5 and classes[class_id] == 'person':  # Assuming 'person' is the class for humans
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                # Calculate bounding box coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('frame', frame)

    # Print the processed image path
    print(f"Processed: {image_path}")

    # Add a delay (2000 milliseconds = 2 seconds)
    if cv2.waitKey(2000) & 0xFF == ord('q'):
        break

# When everything is done, close the display window
cv2.destroyAllWindows()
