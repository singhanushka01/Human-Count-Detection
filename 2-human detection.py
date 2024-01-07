import cv2
import numpy as np

cap = cv2.VideoCapture('hellwork.avi')
human_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    humans = human_cascade.detectMultiScale(gray, 1.9, 1)

    # Apply non-maximum suppression manually
    indices = []
    for i in range(len(humans)):
        for j in range(i + 1, len(humans)):
            # Check if the intersection over union (IoU) is greater than the threshold
            intersection = (
                max(humans[i][0], humans[j][0]),
                max(humans[i][1], humans[j][1]),
                min(humans[i][0] + humans[i][2], humans[j][0] + humans[j][2]),
                min(humans[i][1] + humans[i][3], humans[j][1] + humans[j][3])
            )
            area_intersection = max(0, intersection[2] - intersection[0]) * max(0, intersection[3] - intersection[1])
            area_union = (humans[i][2] * humans[i][3]) + (humans[j][2] * humans[j][3]) - area_intersection
            iou = area_intersection / area_union

            # If IoU is less than the threshold, keep the bounding box
            if iou < 0.4:
                indices.append(i)

    # Display the resulting frame
    for i in indices:
        (x, y, w, h) = humans[i]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    num_humans = len(indices)
    cv2.putText(frame, f'Number of Humans: {num_humans}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
