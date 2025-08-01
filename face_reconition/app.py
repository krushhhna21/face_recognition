import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime

# Load known faces
path = 'static/faces'
images = []
names = []

for filename in os.listdir(path):
    img_path = f"{path}/{filename}"
    img = face_recognition.load_image_file(img_path)
    encoding = face_recognition.face_encodings(img)
    
    if encoding:  # Ensure a face was found
        images.append(encoding[0])
        names.append(os.path.splitext(filename)[0])

# Webcam setup
cap = cv2.VideoCapture(0)

def mark_attendance(name):
    with open('attendance.csv', 'a') as f:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f'{name},{now}\n')
        print(f"[✓] Attendance marked for {name}")

marked_names = []

print("Press 'q' to quit.\nLook into the camera...")

while True:
    success, frame = cap.read()
    if not success:
        break

    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces_cur_frame = face_recognition.face_locations(rgb_small)
    encodes_cur_frame = face_recognition.face_encodings(rgb_small, faces_cur_frame)

    for encodeFace, faceLoc in zip(encodes_cur_frame, faces_cur_frame):
        matches = face_recognition.compare_faces(images, encodeFace)
        face_distances = face_recognition.face_distance(images, encodeFace)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = names[best_match_index]
            if name not in marked_names:
                mark_attendance(name)
                marked_names.append(name)

            y1, x2, y2, x1 = [val * 4 for val in faceLoc]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow('Face Attendance System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
