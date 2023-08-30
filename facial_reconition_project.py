import numpy as np
import face_recognition as fr
import cv2
import os

# face detect through camera feed
video_capture = cv2.VideoCapture(0)

# data set creation
folder_path = "images/"
known_face_encodings = []
known_face_names = []

# data set population
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        face_in_camera = fr.load_image_file(file_path)
        known_face_names.append(filename[:-4])
        known_face_encodings.append(fr.face_encodings(face_in_camera)[0])

#feature extraction
while True: 
    ret, frame = video_capture.read()

    rgb_frame = frame[:, :, ::-1]

    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)


# comparing faces
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = fr.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        face_distances = fr.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        
#Data preprocessing

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left +6, bottom - 6), font, 1.0, (255, 255, 255), 1)

# Output

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()