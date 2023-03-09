import numpy as np
import face_recognition as fr
import cv2

video_capture = cv2.VideoCapture(0)

face_in_camera1 = fr.load_image_file("Shikhar.jpg")
face_encodings1 = fr.face_encodings(face_in_camera1)[0]

# face_in_camera2 = fr.load_image_file("bal.jpg")
# face_encodings2 = fr.face_encodings(face_in_camera2)[0]

face_in_camera3 = fr.load_image_file("Rudransh.jpg")
face_encodings3 = fr.face_encodings(face_in_camera3)[0]

face_in_camera4 = fr.load_image_file("Sachin.jpg")
face_encodings4 = fr.face_encodings(face_in_camera4)[0]

face_in_camera5 = fr.load_image_file("Obama.jpg")
face_encodings5 = fr.face_encodings(face_in_camera5)[0]

face_in_camera6 = fr.load_image_file("Utkarsh.jpg")
face_encodings6 = fr.face_encodings(face_in_camera6)[0]

face_in_camera7 = fr.load_image_file("Ujjwal_jain.jpg")
face_encodings7 = fr.face_encodings(face_in_camera7)[0]

known_face_encondings = [face_encodings1,face_encodings3,face_encodings4,face_encodings5,face_encodings6,face_encodings6]

known_face_names = ["Shikhar","Rudransh","Sachin","Barack Obama","Utkarsh","Ujjwal jain"]
# ,"Rudransh","Sachin","Barack Obama","Utkarsh","Ujjwal jain"]

while True: 
    ret, frame = video_capture.read()

    rgb_frame = frame[:, :, ::-1]

    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = fr.compare_faces(known_face_encondings, face_encoding)

        name = "Unknown"

        face_distances = fr.face_distance(known_face_encondings, face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left +6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()