import face_recognition
import cv2
import os
import numpy as np

video_capture = cv2.VideoCapture(0)

known_person=[]

#For Images
known_image=[]

#ForEncoding
known_face_encoding=[]

for file in os.listdir("Imagefolder"):
    try:
        known_person.append(file.replace(".jpg", ""))
        file=os.path.join("Imagefolder", file)
        known_image = face_recognition.load_image_file(file)
        known_face_encoding.append(face_recognition.face_encodings(known_image)[0])
    except Exception as e:
        pass


face_locations = []
face_encodings = []
face_names = []
process_this_frame = 0

while True:
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color to RGB color
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame%5==0:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            match = face_recognition.compare_faces(known_face_encoding, face_encoding)
            matches=np.where(match)[0] #Checking which image is matched
            if len(matches)>0:
                name = str(known_person[matches[0]])
                face_names.append(name)
            else:
                face_names.append("Unknown")


    process_this_frame =  process_this_frame+1
    if process_this_frame>5:
        process_this_frame=0


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

