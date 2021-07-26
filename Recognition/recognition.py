import face_recognition
import cv2
import numpy as np
import os
import pickle

default_dataset_name = 'dataset_faces.dat'
default_img_path = 'img'
num_match = 10
process_every = 2


class FaceRecogniser:
    def __init__(self, img_path=default_img_path, dataset_name=default_dataset_name):
        dirname = os.path.dirname(__file__)
        self.img_path = os.path.join(dirname, img_path)
        self.dataset_path = os.path.join(dirname, dataset_name)
        self.known_face_names = []
        self.known_face_encodings = []

    def create_export_embeddings(self):
        bases = os.listdir(self.img_path)
        filenames = [os.path.join(self.img_path, base) for base in bases]
        self.known_face_names = [os.path.splitext(base)[0] for base in bases]
        all_face_encodings = {}
        for i, name in enumerate(self.known_face_names):
            img = face_recognition.load_image_file(filenames[i])
            all_face_encodings[name] = face_recognition.face_encodings(img)[0]

        with open(self.dataset_path, 'wb') as f:
            pickle.dump(all_face_encodings, f)

        self.known_face_encodings = np.array(list(all_face_encodings.values()))

    def load_embeddings(self):
        with open(self.dataset_path, 'rb') as f:
            all_face_encodings = pickle.load(f)

        self.known_face_names = list(all_face_encodings.keys())
        self.known_face_encodings = np.array(list(all_face_encodings.values()))

    def rec_video_from_camera(self):
        video_capture = cv2.VideoCapture(0)
        # Initialize some variables
        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True

        face_count = dict()
        count_frames = 0

        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"

                    # # If a match was found in known_face_encodings, just use the first one.
                    # if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = known_face_names[first_match_index]

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]

                    face_names.append(name)

                    face_count[name] = face_count.get(name, 0) + 1
                    if face_count[name] >= num_match:
                        return name

            process_this_frame = (count_frames % process_every == 0)
            count_frames += 1

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()

    def auth_user(self, username):
        return username == self.rec_video_from_camera()


def main():
    recogniser = FaceRecogniser()
    # recogniser.create_export_embeddings()
    recogniser.load_embeddings()
    print(recogniser.rec_video_from_camera())
    print(recogniser.auth_user('nathan'))


if __name__ == '__main__':
    main()
