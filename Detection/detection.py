import pandas as pd
from typing import List
from cv2 import dnn, resize
import cv2
from imutils.video import VideoStream
import imutils

frame_size = 300
frame_count = 2


class FaceDetector:
    def __init__(self, prototxt="deploy.prototxt.txt", model="res10_300x300_ssd_iter_140000.caffemodel", tolerance: float = 0.8):
        self.detector = dnn.readNetFromCaffe(prototxt, model)
        self.target_size = (frame_size, frame_size)
        self.tolerance = tolerance

    def detect_picture(self, frame) -> List:
        small_frame = resize(frame, self.target_size)
        imageBlob = dnn.blobFromImage(image=small_frame)
        self.detector.setInput(imageBlob)
        detections = self.detector.forward()
        detections_df = pd.DataFrame(detections[0][0],
                                     columns=["img_id", "is_face", "confidence", "left", "top", "right", "bottom"])
        detections_df = detections_df[detections_df["is_face"] == 1]  # 0: background, 1: face, 2: confidence
        detections_df = detections_df[detections_df["confidence"] >= self.tolerance]
        aspect_ratio_x, aspect_ratio_y = self.calc_resize_ratio(frame)
        faces = []

        for i, instance in detections_df.iterrows():
            left = int(instance["left"] * frame_size * aspect_ratio_x)
            bottom = int(instance["bottom"] * frame_size * aspect_ratio_y)
            right = int(instance["right"] * frame_size * aspect_ratio_x)
            top = int(instance["top"] * frame_size * aspect_ratio_y)

            faces.append((top, right, bottom, left))

        return faces

    def calc_resize_ratio(self, frame):
        original_size = frame.shape
        aspect_ratio_x = (original_size[1] / self.target_size[1])
        aspect_ratio_y = (original_size[0] / self.target_size[0])
        return aspect_ratio_x, aspect_ratio_y

    def detect_video_from_camera(self):
        vs = VideoStream(src=0).start()
        count = 0
        # loop over the frames from the video stream
        while True:
            if count % frame_count == 0:
                # grab the frame from the threaded video stream and resize it
                # to have a maximum width of 400 pixels
                frame = vs.read()
                frame = imutils.resize(frame, width=400)
                faces = self.detect_picture(frame)
                for top, right, bottom, left in faces:
                    # draw the bounding box of the face along with the associated
                    # probability
                    cv2.rectangle(frame, (left, bottom), (right, top), (0, 255, 0), 2)

                # show the output frame
                cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

            count += 1
        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()

    def detect_face_location(self):
        vs = VideoStream(src=0).start()
        count = 0
        faces = None
        # loop over the frames from the video stream
        while True:
            if count % frame_count == 0:
                # grab the frame from the threaded video stream and resize it
                # to have a maximum width of 400 pixels
                frame = vs.read()
                frame = imutils.resize(frame, width=400)
                faces = self.detect_picture(frame)
                if faces:
                    break
                # show the output frame
                cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

            count += 1
        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()

        loc = [face / 400 for face in faces[0]]
        return loc


def main():
    detector = FaceDetector()
    print(detector.detect_face_location())
    # detector.detect_video_from_camera()

if __name__ == "__main__":
    main()
