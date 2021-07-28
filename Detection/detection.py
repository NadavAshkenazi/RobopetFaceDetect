import pandas as pd
from time import time
from typing import List
from cv2 import dnn, resize
import cv2
from imutils.video import VideoStream
import imutils
import os

frame_size = 300
process_every = 2


class FaceDetector:
    def __init__(self, prototxt="deploy.prototxt.txt", model="res10_300x300_ssd_iter_140000.caffemodel", tolerance: float = 0.8):
        dirname = os.path.dirname(__file__)
        prototxt = os.path.join(dirname, prototxt)
        model = os.path.join(dirname, model)
        self.detector = dnn.readNetFromCaffe(prototxt, model)
        self.target_size = (frame_size, frame_size)
        self.tolerance = tolerance

    def detect_picture(self, frame) -> List:
        small_frame = resize(frame, self.target_size)
        image_blob = dnn.blobFromImage(image=small_frame)
        self.detector.setInput(image_blob)
        detections = self.detector.forward()
        detections_df = pd.DataFrame(detections[0][0],
                                     columns=["img_id", "is_face", "confidence", "left", "top", "right", "bottom"])
        detections_df = detections_df[detections_df["is_face"] == 1]  # 0: background, 1: face, 2: confidence
        detections_df = detections_df[detections_df["confidence"] >= self.tolerance]
        aspect_ratio_x, aspect_ratio_y = self.get_ratio(frame)
        faces = []

        for i, instance in detections_df.iterrows():
            left = int(instance["left"] * frame_size * aspect_ratio_x)
            bottom = int(instance["bottom"] * frame_size * aspect_ratio_y)
            right = int(instance["right"] * frame_size * aspect_ratio_x)
            top = int(instance["top"] * frame_size * aspect_ratio_y)

            faces.append((top, right, bottom, left))

        return faces

    def get_ratio(self, frame):
        orig_size = frame.shape
        x_rat = (orig_size[1] / self.target_size[1])
        y_rat = (orig_size[0] / self.target_size[0])
        return x_rat, y_rat

    def detect_video_from_camera(self):
        vs = VideoStream(src=0).start()
        count = 0
        # loop over the frames from the video stream
        while True:
            if count % process_every == 0:
                # grab the frame from the threaded video stream and resize it
                # to have a maximum width of 400 pixels
                frame = vs.read()
                frame = imutils.resize(frame, width=400)
                faces = self.detect_picture(frame)
                for top, right, bottom, left in faces:
                    # draw the bounding box of the face along with the associated
                    # probability
                    cv2.rectangle(frame, (left, bottom), (right, top), (0, 255, 0), 2)
                    # font = cv2.FONT_HERSHEY_DUPLEX
                    # text = str(left) + "," + str(right) + "," + str(bottom) + "," + str(top)
                    # cv2.putText(frame, text, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

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

    def get_face_location(self, timeout=-1):
        vs = VideoStream(src=0).start()
        count_frames = 0
        faces = None
        exit_on_to = timeout > 0
        start_time = time()
        # loop over the frames from the video stream
        while True:
            if exit_on_to and time() - start_time > timeout:
                print(time() - start_time)
                cv2.destroyAllWindows()
                vs.stop()
                return None

            if count_frames % process_every == 0:
                # grab the frame from the threaded video stream and resize it
                # to have a maximum width of 400 pixels
                frame = vs.read()
                frame = cv2.flip(frame, 0)
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

            count_frames += 1
        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()

        top, right, bottom, left = faces[0]
        return (left + right) / (2 * 400), (bottom + top) / (2 * 300)
        # loc_temp = faces[0]
        # loc = [loc_temp[0] / 300, loc_temp[1] / 400, loc_temp[2] / 300, loc_temp[3] / 400]
        # return tuple(loc)


def main():
    detector = FaceDetector()
    print(detector.get_face_location())
    # detector.detect_video_from_camera()


if __name__ == "__main__":
    main()
