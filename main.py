import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path
import time

cascade_path = 'Cascades/haarcascade_frontalface_default.xml'
dataset_dir = 'dataset'
trainer_dir = 'trainer'


def extract_faces(face_id, video):
    face_detector = cv2.CascadeClassifier(cascade_path)

    print("\n [INFO] Initializing face capture.")

    vidcap = cv2.VideoCapture(video)
    success, img = vidcap.read()
    count = 0

    Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    while success:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1

            # Save the captured image into the datasets folder
            cv2.imwrite(dataset_dir + "/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

            cv2.imshow('image', img)

        success, img = vidcap.read()

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    # cam.release()
    # cv2.destroyAllWindows()
    return count


# function to get the images and label data
def getImagesAndLabels(path):
    detector = cv2.CascadeClassifier(cascade_path)
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
        img_numpy = np.array(PIL_img, 'uint8')

        if imagePath.startswith('subject'):
            id = 0
        else:
            id = int(os.path.split(imagePath)[-1].split(".")[1])

        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)

    return faceSamples, ids


def static_train():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces, ids = getImagesAndLabels(dataset_dir)
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    Path(trainer_dir).mkdir(parents=True, exist_ok=True)
    recognizer.write(trainer_dir + '/trainer.yml')  # recognizer.save() worked on Mac, but not on Pi

    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))


def train(face_id, video):
    count = extract_faces(face_id, video)
    static_train()
    return count


def face_recgonize(face_id):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    faceCascade = cv2.CascadeClassifier(cascade_path)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # names related to ids: example ==> Marcelo: id=1,  etc
    names = ['None', 'Nathan', 'Michal', 'Realshit', 'Z', 'W']

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video widht
    cam.set(4, 480)  # set video height

    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    count = 0
    recognized = False

    while not recognized:
        print(count)
        count += 1
        if count > 100:
            break

        ret, img = cam.read()
        # img = cv2.flip(img, -1)  # Flip vertically

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            if id == int(face_id):
                recognized = True
                break

            # Check if confidence is less them 100 ==> "0" is perfect match
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)


        cv2.imshow('camera', img)

        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break


    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()

    return recognized


def getLocation(timeout=10):
    faceCascade = cv2.CascadeClassifier(cascade_path)
    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video widht
    cam.set(4, 480)  # set video height

    # Define min window size to be recognized as a face
    minW = 0.05 * cam.get(3)
    minH = 0.05 * cam.get(4)

    start_time = time.time()

    while time.time() - start_time < timeout:
        ret, img = cam.read()
        img = cv2.flip(img, -1)  # Flip vertically

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        h_i, w_i, c = img.shape

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            print(f"height is {h_i}, width is {w_i}, location is: ({x + w / 2}, {y + h / 2})")
            return (x + w / 2) / w_i, (y + h / 2) / h_i

        cv2.imshow('camera', img)

        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break

    return None

def getLocationHostile(timeout=10):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    faceCascade = cv2.CascadeClassifier(cascade_path)

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video widht
    cam.set(4, 480)  # set video height

    # Define min window size to be recognized as a face
    minW = 0.05 * cam.get(3)
    minH = 0.05 * cam.get(4)

    start_time = time.time()

    while time.time() - start_time < timeout:
        ret, img = cam.read()
        img = cv2.flip(img, -1)  # Flip vertically

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        h_i, w_i, c = img.shape

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if confidence > 100:
                print(f"height is {h_i}, width is {w_i}, location is: ({x + w / 2}, {y + h / 2})")
                return (x + w / 2) / w_i, (y + h / 2) / h_i

        cv2.imshow('camera', img)

        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break

    return None


def main():
    #face_id = input('\n enter user id and press <return> ==>  ')
    # video = input('\n enter video path and press <return> ==>  ')
    # video = 'nathan.mp4'
    # count = train(face_id, video)
    # print("\nDetector captured " + str(count) + " faces")
    #is_rec = face_recgonize(face_id)
    print(getLocation())


if __name__ == "__main__":
    main()
