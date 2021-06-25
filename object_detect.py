import numpy as np
import pathlib
import tensorflow as tf
import cv2

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util

PATH_TO_LABELS = 'toy/annotations/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# PATH_TO_TEST_IMAGES_DIR = pathlib.Path('toy/images/test')
# TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    # print(output_dict['detection_boxes'][0])
    return output_dict['detection_boxes'][0]



detect_fn = tf.saved_model.load("toy/exported-models/my_model/saved_model")

cap = cv2.VideoCapture(0)
cap.set(3, 640) # set Width
cap.set(4, 480) # set Height
while True:
    ret, img = cap.read()
    h_i, w_i, c = img.shape
    y1, x1, y2, x2 = run_inference_for_single_image(detect_fn, np.array(img))
    print(x1)
    print(x2)
    print(y1)
    print(y2)

    y1 = int(y1 * h_i)
    y2 = int(y2 * h_i)
    x1 = int(x1 * w_i)
    x2 = int(x2 * w_i)

    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # roi_color = img[y:y+h, x:x+w]

    cv2.imshow('video', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cap.release()
cv2.destroyAllWindows()
