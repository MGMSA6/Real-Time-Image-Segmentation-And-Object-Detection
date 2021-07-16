# Import libraries
import cv2
from pixellib.instance import instance_segmentation

# Download model
# https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.2/mask_rcnn_coco.h5


segment_image = instance_segmentation()
segment_image.load_model("mask_rcnn_coco.h5")
camera = cv2.VideoCapture(0)

while camera.isOpened():
    res, frame = camera.read()
    ### Apply Segmentation
    result = segment_image.segmentFrame(frame, show_bboxes=True)
    image = result[1]
    cv2.imshow('Image Segmentation', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
