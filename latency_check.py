import time
from ultralytics import YOLO
import cv2
 
model = YOLO("runs/detect/YOLO weld detection project/weld_yolo_training/weights/best.pt")          # or the full runs/... path
 
image = cv2.imread("datasets/train/images/5ee0159b-e989d42a-images_jpeg_jpg.rf.4243f81f9fb1db6bc02d9fb046d50935.jpg")
 
start = time.time()
results = model(image)
end = time.time()
 
latency = end - start
print("Inference Time:", latency, "seconds")
