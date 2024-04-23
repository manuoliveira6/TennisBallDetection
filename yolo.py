from ultralytics import YOLO 

# This is our base model. The issue with this model is that it doesn't detect the tennis ball
# very well, so we will try to fine-tune this model to improve its performance. 
# Nonetheless, the model itself does detect the players accurately.
model = YOLO('yolov8x') 
# Selected the best Yolo model at that moment.

result = model.track('input/input_video.mp4',  save=True)
print(result)
