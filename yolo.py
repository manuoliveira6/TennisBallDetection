from ultralytics import YOLO 

# This is our base model. The issue with this model is that it doesn't detect the tennis ball
# very well, so we will try to fine-tune this model to improve its performance. 
# Nonetheless, the model itself does detect the players accurately.
model = YOLO('yolov8x') 
# Selected the last Yolo model at that moment.

# The model was trained in google colab to use GPU and to have a better performance.
# Check training folder for more info.

# Model post-finetunning.
# model = YOLO('finetunned_models/best.pt')

result = model.track('input/input_video.mp4',  save=True)
print(result)
