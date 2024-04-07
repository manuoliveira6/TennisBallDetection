from ultralytics import YOLO 

# Este es nuestro modelo base. El problema con este modelo es que no detecta demasiado bien
# la bola de tenis, por lo que intentaremos finetunnear este modelo para mejorar su rendimiento.
# Igualmente, el modelo en si, si que detecta bien los jugadores.
model = YOLO('yolov8x')

result = model.track('input/input_video.mp4',  save=True)
print(result)
print("boxes:")
for box in result[0].boxes:
    print(box)