from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load an official model
#model = YOLO('path/to/best.pt')  # load a custom model

# Predict with the model
results = model('/home/han/code/code/images/bus.jpg')  # predict on an image
#source = '/home/han/code/code/images/bus.jpg'
#results = model.predict(source, save=True, imgsz=320, conf=0.5)
boxes = results[0].boxes
box = boxes[0]  # returns one box
box.xyxy
