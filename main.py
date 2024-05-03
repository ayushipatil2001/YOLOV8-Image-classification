from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

model.train(data='C://Users//ayush//OneDrive//Desktop//yolo//dataset',
            epochs=10, imgsz=244)

model.save('custom_yolov8_model.pt')


