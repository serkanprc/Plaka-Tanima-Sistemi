from ultralytics import YOLO

if __name__ == '__main__':
    # YOLOv8n modelini yükle
    model = YOLO('yolov8n.pt')

    # Modeli eğit
    results = model.train(
        data='C:/Users/serka/Desktop/deneme/data.yaml',  # Veri kümesi dosya yolu
        epochs=100,  # Eğitim epoch sayısı
        batch=16,     # Batch boyutu
        imgsz=1024,   # Görüntü boyutu
        device='cpu'     # GPU kullanımı (0) veya CPU için 'cpu'
    )
