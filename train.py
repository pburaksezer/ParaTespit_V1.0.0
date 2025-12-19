"""
YOLOv8 Classification model eğitim scripti.
"""

from ultralytics import YOLO
import os

def train_model(data_dir='dataset', epochs=100, imgsz=640, batch=8, model_size='n', workers=4):
    """
    YOLOv8 Classification modeli eğitir.
    
    Args:
        data_dir: Veri klasörü (train ve val alt klasörleri içermeli)
        epochs: Eğitim epoch sayısı
        imgsz: Görüntü boyutu
        batch: Batch size
        model_size: Model boyutu ('n', 's', 'm', 'l', 'x')
    """
    
    # Model oluştur (YOLOv8 classification)
    model = YOLO(f'yolov8{model_size}-cls.pt')  # Pre-trained model
    
    # Eğitim parametreleri
    results = model.train(
        data=data_dir,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        name='banknot_classifier',
        project='runs/classify',
        patience=20,  # Early stopping patience
        save=True,
        val=True,
        plots=True,
        verbose=True
    )
    
    print("\nEğitim tamamlandı!")
    print(f"Model kaydedildi: runs/classify/banknot_classifier/weights/best.pt")
    
    return results

if __name__ == '__main__':
    # Veri klasörünün varlığını kontrol et
    if not os.path.exists('dataset'):
        print("HATA: 'dataset' klasörü bulunamadı!")
        print("Önce 'python prepare_data.py' komutunu çalıştırın.")
        exit(1)
    
    # Eğitimi başlat
    train_model(
        data_dir='dataset',
        epochs=100,
        imgsz=640,      # daha küçük çözünürlük
        batch=8,        # VRAM için azaltıldı
        model_size='n', # 'nano' model
        workers=4       # DataLoader worker sayısı
    )

