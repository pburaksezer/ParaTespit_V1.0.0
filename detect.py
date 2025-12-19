"""
YOLOv8 Classification kullanarak banknot tespit scripti.
Resim, video veya webcam üzerinde banknot tespiti yapar.
"""

from ultralytics import YOLO
import cv2
import argparse
from pathlib import Path
import os

# Banknot sınıfları
CLASS_NAMES = {
    0: '5 TL',
    1: '10 TL',
    2: '20 TL',
    3: '50 TL',
    4: '100 TL',
    5: '200 TL'
}

def detect_image(model_path, image_path, conf_threshold=0.25, save=True):
    """
    Tek bir resim üzerinde banknot tespiti yapar.
    
    Args:
        model_path: Eğitilmiş model yolu
        image_path: Tespit edilecek resim yolu
        conf_threshold: Güven eşiği
        save: Sonuçları kaydet
    """
    # Modeli yükle
    model = YOLO(model_path)
    
    # Tespit yap
    results = model(image_path, conf=conf_threshold)
    
    # Sonuçları göster
    for result in results:
        # En yüksek güven skoruna sahip sınıfı al
        probs = result.probs
        top1_idx = probs.top1
        top1_conf = probs.top1conf.item()
        class_name = CLASS_NAMES.get(top1_idx, f'Class {top1_idx}')
        
        print(f"\nResim: {image_path}")
        print(f"Tespit Edilen: {class_name}")
        print(f"Güven Skoru: {top1_conf:.2%}")
        
        # Tüm sınıfların skorlarını göster
        print("\nTüm Sınıf Skorları:")
        for idx, conf in enumerate(probs.data):
            print(f"  {CLASS_NAMES.get(idx, f'Class {idx}')}: {conf:.2%}")
        
        # Görselleştirme
        if save:
            output_path = f"detected_{Path(image_path).name}"
            annotated_img = result.plot()
            cv2.imwrite(output_path, annotated_img)
            print(f"\nSonuç kaydedildi: {output_path}")
    
    return results

def detect_video(model_path, video_path, conf_threshold=0.25, save=True):
    """
    Video üzerinde banknot tespiti yapar.
    
    Args:
        model_path: Eğitilmiş model yolu
        video_path: Tespit edilecek video yolu
        conf_threshold: Güven eşiği
        save: Sonuçları kaydet
    """
    # Modeli yükle
    model = YOLO(model_path)
    
    # Video tespiti
    results = model(video_path, conf=conf_threshold, save=save)
    
    print(f"\nVideo işlendi: {video_path}")
    if save:
        print("Sonuçlar kaydedildi.")
    
    return results

def detect_webcam(model_path, conf_threshold=0.25):
    """
    Webcam üzerinden canlı banknot tespiti yapar.
    
    Args:
        model_path: Eğitilmiş model yolu
        conf_threshold: Güven eşiği
    """
    # Modeli yükle
    model = YOLO(model_path)
    
    # Webcam'i aç
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("HATA: Webcam açılamadı!")
        return
    
    print("Webcam açıldı. Çıkmak için 'q' tuşuna basın.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Tespit yap
        results = model(frame, conf=conf_threshold, verbose=False)
        
        # Sonuçları göster
        for result in results:
            probs = result.probs
            top1_idx = probs.top1
            top1_conf = probs.top1conf.item()
            class_name = CLASS_NAMES.get(top1_idx, f'Class {top1_idx}')
            
            # Görüntü üzerine yazı ekle
            annotated_frame = result.plot()
            cv2.putText(annotated_frame, 
                       f"{class_name}: {top1_conf:.2%}", 
                       (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       1, 
                       (0, 255, 0), 
                       2)
            
            cv2.imshow('Banknot Tespit', annotated_frame)
        
        # 'q' tuşuna basıldığında çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Banknot Tespit Uygulaması')
    parser.add_argument('--model', type=str, default='runs/classify/banknot_classifier/weights/best.pt',
                        help='Eğitilmiş model yolu')
    parser.add_argument('--source', type=str, default=None,
                        help='Kaynak (resim, video yolu veya "webcam")')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Güven eşiği (0-1 arası)')
    parser.add_argument('--save', action='store_true',
                        help='Sonuçları kaydet')
    
    args = parser.parse_args()
    
    # Model dosyasının varlığını kontrol et
    if not os.path.exists(args.model):
        print(f"HATA: Model dosyası bulunamadı: {args.model}")
        print("Önce modeli eğitin: python train.py")
        return
    
    # Kaynak belirtilmemişse, varsayılan olarak test resmi iste
    if args.source is None:
        print("Kaynak belirtilmedi. Lütfen bir resim yolu girin veya 'webcam' yazın.")
        args.source = input("Resim yolu veya 'webcam': ").strip()
    
    # Kaynak tipine göre işlem yap
    if args.source.lower() == 'webcam':
        detect_webcam(args.model, args.conf)
    elif os.path.isfile(args.source):
        # Dosya uzantısına göre resim veya video
        ext = Path(args.source).suffix.lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            detect_image(args.model, args.source, args.conf, args.save)
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            detect_video(args.model, args.source, args.conf, args.save)
        else:
            print(f"Desteklenmeyen dosya formatı: {ext}")
    elif os.path.isdir(args.source):
        # Klasör içindeki tüm resimleri işle
        print(f"Klasör işleniyor: {args.source}")
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(args.source).glob(f'*{ext}'))
            image_files.extend(Path(args.source).glob(f'*{ext.upper()}'))
        
        for img_path in image_files:
            detect_image(args.model, str(img_path), args.conf, args.save)
    else:
        print(f"HATA: Geçersiz kaynak: {args.source}")

if __name__ == '__main__':
    main()









