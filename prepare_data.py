"""
YOLOv8 Classification için veri hazırlama scripti.
train.txt ve validation.txt dosyalarını kullanarak YOLOv8 formatına dönüştürür.
"""

import os
import shutil
from pathlib import Path

def prepare_yolo_dataset(train_txt='train.txt', val_txt='validation.txt', 
                         output_dir='dataset', source_dir='.'):
    """
    train.txt ve validation.txt dosyalarını kullanarak YOLOv8 classification formatına dönüştürür.
    
    Args:
        train_txt: Eğitim dosyalarının listesi
        val_txt: Validasyon dosyalarının listesi
        output_dir: Çıktı klasörü
        source_dir: Kaynak klasör (resimlerin bulunduğu yer)
    """
    
    # Çıktı klasörlerini oluştur
    train_dir = Path(output_dir) / 'train'
    val_dir = Path(output_dir) / 'val'
    
    # Sınıf isimleri (banknot değerleri)
    classes = ['5', '10', '20', '50', '100', '200']
    
    # Her sınıf için klasör oluştur
    for cls in classes:
        (train_dir / cls).mkdir(parents=True, exist_ok=True)
        (val_dir / cls).mkdir(parents=True, exist_ok=True)
    
    # Eğitim verilerini kopyala
    print("Eğitim verileri hazırlanıyor...")
    with open(train_txt, 'r', encoding='utf-8') as f:
        train_files = [line.strip() for line in f.readlines() if line.strip()]
    
    for file_path in train_files:
        # Dosya yolundan sınıfı çıkar (örn: "5/image.png" -> "5")
        class_name = file_path.split('/')[0]
        
        if class_name in classes:
            src = Path(source_dir) / file_path
            dst = train_dir / class_name / Path(file_path).name
            
            if src.exists():
                shutil.copy2(src, dst)
    
    # Validasyon verilerini kopyala
    print("Validasyon verileri hazırlanıyor...")
    with open(val_txt, 'r', encoding='utf-8') as f:
        val_files = [line.strip() for line in f.readlines() if line.strip()]
    
    for file_path in val_files:
        # Dosya yolundan sınıfı çıkar
        class_name = file_path.split('/')[0]
        
        if class_name in classes:
            src = Path(source_dir) / file_path
            dst = val_dir / class_name / Path(file_path).name
            
            if src.exists():
                shutil.copy2(src, dst)
    
    print(f"\nVeri hazırlama tamamlandı!")
    print(f"Eğitim verileri: {train_dir}")
    print(f"Validasyon verileri: {val_dir}")
    
    # İstatistikler
    print("\nİstatistikler:")
    for cls in classes:
        train_count = len(list((train_dir / cls).glob('*')))
        val_count = len(list((val_dir / cls).glob('*')))
        print(f"  {cls} TL: Eğitim={train_count}, Validasyon={val_count}")

if __name__ == '__main__':
    prepare_yolo_dataset()

