# Banknot Tespit Uygulaması (YOLOv8)

Bu uygulama, YOLOv8 Classification kullanarak Türk Lirası banknotlarını (5, 10, 20, 50, 100, 200 TL) tespit eder.

## Özellikler

- 6 farklı banknot değerini tespit edebilir (5, 10, 20, 50, 100, 200 TL)
- Resim, video veya webcam üzerinde çalışabilir
- YOLOv8 Classification modeli kullanır
- Yüksek doğruluk oranı
- **Modern ve kullanıcı dostu grafik arayüz (GUI)**

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

## Kullanım

### 1. Veri Hazırlama

Önce verilerinizi YOLOv8 formatına dönüştürün:
```bash
python prepare_data.py
```

Bu script, `train.txt` ve `validation.txt` dosyalarını kullanarak `dataset` klasörü oluşturur.

### 2. Model Eğitimi

Modeli eğitmek için:
```bash
python train.py
```

Eğitim parametrelerini değiştirmek için `train.py` dosyasını düzenleyebilirsiniz:
- `epochs`: Eğitim epoch sayısı (varsayılan: 100)
- `imgsz`: Görüntü boyutu (varsayılan: 640)
- `batch`: Batch size (varsayılan: 16)
- `model_size`: Model boyutu ('n', 's', 'm', 'l', 'x') (varsayılan: 'n')

Eğitilmiş model `runs/classify/banknot_classifier/weights/best.pt` konumuna kaydedilir.

### 3. Tespit (Detection)

#### Grafik Arayüz (GUI) ile Kullanım (Önerilen):

En kolay kullanım için grafik arayüzü kullanabilirsiniz:
```bash
python gui.py
```

GUI özellikleri:
- 📷 Resim seçip tespit etme
- 🎥 Video seçip işleme
- 📹 Webcam ile canlı tespit
- Model seçimi
- Güven eşiği ayarlama
- Sonuçları görselleştirme
- Tüm sınıf skorlarını görüntüleme

#### Komut Satırı ile Kullanım:

##### Tek bir resim üzerinde tespit:
```bash
python detect.py --source path/to/image.png --model runs/classify/banknot_classifier/weights/best.pt
```

##### Video üzerinde tespit:
```bash
python detect.py --source path/to/video.mp4 --model runs/classify/banknot_classifier/weights/best.pt --save
```

##### Webcam ile canlı tespit:
```bash
python detect.py --source webcam --model runs/classify/banknot_classifier/weights/best.pt
```

##### Klasör içindeki tüm resimleri işle:
```bash
python detect.py --source path/to/images_folder --model runs/classify/banknot_classifier/weights/best.pt --save
```

### Parametreler

- `--model`: Eğitilmiş model yolu (varsayılan: `runs/classify/banknot_classifier/weights/best.pt`)
- `--source`: Kaynak (resim, video yolu, klasör yolu veya "webcam")
- `--conf`: Güven eşiği 0-1 arası (varsayılan: 0.25)
- `--save`: Sonuçları kaydet

## Klasör Yapısı

```
.
├── 5/              # 5 TL banknot resimleri
├── 10/             # 10 TL banknot resimleri
├── 20/             # 20 TL banknot resimleri
├── 50/             # 50 TL banknot resimleri
├── 100/            # 100 TL banknot resimleri
├── 200/            # 200 TL banknot resimleri
├── train.txt       # Eğitim verileri listesi
├── validation.txt  # Validasyon verileri listesi
├── prepare_data.py # Veri hazırlama scripti
├── train.py        # Model eğitim scripti
├── detect.py       # Tespit scripti (komut satırı)
├── gui.py          # Grafik arayüz (GUI)
├── requirements.txt
└── README.md
```

## Notlar

- İlk eğitim GPU kullanıyorsanız daha hızlı olacaktır
- Model boyutunu (`model_size`) ihtiyacınıza göre ayarlayabilirsiniz:
  - `n`: Nano (en hızlı, en küçük)
  - `s`: Small
  - `m`: Medium
  - `l`: Large
  - `x`: XLarge (en yavaş, en büyük)
- Eğitim sırasında `runs/classify/banknot_classifier/` klasöründe sonuçlar ve grafikler kaydedilir

## Sorun Giderme

- **Model bulunamadı hatası**: Önce `python train.py` komutunu çalıştırarak modeli eğitin
- **Dataset bulunamadı hatası**: Önce `python prepare_data.py` komutunu çalıştırarak veriyi hazırlayın
- **CUDA hatası**: CPU kullanmak için PyTorch'u CPU versiyonu ile yükleyin

## Lisans

Bu proje eğitim amaçlıdır.




