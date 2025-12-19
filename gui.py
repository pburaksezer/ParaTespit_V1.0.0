"""
Banknot Tespit Uygulaması - Modern GUI
YOLOv8 Classification modeli ile Türk Lirası banknot tespiti.
Modern ve kullanıcı dostu grafik arayüz.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageFont, ImageDraw
import cv2
import threading
from pathlib import Path
import os
import numpy as np
# YOLO lazy import - sadece gerektiğinde yüklenecek (PyTorch DLL hatası önlemek için)

# Banknot sınıfları - Model yüklendiğinde model.names'den güncellenecek
# YOLOv8 klasör isimlerini alfabetik sıraya göre indexler
# Klasörler ['10', '100', '20', '200', '5', '50'] ise alfabetik sıralama:
# 0: '10' -> '10 TL'
# 1: '100' -> '100 TL'  
# 2: '20' -> '20 TL'
# 3: '200' -> '200 TL'
# 4: '5' -> '5 TL'
# 5: '50' -> '50 TL'
CLASS_NAMES = {
    0: '10 TL',
    1: '100 TL',
    2: '20 TL',
    3: '200 TL',
    4: '5 TL',
    5: '50 TL'
} 
class BanknotDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Banknot Tespit Uygulaması - YOLOv8")
        self.root.geometry("1400x850")
        self.root.configure(bg='#0f172a')  # Modern dark blue-gray
        self.root.minsize(1000, 700)
        
        # Modern renk paleti
        self.colors = {
            'bg_primary': '#0f172a',      # Dark blue-gray
            'bg_secondary': '#1e293b',     # Lighter dark blue-gray
            'bg_card': '#1e293b',          # Card background
            'bg_hover': '#334155',         # Hover color
            'accent': '#3b82f6',           # Modern blue
            'accent_hover': '#2563eb',     # Darker blue
            'success': '#10b981',          # Modern green
            'success_hover': '#059669',
            'warning': '#f59e0b',          # Modern orange
            'warning_hover': '#d97706',
            'danger': '#ef4444',           # Modern red
            'danger_hover': '#dc2626',
            'purple': '#8b5cf6',           # Modern purple
            'purple_hover': '#7c3aed',
            'text_primary': '#f1f5f9',     # Light text
            'text_secondary': '#cbd5e1',    # Gray text
            'text_muted': '#94a3b8',       # Muted text
            'border': '#334155',            # Border color
        }
        
        # Varsayılan model yolu
        self.model_path = 'runs/classify/banknot_classifier/weights/best.pt'
        self.model = None
        self.current_image = None
        self.webcam_running = False
        self.cap = None
        
        # GUI oluştur
        self.create_widgets()
        
        # Model yükleme durumunu kontrol et
        self.check_model()
    
    def create_widgets(self):
        # Modern header with gradient effect
        header_frame = tk.Frame(self.root, bg=self.colors['bg_secondary'], height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        # Header content with padding
        header_content = tk.Frame(header_frame, bg=self.colors['bg_secondary'])
        header_content.pack(fill=tk.BOTH, expand=True, padx=30, pady=15)
        
        title_label = tk.Label(
            header_content, 
            text="💎 Banknot Tespit Sistemi", 
            font=('Segoe UI', 24, 'bold'),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_primary']
        )
        title_label.pack(side=tk.LEFT)
        
        subtitle_label = tk.Label(
            header_content,
            text="YOLOv8 AI ile Akıllı Para Tespiti",
            font=('Segoe UI', 11),
            bg=self.colors['bg_secondary'],
            fg=self.colors['text_muted']
        )
        subtitle_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # Ana içerik alanı
        main_frame = tk.Frame(self.root, bg=self.colors['bg_primary'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Sol panel - Kontroller (Modern card design)
        left_panel = tk.Frame(main_frame, bg=self.colors['bg_card'], relief=tk.FLAT)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 15))
        left_panel.config(width=340)
        left_panel.pack_propagate(False)
        
        # Sağ panel - Görüntü gösterimi (Modern card design)
        right_panel = tk.Frame(main_frame, bg=self.colors['bg_card'], relief=tk.FLAT)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Sol panel içeriği
        self.create_control_panel(left_panel)
        
        # Sağ panel içeriği
        self.create_display_panel(right_panel)
    
    def create_control_panel(self, parent):
        # Modern card style with padding
        content_frame = tk.Frame(parent, bg=self.colors['bg_card'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Model seçimi - Modern card
        model_card = tk.Frame(content_frame, bg=self.colors['bg_card'], relief=tk.FLAT)
        model_card.pack(fill=tk.X, pady=(0, 20))
        
        model_title = tk.Label(
            model_card,
            text="⚙️ Model Ayarları",
            font=('Segoe UI', 13, 'bold'),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary'],
            anchor=tk.W
        )
        model_title.pack(fill=tk.X, pady=(0, 12))
        
        self.model_label = tk.Label(
            model_card,
            text="Model: Yükleniyor...",
            bg=self.colors['bg_card'],
            font=('Segoe UI', 10),
            wraplength=280,
            justify=tk.LEFT,
            fg=self.colors['text_secondary']
        )
        self.model_label.pack(anchor=tk.W, pady=(0, 10))
        
        model_btn = tk.Button(
            model_card,
            text="📁 Model Seç",
            command=self.select_model,
            bg=self.colors['accent'],
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor='hand2',
            activebackground=self.colors['accent_hover'],
            activeforeground='white',
            bd=0
        )
        model_btn.pack(fill=tk.X, pady=(0, 0))
        
        # Güven eşiği - Modern card
        conf_card = tk.Frame(content_frame, bg=self.colors['bg_card'], relief=tk.FLAT)
        conf_card.pack(fill=tk.X, pady=(0, 20))
        
        conf_title = tk.Label(
            conf_card,
            text="🎯 Güven Eşiği",
            font=('Segoe UI', 13, 'bold'),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary'],
            anchor=tk.W
        )
        conf_title.pack(fill=tk.X, pady=(0, 12))
        
        self.conf_var = tk.DoubleVar(value=0.25)
        conf_scale = tk.Scale(
            conf_card,
            from_=0.1,
            to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.conf_var,
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary'],
            highlightthickness=0,
            troughcolor=self.colors['bg_hover'],
            activebackground=self.colors['accent'],
            font=('Segoe UI', 9),
            length=280
        )
        conf_scale.pack(fill=tk.X, pady=(0, 8))
        
        self.conf_label = tk.Label(
            conf_card,
            text="Değer: 0.25",
            bg=self.colors['bg_card'],
            font=('Segoe UI', 11, 'bold'),
            fg=self.colors['accent']
        )
        self.conf_label.pack()
        
        conf_scale.config(command=self.update_conf_label)
        
        # İşlem seçenekleri - Modern card
        action_card = tk.Frame(content_frame, bg=self.colors['bg_card'], relief=tk.FLAT)
        action_card.pack(fill=tk.X, pady=(0, 20))
        
        action_title = tk.Label(
            action_card,
            text="🚀 İşlem Seçenekleri",
            font=('Segoe UI', 13, 'bold'),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary'],
            anchor=tk.W
        )
        action_title.pack(fill=tk.X, pady=(0, 12))
        
        # Resim seç - Modern button
        img_btn = tk.Button(
            action_card,
            text="📷 Resim Seç ve Tespit Et",
            command=self.select_and_detect_image,
            bg=self.colors['success'],
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief=tk.FLAT,
            padx=20,
            pady=14,
            cursor='hand2',
            width=32,
            activebackground=self.colors['success_hover'],
            activeforeground='white',
            bd=0
        )
        img_btn.pack(fill=tk.X, pady=(0, 10))
        
        # Video seç - Modern button
        video_btn = tk.Button(
            action_card,
            text="🎥 Video Seç ve Tespit Et",
            command=self.select_and_detect_video,
            bg=self.colors['warning'],
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief=tk.FLAT,
            padx=20,
            pady=14,
            cursor='hand2',
            width=32,
            activebackground=self.colors['warning_hover'],
            activeforeground='white',
            bd=0
        )
        video_btn.pack(fill=tk.X, pady=(0, 10))
        
        # Webcam - Modern button
        self.webcam_btn = tk.Button(
            action_card,
            text="📹 Webcam'i Başlat",
            command=self.toggle_webcam,
            bg=self.colors['purple'],
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            relief=tk.FLAT,
            padx=20,
            pady=14,
            cursor='hand2',
            width=32,
            activebackground=self.colors['purple_hover'],
            activeforeground='white',
            bd=0
        )
        self.webcam_btn.pack(fill=tk.X, pady=(0, 0))
        
        # Sonuçlar - Modern card
        result_card = tk.Frame(content_frame, bg=self.colors['bg_card'], relief=tk.FLAT)
        result_card.pack(fill=tk.BOTH, expand=True, pady=(0, 0))
        
        result_title = tk.Label(
            result_card,
            text="📊 Tespit Sonuçları",
            font=('Segoe UI', 13, 'bold'),
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary'],
            anchor=tk.W
        )
        result_title.pack(fill=tk.X, pady=(0, 12))
        
        # Sonuç metni için scrollable text - Modern style
        text_container = tk.Frame(result_card, bg=self.colors['bg_hover'], relief=tk.FLAT)
        text_container.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(text_container, bg=self.colors['bg_hover'], 
                                troughcolor=self.colors['bg_hover'],
                                activebackground=self.colors['accent'])
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.result_text = tk.Text(
            text_container,
            wrap=tk.WORD,
            font=('Consolas', 10),
            bg=self.colors['bg_hover'],
            fg=self.colors['text_primary'],
            yscrollcommand=scrollbar.set,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
            insertbackground=self.colors['accent'],
            selectbackground=self.colors['accent'],
            selectforeground='white'
        )
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        scrollbar.config(command=self.result_text.yview)
    
    def create_display_panel(self, parent):
        # Modern content frame
        content_frame = tk.Frame(parent, bg=self.colors['bg_card'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Görüntü gösterim alanı - Modern card
        display_card = tk.Frame(content_frame, bg=self.colors['bg_hover'], relief=tk.FLAT)
        display_card.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        self.image_label = tk.Label(
            display_card,
            text="🖼️ Görüntü burada görüntülenecek\n\nResim seçin veya webcam'i başlatın",
            bg=self.colors['bg_hover'],
            fg=self.colors['text_muted'],
            font=('Segoe UI', 14),
            width=60,
            height=25,
            justify=tk.CENTER
        )
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Durum çubuğu - Modern style
        status_frame = tk.Frame(content_frame, bg=self.colors['bg_secondary'], height=45, relief=tk.FLAT)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)
        
        status_content = tk.Frame(status_frame, bg=self.colors['bg_secondary'])
        status_content.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.status_label = tk.Label(
            status_content,
            text="✓ Hazır",
            bg=self.colors['bg_secondary'],
            fg=self.colors['success'],
            font=('Segoe UI', 11),
            anchor=tk.W
        )
        self.status_label.pack(side=tk.LEFT)
    
    def update_class_names_from_model(self):
        """Model'den class isimlerini al ve CLASS_NAMES'i güncelle"""
        global CLASS_NAMES
        if self.model and hasattr(self.model, 'names'):
            # Model'in class isimlerini al (örn: {0: '10', 1: '100', 2: '20', 3: '200', 4: '5', 5: '50'})
            model_names = self.model.names
            # Class isimlerine ' TL' ekle
            CLASS_NAMES = {idx: f"{name} TL" for idx, name in model_names.items()}
    
    def check_model(self):
        """Model dosyasının varlığını kontrol et"""
        if os.path.exists(self.model_path):
            try:
                # Lazy import - PyTorch DLL hatası önlemek için
                try:
                    from ultralytics import YOLO
                except OSError as e:
                    if "DLL" in str(e) or "WinError" in str(e):
                        self.model_label.config(
                            text="Model: ✗ PyTorch Hatası\nDLL yüklenemedi",
                            fg='red'
                        )
                        self.update_status(
                            "PyTorch yüklenemedi. Lütfen:\n"
                            "1. Visual C++ Redistributables yükleyin\n"
                            "2. pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
                        )
                        messagebox.showerror(
                            "PyTorch Hatası",
                            "PyTorch kütüphanesi yüklenemedi.\n\n"
                            "Çözüm:\n"
                            "1. Visual C++ Redistributables'ı yükleyin\n"
                            "2. PyTorch'u yeniden kurun:\n"
                            "   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n"
                            "3. Veya CPU-only versiyonu kullanın"
                        )
                        return
                    raise
                
                self.model = YOLO(self.model_path)
                # Model'den class isimlerini al ve güncelle
                self.update_class_names_from_model()
                self.model_label.config(
                    text=f"Model: ✓ Yüklendi\n{Path(self.model_path).name}",
                    fg=self.colors['success']
                )
                self.update_status("Model başarıyla yüklendi")
            except Exception as e:
                self.model_label.config(
                    text=f"Model: ✗ Hata\n{str(e)[:30]}...",
                    fg=self.colors['danger']
                )
                self.update_status(f"Model yükleme hatası: {str(e)}")
        else:
            self.model_label.config(
                text=f"Model: ✗ Bulunamadı\n{Path(self.model_path).name if len(self.model_path) < 40 else 'Model yolu çok uzun'}",
                fg=self.colors['danger']
            )
            self.update_status("Model dosyası bulunamadı. Lütfen model seçin veya eğitin.")
    
    def select_model(self):
        """Model dosyası seç"""
        file_path = filedialog.askopenfilename(
            title="Model Dosyası Seç",
            filetypes=[("PyTorch Model", "*.pt"), ("Tüm Dosyalar", "*.*")]
        )
        if file_path:
            self.model_path = file_path
            self.model = None  # Eski modeli temizle
            self.check_model()
    
    def update_conf_label(self, value):
        """Güven eşiği etiketi güncelle"""
        conf_value = float(value)
        self.conf_label.config(text=f"Değer: {conf_value:.2f}")
        # Renk değiştir - Modern colors
        if conf_value < 0.5:
            self.conf_label.config(fg=self.colors['danger'])
        elif conf_value < 0.7:
            self.conf_label.config(fg=self.colors['warning'])
        else:
            self.conf_label.config(fg=self.colors['success'])
    
    def update_status(self, message):
        """Durum çubuğunu güncelle"""
        # Mesaja göre renk ve ikon ekle - Modern colors
        if "✓" in message or "Hazır" in message or "tamamlandı" in message.lower():
            color = self.colors['success']
            prefix = "✓ "
        elif "Hata" in message or "hata" in message.lower() or "✗" in message:
            color = self.colors['danger']
            prefix = "✗ "
        elif "yükleniyor" in message.lower() or "işleniyor" in message.lower():
            color = self.colors['accent']
            prefix = "⏳ "
        else:
            color = self.colors['text_secondary']
            prefix = "ℹ "
        
        self.status_label.config(text=f"{prefix}{message}", fg=color)
        self.root.update_idletasks()
    
    def update_result_text(self, text):
        """Sonuç metnini güncelle"""
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.see(tk.END)
    
    def display_image(self, image_path=None, image_array=None):
        """Görüntüyü göster"""
        try:
            if image_array is not None:
                # OpenCV BGR -> RGB
                if len(image_array.shape) == 3:
                    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image_array
                pil_image = Image.fromarray(image_rgb)
            elif image_path:
                pil_image = Image.open(image_path)
            else:
                return
            
            # Görüntüyü label boyutuna göre ölçekle
            label_width = self.image_label.winfo_width()
            label_height = self.image_label.winfo_height()
            
            # Eğer label henüz render edilmediyse varsayılan değerler kullan
            if label_width <= 1:
                label_width = 600
            if label_height <= 1:
                label_height = 400
            
            # Orijinal boyut oranını koru
            img_width, img_height = pil_image.size
            ratio = min(label_width / img_width, label_height / img_height, 1.0)  # Büyütme yok, sadece küçültme
            
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            # Yüksek kaliteli resize
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Tkinter için dönüştür
            photo = ImageTk.PhotoImage(pil_image)
            self.image_label.config(image=photo, text="", bg=self.colors['bg_hover'])
            self.image_label.image = photo  # Referansı sakla (garbage collection önleme)
        except Exception as e:
            messagebox.showerror("Hata", f"Görüntü gösterilirken hata: {str(e)}")
    
    def select_and_detect_image(self):
        """Resim seç ve tespit et"""
        if not self.model:
            messagebox.showerror("Hata", "Lütfen önce bir model yükleyin!")
            return
        
        file_path = filedialog.askopenfilename(
            title="Resim Seç",
            filetypes=[
                ("Resim Dosyaları", "*.jpg *.jpeg *.png *.bmp"),
                ("Tüm Dosyalar", "*.*")
            ]
        )
        
        if file_path:
            self.update_status("Resim işleniyor...")
            # Thread'de çalıştır
            thread = threading.Thread(target=self.detect_image_thread, args=(file_path,))
            thread.daemon = True
            thread.start()
    
    def detect_image_thread(self, image_path):
        """Resim tespiti (thread'de çalışır) - Basit ve verimli yaklaşım"""
        try:
            if not self.model:
                # Model yoksa lazy load dene
                try:
                    from ultralytics import YOLO
                    self.model = YOLO(self.model_path)
                    # Model'den class isimlerini güncelle
                    self.update_class_names_from_model()
                except Exception as e:
                    error_msg = f"Model yüklenemedi: {str(e)}"
                    self.root.after(0, lambda: messagebox.showerror("Hata", error_msg))
                    self.root.after(0, lambda: self.update_status("Model yükleme hatası"))
                    return
            
            # Görüntüyü yükle
            img = cv2.imread(image_path)
            if img is None:
                error_msg = "Görüntü yüklenemedi!"
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Hata", msg))
                return
            
            # Tespit yap - Modelin kendi yeteneklerini kullan
            results = self.model(image_path, conf=self.conf_var.get(), verbose=False)
            
            # Sonuç metni başlangıcı
            result_text = f"📷 Resim: {Path(image_path).name}\n"
            result_text += "=" * 50 + "\n\n"
            
            detected_results = []
            annotated_img = img.copy()
            
            for result in results:
                # Detection modeli kontrolü
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    num_boxes = len(boxes) if boxes is not None else 0
                    
                    if num_boxes == 0:
                        result_text += "⚠ Hiç banknot tespit edilmedi.\n"
                    else:
                        result_text += f"✅ {num_boxes} adet banknot tespit edildi:\n\n"
                        for i, box in enumerate(boxes):
                            cls_id = int(box.cls[0]) if box.cls is not None else -1
                            conf = float(box.conf[0]) if box.conf is not None else 0.0
                            name = CLASS_NAMES.get(cls_id, f'Class {cls_id}')
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            detected_results.append({
                                'name': name,
                                'conf': conf,
                                'bbox': (x1, y1, x2, y2)
                            })
                            
                            result_text += (
                                f"{i+1}. {name} | Güven: {conf:.2%} | "
                                f"Kutu: ({x1}, {y1}) - ({x2}, {y2})\n"
                            )
                    
                    # YOLO'nun çizdiği kutulu görüntüyü al
                    annotated_img = result.plot()
                    self.current_image = annotated_img
                    
                # Classification modeli kontrolü
                elif hasattr(result, 'probs') and result.probs is not None:
                    probs = result.probs
                    top1_idx = probs.top1
                    top1_conf = probs.top1conf.item()
                    name = CLASS_NAMES.get(top1_idx, f'Class {top1_idx}')
                    
                    if top1_conf >= self.conf_var.get():
                        result_text += f"✅ Tespit Edilen: {name}\n"
                        result_text += f"Güven Skoru: {top1_conf:.2%}\n\n"
                        
                        # Tüm görüntüyü kutu içine al
                        h, w = img.shape[:2]
                        color = (0, 255, 0)
                        cv2.rectangle(annotated_img, (10, 10), (w-10, h-10), color, 3)
                        
                        label = f"{name}: {top1_conf:.1%}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                        cv2.rectangle(annotated_img, (10, 10), (10 + label_size[0] + 20, 10 + label_size[1] + 20), color, -1)
                        cv2.putText(annotated_img, label, (20, 10 + label_size[1] + 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
                        
                        detected_results.append({
                            'name': name,
                            'conf': top1_conf,
                            'bbox': (0, 0, w, h)
                        })
                    else:
                        result_text += "⚠ Hiç banknot tespit edilmedi.\n"
            
            self.current_image = annotated_img
            
            # UI'ı güncelle
            self.root.after(0, lambda img=annotated_img: self.display_image(image_array=img))
            self.root.after(0, lambda txt=result_text: self.update_result_text(txt))
            status_msg = (
                f"Tespit tamamlandı! - {len(detected_results)} banknot"
                if len(detected_results) > 0 else "Tespit tamamlandı fakat banknot bulunamadı"
            )
            self.root.after(0, lambda msg=status_msg: self.update_status(msg))
        except Exception as e:
            error_msg = f"Tespit hatası: {str(e)}"
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("Hata", msg))
            self.root.after(0, lambda: self.update_status("Hata oluştu"))
    
    def select_and_detect_video(self):
        """Video seç ve tespit et"""
        if not self.model:
            messagebox.showerror("Hata", "Lütfen önce bir model yükleyin!")
            return
        
        file_path = filedialog.askopenfilename(
            title="Video Seç",
            filetypes=[
                ("Video Dosyaları", "*.mp4 *.avi *.mov *.mkv"),
                ("Tüm Dosyalar", "*.*")
            ]
        )
        
        if file_path:
            self.update_status("Video işleniyor...")
            # Thread'de çalıştır
            thread = threading.Thread(target=self.detect_video_thread, args=(file_path,))
            thread.daemon = True
            thread.start()
    
    @staticmethod
    def draw_highlight_box_bgr(image_bgr, color=(0, 255, 0), thickness=4, margin=10):
        """
        Tespit edilen parayı vurgulamak için görüntüyü çevreleyen kutu çiz.
        Not: Bu uygulamada kullanılan model sınıflandırma (classification)
        modeli olduğu için tam konum (bounding box) bilgisi üretmez; bu yüzden
        burada tüm görüntüyü saran bir vurgu kutusu kullanıyoruz.
        """
        if image_bgr is None:
            return image_bgr
        # Ultralytics'in ürettiği bazı NumPy array'ler read-only olduğu için
        # önce writeable bir kopya oluşturuyoruz.
        if hasattr(image_bgr, "flags") and not image_bgr.flags.writeable:
            image_bgr = image_bgr.copy()
        h, w = image_bgr.shape[:2]
        x1 = margin
        y1 = margin
        x2 = max(margin, w - margin)
        y2 = max(margin, h - margin)
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, thickness)
        return image_bgr
    
    def detect_video_thread(self, video_path):
        """Video tespiti (thread'de çalışır)"""
        try:
            if not self.model:
                try:
                    from ultralytics import YOLO
                    self.model = YOLO(self.model_path)
                    # Model'den class isimlerini güncelle
                    self.update_class_names_from_model()
                except Exception as e:
                    error_msg = f"Model yüklenemedi: {str(e)}"
                    self.root.after(0, lambda: messagebox.showerror("Hata", error_msg))
                    self.root.after(0, lambda: self.update_status("Model yükleme hatası"))
                    return
            
            # Video bilgilerini al
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Video açılamadı!")
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            result_text = f"🎥 Video: {Path(video_path).name}\n"
            result_text += "=" * 50 + "\n\n"
            result_text += f"📊 FPS: {fps}\n"
            result_text += f"🎬 Toplam Frame: {total_frames:,}\n"
            result_text += f"⏱️  Süre: {duration:.1f} saniye\n\n"
            result_text += "⏳ Video işleniyor... (Bu işlem biraz zaman alabilir)\n"
            result_text += "Lütfen bekleyin...\n"
            
            self.root.after(0, lambda: self.update_result_text(result_text))
            
            # Video işleme
            results = self.model(video_path, conf=self.conf_var.get(), save=True, verbose=False)
            
            result_text += "\n" + "=" * 50 + "\n"
            result_text += "✅ Video işleme tamamlandı!\n\n"
            result_text += "💾 Sonuçlar şu klasöre kaydedildi:\n"
            result_text += "   runs/classify/predict/\n"
            
            self.root.after(0, lambda: self.update_result_text(result_text))
            self.root.after(0, lambda: self.update_status("Video işleme tamamlandı!"))
            self.root.after(0, lambda: messagebox.showinfo(
                "Başarılı", 
                f"Video işleme tamamlandı!\n\n"
                f"Sonuçlar 'runs/classify/predict' klasörüne kaydedildi."
            ))
            
            cap.release()
        except Exception as e:
            error_msg = f"Video işleme hatası: {str(e)}"
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("Hata", msg))
            self.root.after(0, lambda: self.update_status("Hata oluştu"))
    
    def toggle_webcam(self):
        """Webcam'i başlat/durdur"""
        if not self.model:
            messagebox.showerror("Hata", "Lütfen önce bir model yükleyin!")
            return
        
        if not self.webcam_running:
            self.start_webcam()
        else:
            self.stop_webcam()
    
    def start_webcam(self):
        """Webcam'i başlat"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Hata", "Webcam açılamadı!")
            return
        
        self.webcam_running = True
        self.webcam_btn.config(text="🛑 Webcam'i Durdur", bg=self.colors['danger'], activebackground=self.colors['danger_hover'])
        self.update_status("Webcam aktif - Çıkmak için 'Durdur' butonuna basın")
        
        # Webcam thread'i başlat
        thread = threading.Thread(target=self.webcam_loop)
        thread.daemon = True
        thread.start()
    
    def stop_webcam(self):
        """Webcam'i durdur"""
        self.webcam_running = False
        if self.cap:
            self.cap.release()
        self.webcam_btn.config(text="📹 Webcam'i Başlat", bg=self.colors['purple'])
        self.update_status("Webcam durduruldu")
        self.image_label.config(image="", text="🖼️ Görüntü burada görüntülenecek\n\nResim seçin veya webcam'i başlatın", 
                               bg=self.colors['bg_hover'], fg=self.colors['text_muted'])
    
    def webcam_loop(self):
        """Webcam döngüsü - Her frame'de tespit"""
        import time
        frame_count = 0
        start_time = time.time()
        
        while self.webcam_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if not self.model:
                try:
                    from ultralytics import YOLO
                    self.model = YOLO(self.model_path)
                    # Model'den class isimlerini güncelle
                    self.update_class_names_from_model()
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("Hata", f"Model yüklenemedi: {str(e)}"))
                    break
            
            # Her frame'de tespit yap - Performans için imgsz parametresi kullan
            # Görüntü boyutunu küçült (640x480 gibi) ama orijinal frame'i göster
            results = self.model(frame, conf=self.conf_var.get(), verbose=False, imgsz=640)
            
            # YOLO'nun kutulu görüntüsünü al
            annotated_frame = None
            detected_results = []
            
            for result in results:
                # Detection modeli kontrolü
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    # Orijinal frame'i kullan, YOLO'nun plot() yerine manuel çizim yapacağız
                    annotated_frame = frame.copy()
                    
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            cls_id = int(box.cls[0]) if box.cls is not None else -1
                            conf = float(box.conf[0]) if box.conf is not None else 0.0
                            
                            # Güven eşiği kontrolü
                            if conf >= self.conf_var.get():
                                name = CLASS_NAMES.get(cls_id, f'Class {cls_id}')
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                detected_results.append({
                                    'name': name,
                                    'conf': conf,
                                    'bbox': (x1, y1, x2, y2)
                                })
                                
                                # Manuel olarak kutu ve etiket çiz
                                color = (0, 255, 0)
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                                label = f"{name}: {conf:.1%}"
                                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                label_y = max(y1 - 5, label_size[1] + 5)
                                cv2.rectangle(annotated_frame, (x1, label_y - label_size[1] - 3), 
                                             (x1 + label_size[0] + 6, label_y + 3), color, -1)
                                cv2.putText(annotated_frame, label, (x1 + 3, label_y),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                
                # Classification modeli kontrolü
                elif hasattr(result, 'probs') and result.probs is not None:
                    probs = result.probs
                    top1_idx = probs.top1
                    top1_conf = probs.top1conf.item()
                    name = CLASS_NAMES.get(top1_idx, f'Class {top1_idx}')
                    
                    # Orijinal frame'i kullan, YOLO'nun plot() yerine manuel çizim yapacağız
                    annotated_frame = frame.copy()
                    
                    # Güven eşiği kontrolü - sadece eşiğin üzerindekileri göster
                    if top1_conf >= self.conf_var.get():
                        detected_results.append({
                            'name': name,
                            'conf': top1_conf,
                            'bbox': (0, 0, frame.shape[1], frame.shape[0])
                        })
                        
                        # Manuel olarak kutu ve etiket çiz
                        h, w = frame.shape[:2]
                        color = (0, 255, 0)
                        cv2.rectangle(annotated_frame, (10, 10), (w-10, h-10), color, 3)
                        label = f"{name}: {top1_conf:.1%}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                        cv2.rectangle(annotated_frame, (10, 10), (10 + label_size[0] + 20, 10 + label_size[1] + 20), color, -1)
                        cv2.putText(annotated_frame, label, (20, 10 + label_size[1] + 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
            
            if annotated_frame is None:
                annotated_frame = frame.copy()
            
            # FPS hesapla
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps = frame_count / elapsed
                fps_text = f"FPS: {fps:.1f}"
            else:
                fps_text = "FPS: --"
            
            # Sonuç metni
            result_text = f"📹 Webcam - Canlı Tespit\n"
            result_text += "=" * 50 + "\n\n"
            if len(detected_results) > 0:
                result_text += f"⚡ {fps_text}\n\n"
                for i, det in enumerate(detected_results):
                    x1, y1, x2, y2 = det['bbox']
                    result_text += (
                        f"{i+1}. {det['name']} | Güven: {det['conf']:.2%} | "
                        f"Kutu: ({x1}, {y1}) - ({x2}, {y2})\n"
                    )
            else:
                result_text += f"⚡ {fps_text}\n"
            
            # FPS'i sağ alta yaz
            h, w = annotated_frame.shape[:2]
            fps_label_size, _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            fps_x = w - fps_label_size[0] - 10
            fps_y = h - 10
            
            # Arka plan için kutu çiz
            cv2.rectangle(annotated_frame, 
                         (fps_x - 5, fps_y - fps_label_size[1] - 5),
                         (fps_x + fps_label_size[0] + 5, fps_y + 5),
                         (0, 0, 0), -1)
            
            cv2.putText(
                annotated_frame,
                fps_text,
                (fps_x, fps_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            
            # UI'ı güncelle
            self.root.after(0, lambda img=annotated_frame: self.display_image(image_array=img))
            self.root.after(0, lambda txt=result_text: self.update_result_text(txt))
        
        if self.cap:
            self.cap.release()

def main():
    root = tk.Tk()
    app = BanknotDetectionGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()

