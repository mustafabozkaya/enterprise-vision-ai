---
title: Enterprise Vision AI Demo
emoji: 🏭
colorFrom: blue
colorTo: blue
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
pinned: false
license: apache-2.0
---

# 🏭 Enterprise Vision AI - HuggingFace Space

Endüstriyel yapay zeka çözümleri için interaktif demo uygulaması.

## 📋 Proje Hakkında

Enterprise Vision AI, endüstriyel üretim süreçlerinde yapay zeka kullanarak:

- **Defekt Tespiti**: Ürün yüzeylerindeki kusurların otomatik tespiti
- **Cevher Ön Seçimi**: Madencilik süreçlerinde cevher sınıflandırma ve ayrıştırma

## 🚀 Kullanım

### Defekt Tespiti

1. **Görüntü Yükle**: Analiz etmek istediğiniz ürün görüntüsünü yükleyin
2. **Güven Eşiği Ayarla**: Algılama hassasiyetini ayarlayın (0.1-1.0)
3. **Analiz Et**: Sonuçları görüntüleyin

### Cevher Sınıflandırma

1. **Görüntü Yükle**: Cevher görüntüsünü yükleyin
2. **Güven Eşiği Ayarla**: Sınıflandırma hassasiyetini ayarlayın
3. **Sınıflandır**: Cevher türlerini ve oranlarını görüntüleyin

## 🛠️ Teknoloji

- **YOLO (You Only Look Once)**: Nesne tespit algoritması
- **Ultralytics**: YOLO model framework'ü
- **PyTorch**: Derin öğrenme backend
- **Gradio**: Web arayüzü

## 🤖 Model

Modeller HuggingFace Hub'dan yüklenmektedir:

- [Defect Detection Model](https://huggingface.co/bas-industriel/yolo-defect-detection)
- [Ore Classification Model](https://huggingface.co/bas-industriel/yolo-ore-classification)

## 💻 Yerel Çalıştırma

```bash
# Gerekli paketleri yükleyin
pip install -r requirements.txt

# Uygulamayı başlatın
python app.py
```

## 📝 Lisans

Apache License 2.0 - Tüm hakları saklıdır.

## 👥 Katkıda Bulunanlar

Enterprise Vision AI Geliştirme Ekibi

---

*Bu bir demo uygulamasıdır. Gerçek model ağırlıkları HuggingFace Hub'a yüklendikten sonra otomatik olarak kullanılacaktır.*
