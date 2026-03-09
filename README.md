# Enterprise Vision AI

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-yellow)](https://huggingface.co/bas-industrial)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)](https://streamlit.io/)
[![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-orange)](https://docs.ultralytics.com/)

[![CI/CD](https://github.com/bas-industrial/enterprise-vision-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/bas-industrial/enterprise-vision-ai/actions)
[![Coverage](https://codecov.io/gh/bas-industrial/enterprise-vision-ai/branch/main/graph/badge.svg)](https://codecov.io/gh/bas-industrial/enterprise-vision-ai)
[![Last Commit](https://img.shields.io/github/last-commit/bas-industrial/enterprise-vision-ai)](https://github.com/bas-industrial/enterprise-vision-ai)

> **Türkçe:** Endüstriyel kalite kontrol için açık kaynaklı bilgisayarlı görü platformu  
> **English:** Open-source computer vision platform for industrial quality control

## 📋 Proje Açıklaması | Project Description

Enterprise Vision AI, üretim ve madencilik ortamlarında gerçek zamanlı kusur tespiti ve cevher sınıflandırması için tasarlanmış, kurumsal kalitede açık kaynaklı bir bilgisayarlı görü platformudur.

Enterprise Vision AI is an enterprise-grade open-source computer vision platform designed for real-time defect detection and ore classification in manufacturing and mining environments.

### Hedef Kitle | Target Audience

- **Üretim Mühendisleri** | Manufacturing Engineers
- **Maden Operatörleri** | Mining Operators
- **Kalite Kontrol Uzmanları** | Quality Control Specialists
- **Makine Öğrenimi Mühendisleri** | ML Engineers
- **Araştırmacılar** | Researchers

---

## ✨ Özellikler | Features

### 🖼️ Görüntü İşleme | Image Processing

- [x] Gerçek zamanlı görüntü analizi | Real-time image analysis
- [x] YOLO11 tabanlı nesne tespiti | YOLO11-based object detection
- [x] OpenCV entegrasyonu | OpenCV integration
- [x] Görüntü ön işleme pipeline'ı | Image preprocessing pipeline

### 🔍 Defekt Tespiti | Defect Detection

- [x] Yüzey kusuru tespiti | Surface defect detection
- [x] Boyut analizi | Dimension analysis
- [x] Kusur sınıflandırma | Defect classification
- [x] Görselleştirme araçları | Visualization tools

### �矿物 Tanıma | Ore Recognition

- [x] Cevher türü sınıflandırması | Ore type classification
- [x] Kalite değerlendirmesi | Quality assessment
- [x] Ön seçim optimizasyonu | Pre-selection optimization
- [x] Histogram analizi | Histogram analysis

### 🌐 Web Arayüzü | Web Interface

- [x] Streamlit tabanlı UI | Streamlit-based UI
- [x] WebRTC canlı akış | WebRTC live streaming
- [x] Plotly grafikleri | Plotly charts
- [x] Mobil uyumlu tasarım | Mobile-responsive design

### 🏗️ Kurumsal Özellikler | Enterprise Features

- [ ] MLflow model versiyonlama | MLflow model versioning
- [ ] REST API endpoint'leri | REST API endpoints
- [ ] PostgreSQL veritabanı desteği | PostgreSQL database support
- [ ] Redis önbellek entegrasyonu | Redis cache integration
- [ ] Docker/Kubernetes dağıtımı | Docker/Kubernetes deployment

---

## 🚀 Hızlı Başlangıç | Quick Start

### Kurulum | Installation

```bash
# Depoyu klonlayın | Clone the repository
git clone https://github.com/bas-industrial/enterprise-vision-ai.git
cd enterprise-vision-ai

# Sanal ortam oluşturun | Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya | or
venv\Scripts\activate     # Windows

# Bağımlılıkları yükleyin | Install dependencies
pip install -r requirements.txt

# Uygulamayı çalıştırın | Run the application
streamlit run app.py
```

### Docker ile Çalıştırma | Run with Docker

```bash
# Docker Compose ile başlatın | Start with Docker Compose
docker-compose up -d

# Veya manuel olarak | Or manually
docker build -t bas-industrial-ai .
docker run -p 8501:8501 bas-industrial-ai
```

### Model İndirme | Download Models

```bash
# Modelleri indirin | Download models
python models/download_models.py
```

Modeller varsayılan olarak `models/` dizinine indirilecektir.

Models will be downloaded to the `models/` directory by default.

### Kullanım | Usage

1. Tarayıcınızda `http://localhost:8501` adresine gidin
2. Sol menüden modülü seçin (Defekt Tespiti / Cevher Ön Seçimi)
3. Görüntü yükleyin veya canlı kamera akışını kullanın
4. Sonuçları analiz edin

---

## 📖 Demo

| Modül | URL | Açıklama |
|-------|-----|----------|
| Defekt Tespiti | `/defect` | Yüzey kusuru tespiti |
| Cevher Ön Seçimi | `/ore` | Cevher sınıflandırma |

---

## 📂 Proje Yapısı | Project Structure

```
enterprise-vision-ai/
├── app.py                 # Ana Streamlit uygulaması
├── services/             # Servis katmanı
│   └── utils.py         # Yardımcı fonksiyonlar (formerly utils.py)
├── pages/                # Sayfa modülleri
│   ├── 01_Defekt_Tespiti.py  # Defekt tespiti sayfası
│   ├── defect.py        # Defekt tespiti modülü
│   ├── 02_Cevher_On_Secimi.py # Cevher sınıflandırma sayfası
│   └── ore.py           # Cevher sınıflandırma modülü
├── api/                 # FastAPI gateway
├── clients/              # İstemci kütüphaneleri
├── models/               # ML modelleri
├── datasets/             # Veri setleri
├── notebooks/           # Jupyter notebooks
├── services/             # Servis katmanı
├── src/                  # Kaynak kod (enterprise_vision_ai paketi)
├── tests/                # Birim testleri
├── huggingface_space/   # HuggingFace Space uygulaması
├── docker-compose.yml   # Docker Compose konfigürasyonu
├── Dockerfile           # Docker image tanımı
├── pyproject.toml       # Python proje konfigürasyonu
├── requirements.txt     # Python bağımlılıkları
└── Makefile            # Geliştirme komutları
```

---

## 🤝 Katkıda Bulunma | Contributing

Katkılarınızı bekliyoruz! Lütfen [CONTRIBUTING.md](CONTRIBUTING.md) dosyasını okuyun.

We welcome contributions! Please read [CONTRIBUTING.md](CONTRIBUTING.md).

### Katkı Adımları | Contribution Steps

1. Fork edin | Fork the repository
2. Feature branch oluşturun | Create feature branch (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin | Commit your changes (`git commit -m 'Add amazing feature'`)
4. Branch'i push edin | Push to branch (`git push origin feature/amazing-feature`)
5. Pull Request açın | Open a Pull Request

---

## 📜 Lisans | License

Bu proje Apache License 2.0 altında lisanslanmıştır - detaylar için [LICENSE](LICENSE) dosyasına bakın.

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE) for details.

---

## 📞 İletişim | Contact

- **GitHub Issues**: [Issues](https://github.com/bas-industrial/enterprise-vision-ai/issues)
- **Tartışmalar**: [Discussions](https://github.com/bas-industrial/enterprise-vision-ai/discussions)
- **E-posta**: contact@bas-industrial.ai

---

## 🙏 Teşekkürler | Acknowledgments

- [Ultralytics](https://ultralytics.com) - YOLO modelleri
- [Streamlit](https://streamlit.io) - Web framework
- [OpenCV](https://opencv.org) - Görüntü işleme
- [Plotly](https://plotly.com) - Veri görselleştirme

---

<div align="center">

**Made with ❤️ by Enterprise Vision AI Team**

</div>
