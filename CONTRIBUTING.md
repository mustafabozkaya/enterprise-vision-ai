# Katkıda Bulunma Rehberi | Contributing Guide

Bu belge, Enterprise Vision AI projesine nasıl katkıda bulunacağınızı açıklar.

This guide explains how to contribute to the Enterprise Vision AI project.

---

## 📋 İçerik | Table of Contents

1. [Davranış Kuralları | Code of Conduct](#davranış-kuralları--code-of-conduct)
2. [Nasıl Katkıda Bulunabilirsiniz | How to Contribute](#nasıl-katkıda-bulunabilirsiniz--how-to-contribute)
3. [Geliştirme Ortamı Kurulumu | Development Environment Setup](#geliştirme-ortamı-kurulumu--development-environment-setup)
4. [Pull Request Süreci | Pull Request Process](#pull-request-süreci--pull-request-process)
5. [Kod Stili | Code Style](#kod-stili--code-style)
6. [Commit Mesajları | Commit Messages](#commit-mesajları--commit-messages)
7. [Test Yazımı | Writing Tests](#test-yazımı--writing-tests)
8. [Dokümantasyon | Documentation](#dokümantasyon--documentation)

---

## 📜 Davranış Kuralları | Code of Conduct

Bu proje ve topluluk için davranış kurallarımız [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) dosyasında belirtilmiştir. Katkıda bulunarak, bu kurallara uymanızı bekliyoruz.

Our code of conduct for this project and community is outlined in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). By contributing, you are expected to uphold this code.

---

## 🤝 Nasıl Katkıda Bulunabilirsiniz | How to Contribute

### Katkı Türleri | Types of Contributions

Katkıda bulunmanın birçok yolu vardır:

There are many ways to contribute:

- **🐛 Hata Raporları | Bug Reports**: Bir hata bulursanız, GitHub Issues'da raporlayın
- **💡 Özellik Talepleri | Feature Requests**: Yeni özellikler önerin
- **📝 Dokümantasyon**: Dokümantasyonu iyileştirin veya çevirin
- **💻 Kod Katkısı**: Kod yazın, düzeltin veya refactor edin
- **🧪 Test**: Birim testleri veya entegrasyon testleri yazın
- **🎨 Tasarım**: UI/UX iyileştirmeleri önerin
- **🌍 Çeviri**: Projeyi farklı dillere çevirin

### Başlamadan Önce | Before You Start

1. Projenin [README.md](README.md) dosyasını okuyun
2. [Açık Issues'ları](https://github.com/bas-industrial/enterprise-vision-ai/issues) kontrol edin
3. Büyük değişiklikler için önceden tartışmak için Issue açın

---

## 🛠️ Geliştirme Ortamı Kurulumu | Development Environment Setup

### Gereksinimler | Requirements

- Python 3.10+
- Git
- Docker (isteğe bağlı)

### Adım Adım Kurulum | Step-by-Step Setup

```bash
# 1. Depoyu fork edin ve klonlayın
git clone https://github.com/YOUR_USERNAME/enterprise-vision-ai.git
cd enterprise-vision-ai

# 2. Sanal ortam oluşturun
python -m venv venv

# 3. Sanal ortamı etkinleştirin
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 4. Geliştirme bağımlılıklarını yükleyin
pip install -e ".[dev]"

# 5. Pre-commit hook'larını kurun
pre-commit install
```

### Docker ile Geliştirme | Development with Docker

```bash
# Docker Compose ile geliştirme ortamını başlatın
docker-compose up -d

# Geliştirme modunda yeniden inşa edin
docker-compose up -d --build
```

---

## 🔄 Pull Request Süreci | Pull Request Process

### PR Hazırlama | Preparing Your PR

1. **Fork edin**: Depoyu fork edin
2. **Branch oluşturun**: Özellik branch'i oluşturun
   ```bash
   git checkout -b feature/awesome-feature
   # veya
   git checkout -b fix/bug-description
   ```
3. **Değişiklik yapın**: Kodunuzu yazın
4. **Test edin**: Değişikliklerinizi test edin
5. **Commit edin**: Değişikliklerinizi commit edin
6. **Push edin**: Branch'inizi push edin
7. **PR açın**: GitHub'da Pull Request açın

### PR Şablonu | PR Template

```markdown
## Açıklama | Description
<!-- Değişikliğin ne yaptığını açıklayın -->

## Tür | Type
- [ ] 🐛 Hata düzeltmesi | Bug fix
- [ ] 💡 Yeni özellik | New feature
- [ ] 📝 Dokümantasyon
- [ ] ♻️ Refactoring
- [ ] ⚡ Performans iyileştirmesi
- [ ] 🧪 Test

## Test Edildi mi? | Tested?
- [ ] Evet | Yes
- [ ] Hayır | No

## Ekran Görüntüleri | Screenshots
(UI değişiklikleri için)
```

### Review Süreci | Review Process

1. CI/CD kontrolleri otomatik olarak çalışır
2. En az bir maintainer tarafından review edilir
3. Review feedback'lerine yanıt verin
4. Gerekli değişiklikleri yapın
5. Approval alındığında merge edilir

---

## 📏 Kod Stili | Code Style

### Python

[PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide'ını takip edin ve aşağıdaki araçları kullanın:

- **Black**: Otomatik formatlama
- **isort**: Import sıralama
- **flake8**: Linting
- **mypy**: Tip kontrolü

```bash
# Formatlama
black .
isort .

# Linting
flake8 .
mypy .
```

### Değişken Adlandırma | Naming Conventions

- **Değişkenler**: `snake_case` (örn: `image_path`, `detection_results`)
- **Fonksiyonlar**: `snake_case` (örn: `detect_defects`, `classify_ore`)
- **Sınıflar**: `PascalCase` (örn: `DefectDetector`, `OreClassifier`)
- **Sabitler**: `UPPER_SNAKE_CASE` (örn: `MAX_IMAGE_SIZE`)

### Docstring Formatı | Docstring Format

```python
def function_name(param1: str, param2: int) -> bool:
    """Kısa açıklama.

    Uzun açıklama için gerektiğinde paragraf ekleyin.

    Args:
        param1: Parametrenin açıklaması.
        param2: Parametrenin açıklaması.

    Returns:
        Dönüş değerinin açıklaması.

    Raises:
        ValueError: Hata durumunun açıklaması.

    Example:
        >>> function_name("test", 5)
        True
    """
    pass
```

---

## ✉️ Commit Mesajları | Commit Messages

### Format

```
<tip>(<kapsam>): <kısa açıklama>

[isteğe bağlı gövde]

[isteğe bağlı footer]
```

### Tipler | Types

- `feat`: Yeni özellik
- `fix`: Hata düzeltmesi
- `docs`: Dokümantasyon değişiklikleri
- `style`: Kod formatı (semantik değişiklik yok)
- `refactor`: Kod refactoring
- `test`: Test ekleme/değiştirme
- `chore`: Bakım görevleri

### Örnekler | Examples

```
feat(defect): add YOLO11 model for surface defect detection

fix(ore): resolve classification accuracy issue for iron ore

docs(readme): add Turkish translation for quick start guide

refactor(utils): extract image preprocessing to separate module

test(classifier): add unit tests for ore classification
```

---

## 🧪 Test Yazımı | Writing Tests

### Test Konumu | Test Location

```
enterprise-vision-ai/
├── tests/
│   ├── unit/
│   │   ├── test_utils.py
│   │   └── test_models.py
│   ├── integration/
│   │   └── test_api.py
│   └── conftest.py
```

### Test Yazım Kuralları | Test Writing Guidelines

```python
import pytest
from pathlib import Path
from utils import detect_defects

class TestDefectDetection:
    """Defect detection modülü için testler."""

    @pytest.fixture
    def sample_image(self, tmp_path):
        """Test görüntüsü oluştur."""
        # Test için örnek görüntü oluştur
        pass

    def test_detect_defects_returns_list(self, sample_image):
        """Defect detection sonuçların liste döndürdüğünü doğrula."""
        result = detect_defects(sample_image)
        assert isinstance(result, list)

    def test_detect_defects_with_no_defects(self, sample_image):
        """Kusursuz görüntü için boş liste döndür."""
        result = detect_defects(sample_image)
        assert len(result) == 0
```

### Test Çalıştırma | Running Tests

```bash
# Tüm testleri çalıştır
pytest

# Belirli dosyayı test et
pytest tests/test_utils.py

# Coverage ile test et
pytest --cov=. --cov-report=html
```

---

## 📚 Dokümantasyon | Documentation

### Dokümantasyon Türleri | Types of Documentation

1. **Kod içi dokümantasyon**: Docstring'ler
2. **API dokümantasyonu**: REST API endpoint'leri
3. **Kullanıcı dokümantasyonu**: README, tutorials
4. **Geliştirici dokümantasyonu**: Architecture, contributing

### README Yapısı | README Structure

Her yeni modül için:
- Kısa açıklama
- Kurulum adımları
- Kullanım örnekleri
- API referansı

### Dokümantasyon Dili | Documentation Language

- Ana dokümantasyon dili: **English**
- Kullanıcı arayüzü: **Türkçe**
- Commit mesajları: İngilizce

---

## 📞 Yardım Alma | Getting Help

- **GitHub Discussions**: Sorularınız için
- **GitHub Issues**: Hata raporları için
- **Discord**: Gerçek zamanlı sohbet için

---

## 🙏 Teşekkürler | Acknowledgments

Katkıda bulunan herkese teşekkür ederiz!

Thank you to all contributors!

---

<div align="center">

**Bu proje topluluk tarafından yapılır | Made with ❤️ by the community**

</div>
