"""
Image Preprocessing Service
Görüntü ön işleme fonksiyonları
"""

import base64
import io
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


class ImagePreprocessor:
    """
    Görüntü ön işleme servisi.

    YOLO modeli için görüntüleri ön işler.
    """

    # Varsayılan boyutlar
    DEFAULT_SIZE = (640, 640)
    MIN_SIZE = 320
    MAX_SIZE = 1280

    # İzin verilen formatlar
    ALLOWED_FORMATS = ["JPEG", "PNG", "JPG", "WEBP", "BMP"]

    def __init__(self):
        """Preprocessor başlatır"""
        pass

    def preprocess(
        self,
        image: Union[np.ndarray, Image.Image, bytes, str],
        target_size: Tuple[int, int] = DEFAULT_SIZE,
        normalize: bool = False,
        maintain_aspect_ratio: bool = True,
    ) -> np.ndarray:
        """
        Görüntüyü ön işler.

        Args:
            image: Giriş görüntüsü
            target_size: Hedef boyut
            normalize: Normalizasyon uygula
            maintain_aspect_ratio: En-boy oranını koru

        Returns:
            np.ndarray: İşlenmiş görüntü
        """
        # Görüntüyü numpy array'e çevir
        image = self._to_numpy(image)

        # Boyutlandır
        if maintain_aspect_ratio:
            image = self._resize_with_aspect_ratio(image, target_size)
        else:
            image = cv2.resize(image, target_size)

        # Normalize et
        if normalize:
            image = image.astype(np.float32) / 255.0

        return image

    def _to_numpy(self, image: Union[np.ndarray, Image.Image, bytes, str]) -> np.ndarray:
        """
        Görüntüyü numpy array'e çevirir.

        Args:
            image: Giriş görüntüsü

        Returns:
            np.ndarray: Numpy dizisi
        """
        if isinstance(image, np.ndarray):
            return image

        elif isinstance(image, Image.Image):
            return np.array(image)

        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
            return np.array(image)

        elif isinstance(image, str):
            if image.startswith("data:image"):
                # Base64 data URL
                base64_data = image.split(",")[1]
                image = base64.b64decode(base64_data)
                image = Image.open(io.BytesIO(image))
                return np.array(image)
            elif image.startswith("http"):
                # URL
                image = Image.open(io.BytesIO(requests.get(image).content))
                return np.array(image)
            else:
                # Dosya yolu
                image = Image.open(image)
                return np.array(image)

        raise ValueError(f"Desteklenmeyen görüntü tipi: {type(image)}")

    def _resize_with_aspect_ratio(
        self, image: np.ndarray, target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        En-boy oranını koruyarak yeniden boyutlandırır.

        Args:
            image: Giriş görüntüsü
            target_size: Hedef boyut

        Returns:
            np.ndarray: Yeniden boyutlandırılmış görüntü
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size

        # Ölçek faktörünü hesapla
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Yeniden boyutlandır
        resized = cv2.resize(image, (new_w, new_h))

        # Gerekirse padding ekle
        if new_w < target_w or new_h < target_h:
            # Yeni görüntü oluştur (siyah padding)
            result = np.zeros((target_h, target_w, 3), dtype=image.dtype)

            # Ortala
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2

            result[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
            return result

        return resized

    def validate_image(
        self, image: Union[np.ndarray, Image.Image, bytes]
    ) -> Tuple[bool, Optional[str]]:
        """
        Görüntüyü doğrular.

        Args:
            image: Görüntü

        Returns:
            Tuple[bool, Optional[str]]: Geçerli mi, hata mesajı
        """
        try:
            # Numpy array'e çevir
            arr = self._to_numpy(image)

            # Boyut kontrolü
            if len(arr.shape) < 2:
                return False, "Görüntü boyutu geçersiz"

            h, w = arr.shape[:2]

            if h < 32 or w < 32:
                return False, "Görüntü çok küçük (min 32x32)"

            if h > 4096 or w > 4096:
                return False, "Görüntü çok büyük (max 4096x4096)"

            return True, None

        except Exception as e:
            return False, str(e)

    def enhance_contrast(
        self, image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)
    ) -> np.ndarray:
        """
        Görüntü kontrastını artırır (CLAHE).

        Args:
            image: Giriş görüntüsü
            clip_limit: CLAHE clip limit
            tile_grid_size: Tile grid boyutu

        Returns:
            np.ndarray: İyileştirilmiş görüntü
        """
        # Griye çevir
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # CLAHE uygula
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(gray)

        # RGB'ye geri çevir
        if len(image.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        return enhanced

    def remove_noise(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Gürültüyü azaltır.

        Args:
            image: Giriş görüntüsü
            kernel_size: Kernel boyutu

        Returns:
            np.ndarray: Gürültüsü azaltılmış görüntü
        """
        return cv2.bilateralFilter(image, kernel_size, 75, 75)

    def apply_sharpen(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Keskinleştirme uygular.

        Args:
            image: Giriş görüntüsü
            strength: Keskinleştirme strength

        Returns:
            np.ndarray: Keskinleştirilmiş görüntü
        """
        # Kernel oluştur
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) * strength / 9

        # Uygula
        return cv2.filter2D(image, -1, kernel)

    def normalize_lighting(self, image: np.ndarray) -> np.ndarray:
        """
        Aydınlatmayı normalize eder.

        Args:
            image: Giriş görüntüsü

        Returns:
            np.ndarray: Normalize edilmiş görüntü
        """
        # LAB renk uzayına çevir
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # L kanalını CLAHE ile iyileştir
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # Birleştir
        lab = cv2.merge([l, a, b])

        # BGR'ye geri çevir
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return result

    def to_base64(self, image: np.ndarray, format: str = "JPEG", quality: int = 95) -> str:
        """
        Görüntüyü base64'e çevirir.

        Args:
            image: Giriş görüntüsü
            format: Çıktı formatı
            quality: Kalite

        Returns:
            str: Base64 string
        """
        # PIL Image'a çevir
        if len(image.shape) == 3 and image.shape[2] == 3:
            # BGR'den RGB'ye
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(image)

        # Bytes'a çevir
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format, quality=quality)

        # Base64'e çevir
        return base64.b64encode(buffer.getvalue()).decode()


# Lazy import
import requests
