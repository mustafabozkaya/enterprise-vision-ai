"""
Enterprise Vision AI MVP - Ortak Fonksiyonlar
Defekt Tespiti ve Cevher Ön Seçimi için yardımcı fonksiyonlar
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# =============================================================================
# RENK FONKSİYONLARI
# =============================================================================


def get_ore_class_colors() -> Dict[str, Tuple[int, int, int]]:
    """
    Cevher sınıfı renklerini döndürür.

    Returns:
        Dict: Sınıf adı -> RGB renk tuple
    """
    return {
        "manyetit": (220, 20, 60),  # Kırmızı
        "krom": (0, 255, 127),  # Yeşil
        "atık": (128, 128, 128),  # Gri
        "düşük tenör": (255, 165, 0),  # Turuncu
        "defekt": (255, 0, 255),  # Magenta
        "normal": (0, 255, 0),  # Açık yeşil
    }


def get_defect_colors() -> Dict[str, Tuple[int, int, int]]:
    """
    Defekt türü renklerini döndürür.

    Returns:
        Dict: Defekt türü -> RGB renk tuple
    """
    return {
        "çatlak": (255, 0, 0),  # Kırmızı
        "çizik": (0, 255, 255),  # Sarı
        "delik": (255, 165, 0),  # Turuncu
        "leke": (128, 0, 128),  # Mor
        "deformasyon": (0, 0, 255),  # Mavi
    }


def get_severity_color(severity: str) -> Tuple[int, int, int]:
    """
    Severity seviyesine göre renk döndürür.

    Args:
        severity: 'düşük', 'orta', 'yüksek'

    Returns:
        Tuple: RGB renk
    """
    colors = {
        "düşük": (0, 255, 0),  # Yeşil
        "orta": (255, 165, 0),  # Turuncu
        "yüksek": (255, 0, 0),  # Kırmızı
    }
    return colors.get(severity.lower(), (128, 128, 128))


# =============================================================================
# GÖRSELLEŞTİRME FONKSİYONLARI
# =============================================================================


def draw_annotations(
    image: np.ndarray,
    results,
    class_colors: Dict[str, Tuple[int, int, int]],
    show_labels: bool = True,
    show_confidence: bool = True,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    YOLO sonuçlarını görselleştirir.

    Args:
        image: Giriş görüntüsü (BGR format)
        results: YOLO model sonuçları
        class_colors: Sınıf renkleri dict
        show_labels: Etiketleri göster
        show_confidence: Güven skorlarını göster
        line_thickness: Çizgi kalınlığı

    Returns:
        np.ndarray: Açıklama eklenmiş görüntü
    """
    result_image = image.copy()

    # RGB'ye çevir (OpenCV BGR -> RGB)
    if len(result_image.shape) == 3 and result_image.shape[2] == 3:
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    if results is None or len(results) == 0:
        return result_image

    # Her bir sonuç için
    for result in results:
        # Segmentasyon maskesi varsa
        if hasattr(result, "masks") and result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.data.cpu().numpy()

            for mask, box in zip(masks, boxes):
                cls_id = int(box[5])
                conf = float(box[4])

                # Sınıf adını al
                if hasattr(result, "names") and result.names:
                    class_name = result.names.get(cls_id, f"class_{cls_id}")
                else:
                    class_name = f"class_{cls_id}"

                # Renk seç
                color = class_colors.get(class_name, (255, 255, 255))

                # Maskeyi renklendir
                mask_uint8 = (mask * 255).astype(np.uint8)

                # Debug: Print shapes
                # print(f"Mask shape: {mask_uint8.shape}, result_image shape: {result_image.shape}")

                # Maskeyi orijinal görüntü boyutuna yeniden boyutlandır
                target_h, target_w = result_image.shape[:2]
                mask_resized = cv2.resize(
                    mask_uint8, (target_w, target_h), interpolation=cv2.INTER_LINEAR
                )

                # mask_resized'i 3 kanala genişlet
                mask_resized_3ch = np.stack([mask_resized] * 3, axis=-1)

                mask_colored = np.zeros_like(result_image)
                mask_colored[:, :, 0] = mask_resized_3ch[:, :, 0]
                mask_colored[:, :, 1] = mask_resized_3ch[:, :, 1]
                mask_colored[:, :, 2] = mask_resized_3ch[:, :, 2]

                # Maskeyi görüntüye uygula
                result_image = cv2.addWeighted(result_image, 1, mask_colored, 0.3, 0)

        # Bounding box'lar
        if hasattr(result, "boxes") and result.boxes is not None:
            boxes = result.boxes.data.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2, conf, cls_id = box
                cls_id = int(cls_id)

                # Sınıf adı
                if hasattr(result, "names") and result.names:
                    class_name = result.names.get(cls_id, f"class_{cls_id}")
                else:
                    class_name = f"class_{cls_id}"

                # Renk
                color = class_colors.get(class_name, (255, 255, 255))

                # Box çiz
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, line_thickness)

                # Etiket oluştur
                if show_labels:
                    label = class_name
                    if show_confidence:
                        label += f" {conf:.2f}"

                    # Label arka plan
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(
                        result_image, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1
                    )
                    cv2.putText(
                        result_image,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

    return result_image


def draw_severity_indicator(image: np.ndarray, severity: str, score: float) -> np.ndarray:
    """
    Görüntüye severity indicator ekler.

    Args:
        image: Giriş görüntüsü
        severity: Severity seviyesi
        score: Anomali skoru

    Returns:
        np.ndarray: İndeks eklenmiş görüntü
    """
    result = image.copy()
    h, w = result.shape[:2]

    # Arka plan
    color = get_severity_color(severity)

    # Indicator box
    box_w, box_h = 200, 80
    x, y = w - box_w - 20, 20

    # Yarı saydam arka plan
    overlay = result.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), color, -1)
    cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)

    # Border
    cv2.rectangle(result, (x, y), (x + box_w, y + box_h), color, 3)

    # Text
    cv2.putText(
        result,
        f"Severity: {severity.upper()}",
        (x + 10, y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        result,
        f"Score: {score:.1f}",
        (x + 10, y + 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    return result


# =============================================================================
# METRİK FONKSİYONLARI
# =============================================================================


def calculate_anomaly_score(results) -> float:
    """
    Anomali skorunu hesaplar.

    Args:
        results: YOLO model sonuçları

    Returns:
        float: Anomali skoru (0-100)
    """
    if results is None or len(results) == 0:
        return 0.0

    total_score = 0.0
    count = 0

    for result in results:
        if hasattr(result, "boxes") and result.boxes is not None:
            boxes = result.boxes.data.cpu().numpy()

            for box in boxes:
                conf = float(box[4])
                # Yüksek güvenilirlik = yüksek anomali
                total_score += conf * 100
                count += 1

    if count == 0:
        return 0.0

    return min(100.0, total_score / count)


def get_severity_level(score: float) -> str:
    """
    Anomali skoruna göre severity seviyesi döndürür.

    Args:
        score: Anomali skoru (0-100)

    Returns:
        str: 'düşük', 'orta', 'yüksek'
    """
    if score < 30:
        return "düşük"
    elif score < 70:
        return "orta"
    else:
        return "yüksek"


def get_maintenance_recommendation(score: float) -> str:
    """
    Anomali skoruna göre bakım önerisi döndürür.

    Args:
        score: Anomali skoru (0-100)

    Returns:
        str: Bakım önerisi
    """
    if score < 30:
        return "✅ Normal çalışma koşulları. Rutin bakım yeterli."
    elif score < 50:
        return "⚠️ Hafif anomali tespit edildi. Yakın izleme önerilir."
    elif score < 70:
        return "🔶 Orta düzey anomali. Planlı bakım zamanlaması önerilir."
    elif score < 85:
        return "🔴 Yüksek anomali. Acil bakım önerilir."
    else:
        return "🚨 Kritik! Üretim durdurularak bakım yapılmalı."


def calculate_ore_metrics(results) -> Dict[str, int]:
    """
    Cevher sınıflandırma metriklerini hesaplar.

    Args:
        results: YOLO model sonuçları

    Returns:
        Dict: Sınıf adı -> sayı
    """
    counts = {"manyetit": 0, "krom": 0, "atık": 0, "düşük tenör": 0}

    if results is None or len(results) == 0:
        return counts

    for result in results:
        if hasattr(result, "boxes") and result.boxes is not None:
            boxes = result.boxes.data.cpu().numpy()

            for box in boxes:
                cls_id = int(box[5])

                if hasattr(result, "names") and result.names:
                    class_name = result.names.get(cls_id, "").lower()

                    for key in counts.keys():
                        if key in class_name:
                            counts[key] += 1
                            break

    return counts


def calculate_metal_ratio(counts: Dict[str, int]) -> float:
    """
    Metal oranını hesaplar.

    Args:
        counts: Sınıf sayıları

    Returns:
        float: Metal oranı (0-100)
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0

    # Değerli cevherler: manyetit, krom
    valuable = counts.get("manyetit", 0) + counts.get("krom", 0)

    return (valuable / total) * 100


def get_diverter_recommendation(metal_ratio: float) -> str:
    """
    Diverter önerisi döndürür.

    Args:
        metal_ratio: Metal oranı (0-100)

    Returns:
        str: Diverter önerisi
    """
    if metal_ratio >= 60:
        return "⬆️ İLERİ İŞLEME: Yüksek metal oranı. Üretime gönder."
    elif metal_ratio >= 30:
        return "⏸️ KONTROL: Orta metal oranı. Ek analiz gerekli."
    else:
        return "⬇️ ATIK: Düşük metal oranı. Atık hattına yönlendir."


def create_metrics_dataframe(data: List[Dict]) -> pd.DataFrame:
    """
    Metrik DataFrame oluşturur.

    Args:
        data: Metrik verileri listesi

    Returns:
        pd.DataFrame: Metrik DataFrame
    """
    if not data:
        return pd.DataFrame(columns=["Zaman", "Anomali Skoru", "Severity"])

    df = pd.DataFrame(data)
    return df


def create_ore_dataframe(counts: Dict[str, int]) -> pd.DataFrame:
    """
    Cevher dağılımı DataFrame oluşturur.

    Args:
        counts: Sınıf sayıları

    Returns:
        pd.DataFrame: Cevher DataFrame
    """
    return pd.DataFrame(list(counts.items()), columns=["Sınıf", "Adet"])


# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================


def load_image(file) -> Optional[np.ndarray]:
    """
    Görüntü dosyasını yükler.

    Args:
        file: Streamlit file upload object

    Returns:
        np.ndarray: Görüntü dizisi veya None
    """
    try:
        image = Image.open(file)
        image = np.array(image)

        # RGB'den BGR'ye çevir
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return image
    except Exception as e:
        print(f"Görüntü yükleme hatası: {e}")
        return None


def resize_image(image: np.ndarray, max_size: int = 1280) -> np.ndarray:
    """
    Görüntüyü yeniden boyutlandırır.

    Args:
        image: Giriş görüntüsü
        max_size: Maksimum kenar uzunluğu

    Returns:
        np.ndarray: Yeniden boyutlandırılmış görüntü
    """
    h, w = image.shape[:2]

    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h))

    return image


def preprocess_for_model(image: np.ndarray) -> np.ndarray:
    """
    Model için görüntü ön işleme.

    Args:
        image: Giriş görüntüsü

    Returns:
        np.ndarray: İşlenmiş görüntü
    """
    # Boyutlandır
    image = resize_image(image)

    return image


def create_dummy_results(class_name: str = "defect", count: int = 1) -> List:
    """
    Model olmadığında dummy sonuç oluşturur.

    Args:
        class_name: Sınıf adı
        count: Nesne sayısı

    Returns:
        List: Dummy sonuçlar
    """
    # Bu fonksiyon gerçek bir dummy sonuç döndürmez
    # Sadece API uyumluluğu için
    return []


def format_time(seconds: float) -> str:
    """
    Saniyeyi HH:MM:SS formatına çevirir.

    Args:
        seconds: Saniye

    Returns:
        str: Formatlanmış zaman
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
