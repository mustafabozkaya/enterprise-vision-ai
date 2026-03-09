# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project-Specific Non-Obvious Information

### Turkish UI Language
All code comments, UI text, error messages, and documentation are in Turkish. Common terms include:
- **Defekt Tespiti** - Defect Detection page
- **Cevher Ön Seçimi** - Ore Pre-Selection/Classification page
- **Algılama Sonuçları** - Detection Results
- **Güven Eşiği** - Confidence Threshold

### Turkish Color Constants (utils.py)
Mineral/ore class names use Turkish dictionary keys:
- `manyetit` (magnetite), `krom` (chromite), `atık` (waste), `düşük tenör` (low grade), `defekt` (defect), `normal`

Defect type keys:
- `çatlak` (crack), `çizik` (scratch), `delik` (hole), `leke` (stain), `deformasyon` (deformation)

Severity levels:
- `düşük` (low), `orta` (medium), `yüksek` (high)

### YOLO Model Requirements
The project requires Ultralytics v8+ (YOLO11/YOLO26). The model loading priority is:

**Defect Detection (pages/defect.py):**
1. `yolo26-seg.pt` (tries first)
2. `yolo11s-seg.pt` (fallback)
3. `yolo11n-seg.pt` (last resort)

**Ore Classification (pages/ore.py):**
1. `yolo11s-seg.pt` (tries first)
2. `yolo11n-seg.pt` (fallback)

The package is `enterprise_vision_ai` and models should be downloaded using `python models/download_models.py`.

### Severity Level Calculation
Severity levels are calculated based on the anomaly score (0-100):
- **düşük (low)**: score < 30
- **orta (medium)**: 30 <= score < 70
- **yüksek (high)**: score >= 70

The anomaly score is calculated from the average confidence of detected defects.

### Class Names

This project has two different applications with different class names:

| Application | Defect Classes | Ore Classes |
|------------|----------------|-------------|
| Streamlit (main app) | `çatlak`, `çizik`, `delik`, `leke`, `deformasyon` | `manyetit`, `krom`, `atık`, `düşük tenör` |
| HuggingFace Space | `çizik`, `çatlak`, `delik`, `ezilme`, `yanık`, `pas`, `diğer` | `manyetit`, `kromit`, `pirit`, `kalkopirit`, `atık`, `düşük tenörlü` |

**Note:** When working on code, use the appropriate class names based on which application you're modifying:
- Streamlit app: `app.py`, `pages/defect.py`, `pages/ore.py`
- HuggingFace Space: `huggingface_space/app.py`

### Streamlit Page Structure
Pages in `pages/` directory use numeric prefixes (`01_`, `02_`) for navigation ordering. Each page file is a thin wrapper that imports the actual implementation:
- `pages/01_Defekt_Tespiti.py` → imports from `pages/defect.py`
- `pages/02_Cevher_On_Secimi.py` → imports from `pages/ore.py`

### HuggingFace Space Separation
The `huggingface_space/` directory contains a separate Gradio-based application (not Streamlit). It has its own:
- `app.py` - Main Gradio application
- `requirements.txt` - Separate dependencies
- Uses different class names than the Streamlit application (see Class Name Sets above)

