# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Ask Mode Specific Guidelines

### Turkish UI Language
All UI text, error messages, and documentation use Turkish. Common terms:
- **Defekt Tespiti** - Defect Detection page
- **Cevher Ön Seçimi** - Ore Pre-Selection/Classification page
- **Algılama Sonuçları** - Detection Results
- **Güven Eşiği** - Confidence Threshold

### Two Separate Apps
This project has two deployment targets:
1. **Streamlit app (root)**: `app.py`, runs with `streamlit run app.py`
   - Pages: `pages/01_Defekt_Tespiti.py`, `pages/02_Cevher_On_Secimi.py`
   - Uses classes: `manyetit`, `krom`, `atık`, `düşük tenör`, `çatlak`, `çizik`, `delik`

2. **Gradio app (huggingface_space/)**: `huggingface_space/app.py`
   - Uses different classes: `çizik`, `çatlak`, `delik`, `ezilme`, `yanık`, `pas`, `diğer`
   - Ore classes: `manyetit`, `kromit`, `pirit`, `kalkopirit`, `atık`, `düşük tenörlü`

### Model Registry
Models are defined in `models/registry.yaml`:
- YOLO11 segmentation models for defect detection
- Classification models for ore pre-selection
- Run `python models/download_models.py` to download models
