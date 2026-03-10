# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Code Mode Specific Guidelines

### Turkish Color Constants (utils.py)
Always use Turkish dictionary keys for mineral/ore class names:
- `manyetit` (magnetite), `krom` (chromite), `atık` (waste), `düşük tenör` (low grade), `defekt` (defect), `normal`

Defect type keys:
- `çatlak` (crack), `çizik` (scratch), `delik` (hole), `leke` (stain), `deformasyon` (deformation)

Severity levels:
- `düşük` (low), `orta` (medium), `yüksek` (high)

### YOLO11 Model Requirement
- Project requires Ultralytics v8+ (YOLO11 specifically)
- Use `yolo11n-seg.pt` as fallback model
- Do NOT use older YOLO versions (YOLOv5, YOLOv8 older versions may have compatibility issues)

### Streamlit Page Structure
Pages in `pages/` directory use numeric prefixes (`01_`, `02_`) for navigation ordering. Each page file is a thin wrapper that imports the actual implementation:
- `pages/01_Defekt_Tespiti.py` → imports from `pages/defect.py`
- `pages/02_Cevher_On_Secimi.py` → imports from `pages/ore.py`

### HuggingFace Space Differences
The `huggingface_space/` directory contains a separate Gradio-based application with different class names:
- Defect classes: `çizik`, `çatlak`, `delik`, `ezilme`, `yanık`, `pas`, `diğer`
- Ore classes: `manyetit`, `kromit`, `pirit`, `kalkopirit`, `atık`, `düşük tenörlü`
