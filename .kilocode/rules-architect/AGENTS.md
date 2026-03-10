# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Architect Mode Specific Guidelines

### YOLO11 for Segmentation
This project uses YOLO11 (Ultralytics v8+) for instance segmentation, not just detection:
- `yolo11n-seg.pt` is the fallback model
- Segmentation provides pixel-level masks for defect localization
- Model configuration in `models/registry.yaml`

### Two Deployment Targets
The project supports two deployment frameworks:
1. **Streamlit** (root): Main production app, Python-based
   - Pages in `pages/` directory with numeric prefixes
   - Fast development iteration, good for internal tools

2. **HuggingFace Space** (`huggingface_space/`): Gradio-based
   - Separate class labels (7 defect classes, 6 ore classes)
   - Optimized for cloud deployment with hardware.yaml
   - Better for public demo/API access

### Mineral Detection Domain
This is a mining/computer vision application with Turkish classification:
- **Defect types**: `çatlak` (crack), `çizik` (scratch), `delik` (hole), `leke` (stain), `deformasyon` (deformation)
- **Ore types**: `manyetit` (magnetite), `krom` (chromite), `atık` (waste), `düşük tenör` (low grade)
- Severity levels: `düşük` (low), `orta` (medium), `yüksek` (high)

### Dataset Structure
- `datasets/defect_detection/dataset.yaml` - YOLO format dataset config
- Train/validation split for defect segmentation
- Consider adding more mineral types for expansion
