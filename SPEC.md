# Enterprise Vision AI - Project Specification

> **Version:** 2.0.0-Enterprise  
> **Last Updated:** 2026-03-08  
> **Status:** Draft for Review

---

## 1. Executive Summary

### 1.1 Project Vision

Enterprise Vision AI (Industrial Computer Vision) is an open-source enterprise-grade computer vision platform designed for real-time defect detection and ore classification in manufacturing and mining environments. The project aims to become the leading open-source solution for industrial quality control, providing scalable, production-ready AI capabilities that can be deployed on edge devices, cloud infrastructure, or hybrid environments.

### 1.2 Mission Statement

To democratize industrial AI by providing a free, open-source, production-grade computer vision platform that enables manufacturers and mining operators to implement intelligent quality control systems without proprietary vendor lock-in.

### 1.3 Core Values

| Value | Description |
|-------|-------------|
| **Openness** | Full transparency in algorithms, training pipelines, and model architectures |
| **Scalability** | Design for deployment from single-camera edge devices to multi-site enterprise deployments |
| **Performance** | Real-time inference with minimal latency for production line integration |
| **Community** | Foster contributions from researchers, engineers, and industry experts |
| **Reliability** | Enterprise-grade stability with comprehensive testing and validation |

### 1.4 Current State vs. Target State

```mermaid
graph LR
    A[MVP v1.0] -->|Transform| B[Enterprise v2.0]
    B --> C[Platform v3.0]
    
    subgraph Current
        A1[Streamlit Only]
        A2[2 Modules]
        A3[Basic Model Loading]
        A4[No CI/CD]
    end
    
    subgraph Target v2.0
        B1[Modular Architecture]
        B2[6+ Modules]
        B3[MLflow Integration]
        B4[Full CI/CD]
    end
    
    subgraph Platform v3.0
        C1[Plugin System]
        C2[Multi-modal AI]
        C3[AutoML]
        C4[Enterprise SaaS]
    end
```

---

## 2. Technical Architecture

### 2.1 System Architecture Overview

```mermaid
graph TB
    subgraph "Client Layer"
        Web[Web Application]
        Mobile[Mobile App]
        API[REST API]
        Edge[Edge Devices]
    end
    
    subgraph "Application Layer"
        Streamlit[Streamlit UI]
        API_Server[FastAPI Backend]
        WebSocket[WebSocket Server]
    end
    
    subgraph "Core Services"
        ML_Service[ML Inference Service]
        Preprocessor[Image Preprocessing]
        Postprocessor[Result Postprocessing]
        Scheduler[Task Scheduler]
    end
    
    subgraph "Model Layer"
        YOLO[YOLO11 Models]
        Transform[Vision Transformers]
        Registry[Model Registry]
        Versioning[MLflow Versioning]
    end
    
    subgraph "Data Layer"
        Dataset[Dataset Manager]
        Cache[Redis Cache]
        Storage[Object Storage]
        DB[(PostgreSQL)]
    end
    
    subgraph "Infrastructure"
        Docker[Docker/Kubernetes]
        HF[HuggingFace Spaces]
        Cloud[Cloud Providers]
    end
    
    Web --> Streamlit
    Mobile --> API
    API --> API_Server
    Edge --> WebSocket
    
    Streamlit --> ML_Service
    API_Server --> ML_Service
    WebSocket --> ML_Service
    
    ML_Service --> YOLO
    ML_Service --> Transform
    ML_Service --> Registry
    Registry --> Versioning
    
    Preprocessor --> Dataset
    Postprocessor --> Cache
    ML_Service --> Storage
    ML_Service --> DB
```

### 2.2 Component Architecture

```mermaid
classDiagram
    class BaseModel {
        <<abstract>>
        +load()
        +predict()
        +export()
    }
    
    class YOLOAdapter {
        -model: YOLO
        -config: ModelConfig
        +load()
        +predict()
        +predict_video()
        +export_onnx()
    }
    
    class DefectDetector {
        -model: YOLOAdapter
        -classes: List[str]
        +detect()
        +calculate_severity()
        +generate_report()
    }
    
    class OreClassifier {
        -model: YOLOAdapter
        -classes: List[str]
        +classify()
        +calculate_metal_ratio()
        +get_diverter_action()
    }
    
    class ModelRegistry {
        -storage: ModelStorage
        -metadata: MetadataStore
        +register()
        +get_latest()
        +get_by_version()
        +list_models()
    }
    
    class InferenceEngine {
        -models: Dict[str, BaseModel]
        -cache: InferenceCache
        +run()
        +batch_run()
        +stream_run()
    }
    
    BaseModel <|-- YOLOAdapter
    YOLOAdapter <|-- DefectDetector
    YOLOAdapter <|-- OreClassifier
    InferenceEngine --> BaseModel
    InferenceEngine --> ModelRegistry
```

### 2.3 Technology Stack

#### Core Technologies

| Category | Technology | Version | Purpose |
|----------|------------|---------|---------|
| **Frontend** | Streamlit | ≥1.35.0 | Web UI Framework |
| **Frontend** | React | ≥18.2.0 | Alternative Web UI |
| **Backend** | FastAPI | ≥0.115.0 | REST API Server |
| **ML Framework** | Ultralytics | ≥8.3.0 | YOLO Implementation |
| **ML Framework** | PyTorch | ≥2.5.0 | Deep Learning Backend |
| **Computer Vision** | OpenCV | ≥4.10.0 | Image Processing |
| **Visualization** | Plotly | ≥5.24.0 | Charts and Graphs |

#### Data & Storage

| Category | Technology | Version | Purpose |
|----------|------------|---------|---------|
| **Database** | PostgreSQL | ≥16.0 | Primary Database |
| **Cache** | Redis | ≥7.4 | Inference Caching |
| **Object Storage** | MinIO/S3 | Latest | Model & Data Storage |
| **Model Registry** | MLflow | ≥2.19.0 | Model Versioning |

#### DevOps & Infrastructure

| Category | Technology | Version | Purpose |
|----------|------------|---------|---------|
| **Container** | Docker | ≥26.0 | Application Packaging |
| **Orchestration** | Kubernetes | ≥1.30 | Container Orchestration |
| **CI/CD** | GitHub Actions | Latest | Automation |
| **HF Spaces** | Gradio | ≥4.44.0 | Demo Platform |

---

## 3. GitHub Project Structure

### 3.1 Recommended Repository Layout

```
bas-endustriyel-ai/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   ├── feature_request.md
│   │   └── config.yml
│   ├── workflows/
│   │   ├── ci.yml
│   │   ├── release.yml
│   │   ├── docker.yml
│   │   └── hf-deploy.yml
│   └── dependabot.yml
├── .gitignore
├── .dockerignore
├── .env.example
├── .pre-commit-config.yaml
├── LICENSE
├── README.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── pyproject.toml
├── setup.py
├── requirements.txt
├── requirements-dev.txt
├── requirements-test.txt
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.api
│   ├── Dockerfile.worker
│   ├── docker-compose.yml
│   └── nginx.conf
├── kubernetes/
│   ├── base/
│   ├── overlays/
│   └── kustomization.yaml
├── docs/
│   ├── index.md
│   ├── getting-started.md
│   ├── architecture.md
│   ├── api-reference.md
│   ├── models.md
│   ├── deployment.md
│   ├── contributing.md
│   └── notebooks/
│       ├── training.ipynb
│       ├── evaluation.ipynb
│       └── deployment.ipynb
├── src/
│   └── bas_ai/
│       ├── __init__.py
│       ├── __version__.py
│       ├── main.py
│       ├── config.py
│       ├── constants.py
│       ├── exceptions.py
│       ├── logging.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── main.py
│       │   ├── routes/
│       │   │   ├── __init__.py
│       │   │   ├── health.py
│       │   │   ├── inference.py
│       │   │   ├── models.py
│       │   │   └── datasets.py
│       │   ├── schemas/
│       │   │   ├── __init__.py
│       │   │   ├── inference.py
│       │   │   └── models.py
│       │   └── dependencies.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── model_loader.py
│       │   ├── inference_engine.py
│       │   ├── preprocessor.py
│       │   └── postprocessor.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── yolo_adapter.py
│       │   ├── defect_detector.py
│       │   └── ore_classifier.py
│       ├── services/
│       │   ├── __init__.py
│       │   ├── defect_service.py
│       │   ├── ore_service.py
│       │   ├── dataset_service.py
│       │   └── model_registry.py
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── image_utils.py
│       │   ├── video_utils.py
│       │   ├── visualization.py
│       │   ├── metrics.py
│       │   └── io_utils.py
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── s3_storage.py
│       │   └── local_storage.py
│       └── database/
│           ├── __init__.py
│           ├── connection.py
│           ├── models.py
│           └── repositories/
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── fixtures/
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_models/
│   │   ├── test_services/
│   │   └── test_utils/
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_api/
│   │   └── test_inference/
│   └── e2e/
│       ├── __init__.py
│       └── test_full_pipeline.py
├── notebooks/
│   ├── exploration/
│   ├── training/
│   └── evaluation/
├── weights/
│   ├── defect/
│   │   ├── baseline/
│   │   └── experiments/
│   └── ore/
│       ├── baseline/
│       └── experiments/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── datasets/
│   └── samples/
├── scripts/
│   ├── train_defect.py
│   ├── train_ore.py
│   ├── evaluate.py
│   ├── export_model.py
│   ├── prepare_dataset.py
│   └── benchmark.py
├── configs/
│   ├── default.yaml
│   ├── development.yaml
│   ├── production.yaml
│   └── model_configs/
│       ├── yolo11n.yaml
│       ├── yolo11s.yaml
│       ├── yolo11m.yaml
│       └── yolo11l.yaml
├── mlflow/
│   ├── mlruns/
│   └── mlartifacts/
├── examples/
│   ├── python/
│   ├── api/
│   └── streamlit/
├── .streamlit/
│   ├── config.toml
│   └── theme.css
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── pages/
│   │   ├── __init__.py
│   │   ├── home.py
│   │   ├── defect_detection.py
│   │   ├── ore_classification.py
│   │   ├── model_management.py
│   │   ├── dataset_manager.py
│   │   ├── analytics.py
│   │   └── settings.py
│   └── components/
│       ├── __init__.py
│       ├── sidebar.py
│       ├── header.py
│       ├── uploaders.py
│       ├── viewers.py
│       └── charts.py
└── huggingface/
    ├── app.py
    ├── requirements.txt
    ├── README.md
    └── hardware.yaml
```

### 3.2 File Descriptions

| Path | Description |
|------|-------------|
| `.github/workflows/ci.yml` | Continuous Integration pipeline |
| `.github/workflows/release.yml` | Release and version management |
| `docker/Dockerfile` | Multi-stage build for production |
| `docker/docker-compose.yml` | Local development environment |
| `src/bas_ai/` | Core package with modular structure |
| `tests/` | Comprehensive test suite |
| `configs/` | Environment-specific configurations |
| `huggingface/` | HuggingFace Spaces deployment files |

---

## 4. Feature Roadmap

### 4.1 Version 1.0 - Foundation (Current MVP)

**Status:** ✅ Completed  
**Timeline:** Q1 2026

| Feature | Description | Status |
|---------|-------------|--------|
| Basic UI | Streamlit-based interface | ✅ |
| Defect Detection | YOLO11-based surface defect detection | ✅ |
| Ore Classification | Mineral classification (Magnetite, Chrome, Waste) | ✅ |
| Image Upload | Single/multiple image inference | ✅ |
| Video Processing | Video file upload and analysis | ✅ |
| Visualization | Bounding boxes, masks, confidence scores | ✅ |
| Basic Metrics | Anomaly scoring, severity levels | ✅ |

### 4.2 Version 2.0 - Enterprise (Q2-Q3 2026)

**Focus:** Production readiness, scalability, and enterprise features

#### 2.0.1 - Core Infrastructure (Month 1-2)

| Feature | Description | Priority | Status |
|---------|-------------|----------|--------|
| REST API | FastAPI-based inference API | P0 | ✅ Implemented |
| Model Registry | MLflow integration for version control | P0 | ⏳ Planned |
| Database | PostgreSQL for metadata and logs | P0 | ⏳ Planned |
| Redis Caching | Inference result caching | P1 | ⏳ Planned |
| Docker Support | Containerized deployment | P0 | ✅ Implemented |
| Kubernetes Configs | K8s manifests for production | P1 | ⏳ Planned |

#### 2.0.2 - Advanced Features (Month 2-3)

| Feature | Description | Priority | Status |
|---------|-------------|----------|--------|
| Real-time Streaming | RTSP/IP camera integration | P0 | ✅ Implemented |
| Batch Processing | Large dataset inference | P0 | ✅ Implemented |
| Multi-model Support | Switch between model versions | P1 | ✅ Implemented |
| WebSocket Support | Real-time inference updates | P1 | ⏳ Planned |
| Model Export | ONNX/TensorRT export | P1 | ⏳ Planned |
| Performance Monitoring | Latency, throughput tracking | P2 | ⏳ Planned |

#### 2.0.3 - Enterprise Features (Month 3-4)

| Feature | Description | Priority |
|---------|-------------|----------|
| User Authentication | OAuth2/JWT authentication | P1 |
| Role-based Access | Admin, Operator, Viewer roles | P1 |
| Audit Logging | Complete action logging | P1 |
| Alerting System | Email/Slack notifications | P2 |
| Report Generation | PDF/CSV export reports | P2 |
| API Rate Limiting | Request throttling | P2 |

#### 2.0.4 - Training Pipeline (Month 4-5)

| Feature | Description | Priority |
|---------|-------------|----------|
| Dataset Manager | Upload, validate, organize datasets | P0 |
| Training Scripts | End-to-end training pipeline | P1 |
| Evaluation Framework | Comprehensive metrics and reports | P1 |
| Experiment Tracking | MLflow experiments | P1 |
| Model Benchmarking | Compare model performance | P2 |

### 4.3 Version 3.0 - Platform (Q4 2026+)

**Focus:** Platform expansion, automation, and ecosystem

| Feature | Description | Priority |
|---------|-------------|----------|
| Plugin System | Third-party model integration | P1 |
| AutoML | Automated model selection and tuning | P2 |
| Edge Deployment | NVIDIA Jetson, Raspberry Pi support | P1 |
| Mobile App | iOS/Android companion app | P2 |
| Multi-modal AI | Text, audio insights from video | P2 |
| Federated Learning | Privacy-preserving training | P2 |
| SaaS Platform | Cloud-hosted multi-tenant solution | P2 |

### 4.4 Roadmap Timeline

```mermaid
gantt
    title Enterprise Vision AI Development Roadmap
    dateFormat  YYYY-MM-DD
    
    section v1.0 Foundation
    MVP Development       :done,    des1, 2026-01-01, 2026-02-28
    
    section v2.0 Enterprise
    Core Infrastructure   :active,  des2, 2026-03-01, 2026-04-30
    Advanced Features     :         des3, 2026-05-01, 2026-06-30
    Enterprise Features   :         des4, 2026-07-01, 2026-08-31
    Training Pipeline     :         des5, 2026-09-01, 2026-10-31
    
    section v3.0 Platform
    Platform Development  :         des6, 2026-11-01, 2027-03-31
```

---

## 5. HuggingFace Integration

### 5.1 Space Architecture

```mermaid
graph LR
    subgraph "HuggingFace Space"
        HF_UI[Gradio Interface]
        HF_API[Inference API]
        HF_Cache[Model Cache]
        HF_Storage[Static Storage]
    end
    
    subgraph "Model Hub"
        HF_Model[🤗 Model Hub]
        HF_Repo[Model Repository]
    end
    
    subgraph "External"
        S3[AWS S3 / MinIO]
        MLflow[MLflow Server]
    end
    
    User --> HF_UI
    HF_UI --> HF_API
    HF_API --> HF_Cache
    HF_Cache --> HF_Model
    HF_Model --> HF_Repo
    
    HF_API -.-> S3
    HF_API -.-> MLflow
```

### 5.2 HuggingFace Configuration

#### `huggingface/app.py`

```python
"""
Enterprise Vision AI - HuggingFace Space Demo
"""

import gradio as gr
from huggingface_hub import hf_hub_download
import cv2
import numpy as np
from PIL import Image

# Model configuration
MODEL_REPO = "basai/defect-detection"
ORE_MODEL_REPO = "basai/ore-classification"

def load_model(repo_id: str):
    """Load model from HuggingFace Hub"""
    model_path = hf_hub_download(repo_id, "weights/best.pt")
    from ultralytics import YOLO
    return YOLO(model_path)

# Global model instances (loaded once)
def get_defect_model():
    if not hasattr(get_defect_model, "_model"):
        get_defect_model._model = load_model(MODEL_REPO)
    return get_defect_model._model

def get_ore_model():
    if not hasattr(get_ore_model, "_model"):
        get_ore_model._model = load_model(ORE_MODEL_REPO)
    return get_ore_model._model

def process_defect(image, conf_threshold=0.25):
    """Process defect detection"""
    model = get_defect_model()
    results = model.predict(image, conf=conf_threshold)
    
    # Visualize results
    annotated = results[0].plot()
    return annotated

def process_ore(image, conf_threshold=0.25):
    """Process ore classification"""
    model = get_ore_model()
    results = model.predict(image, conf=conf_threshold)
    
    # Visualize results
    annotated = results[0].plot()
    return annotated

# Gradio Interface
with gr.Blocks(title="Enterprise Vision AI Demo") as demo:
    gr.Markdown("# 🏭 Enterprise Vision AI")
    gr.Markdown("Industrial Computer Vision - Defect Detection & Ore Classification")
    
    with gr.Tab("Defect Detection"):
        with gr.Row():
            with gr.Column():
                defect_input = gr.Image(type="numpy", label="Input Image")
                defect_conf = gr.Slider(0, 1, 0.25, label="Confidence Threshold")
                defect_btn = gr.Button("Detect Defects", variant="primary")
            with gr.Column():
                defect_output = gr.Image(type="numpy", label="Result")
        
        defect_btn.click(
            fn=process_defect,
            inputs=[defect_input, defect_conf],
            outputs=defect_output
        )
    
    with gr.Tab("Ore Classification"):
        with gr.Row():
            with gr.Column():
                ore_input = gr.Image(type="numpy", label="Input Image")
                ore_conf = gr.Slider(0, 1, 0.25, label="Confidence Threshold")
                ore_btn = gr.Button("Classify Ore", variant="primary")
            with gr.Column():
                ore_output = gr.Image(type="numpy", label="Result")
        
        ore_btn.click(
            fn=process_ore,
            inputs=[ore_input, ore_conf],
            outputs=ore_output
        )

demo.launch()
```

#### `huggingface/requirements.txt`

```
gradio>=4.44.0
ultralytics>=8.3.0
opencv-python-headless>=4.10.0
numpy>=1.24.0
pillow>=10.0.0
huggingface-hub>=0.24.0
torch>=2.5.0
torchvision>=0.20.0
```

#### `huggingface/README.md`

```markdown
---
title: Enterprise Vision AI Demo
emoji: 🏭
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
tags:
- computer-vision
- industrial
- defect-detection
- ore-classification
- yolo
---

# Enterprise Vision AI

Industrial Computer Vision Demo for defect detection and ore classification.

## Features

- **Defect Detection**: Detect surface defects (cracks, scratches, holes)
- **Ore Classification**: Classify minerals (magnetite, chrome, waste)

## Usage

1. Select a tab (Defect Detection or Ore Classification)
2. Upload an image
3. Adjust confidence threshold
4. Click the process button
```

#### `huggingface/hardware.yaml`

```yaml
# Hardware configuration for HF Spaces
# https://huggingface.co/docs/hub/spaces-sdks-docker

hardware:
  - tier: free
    spec: CPU
  - tier: plus
    spec: CPU
  - tier: pro
    spec: GPU-T4
  - tier: dedicated
    spec: GPU-A10G
```

### 5.3 Model Publishing Pipeline

```mermaid
sequenceDiagram
    participant Dev as Developer
    participant GH as GitHub
    participant CI as GitHub Actions
    participant HF as HuggingFace Hub
    participant Space as HF Space
    
    Dev->>GH: Push code/model
    CI->>CI: Trigger on model update
    CI->>CI: Run tests & validation
    CI->>HF: Push model to Hub
    HF->>Space: Auto-deploy updated model
    Space->>Dev: Demo available
```

---

## 6. CI/CD Pipeline Design

### 6.1 GitHub Actions Workflows

#### Main CI Pipeline (`.github/workflows/ci.yml`)

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: '3.11'
  DOCKER_REGISTRY: ghcr.io

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Run linters
        run: |
          ruff check src/ tests/
          mypy src/bas_ai/ --ignore-missing-imports
      
      - name: Run tests
        run: |
          pytest tests/ -v --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml

  docker:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Login to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/Dockerfile
          push: true
          tags: |
            ghcr.io/${{ github.repository }}:latest
            ghcr.io/${{ github.repository }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  build-docs:
    needs: test
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Build documentation
        uses: actions/upload-artifact@v4
        with:
          name: docs
          path: docs/_build/
```

#### Release Pipeline (`.github/workflows/release.yml`)

```yaml
name: Release

on:
  release:
    types: [published]
  workflow_dispatch:
    inputs:
      version:
        description: 'Release version (e.g., v2.0.0)'
        required: true

env:
  DOCKER_REGISTRY: ghcr.io

jobs:
  release:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install release tools
        pip install build twine wheel
      
      - name: Build package
        python -m build
      
      - name: Publish to PyPI
        if: github.event_name == 'release'
        env:
          PYPI_TOKEN: ${{ secrets.PYPI_TOKEN }}
        run: |
          twine upload dist/*
      
      - name: Create GitHub Release
        if: github.event_name == 'release'
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: ${{ github.ref }}
          release_name: Release ${{ github.ref }}
          draft: false
          prerelease: false

  docker-release:
    needs: release
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Pull latest image
        run: docker pull ghcr.io/${{ github.repository }}:latest
      
      - name: Tag for release
        run: |
          docker tag ghcr.io/${{ github.repository }}:latest \
            ghcr.io/${{ github.repository }}:${{ github.ref_name }}
      
      - name: Push release tag
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Push to registry
        run: |
          docker push ghcr.io/${{ github.repository }}:${{ github.ref_name }}
```

### 6.2 Pipeline Stages

```mermaid
flowchart LR
    subgraph Stage1[Stage 1: Code Quality]
        Lint[Linting]
        Format[Code Formatting]
        Type[Type Checking]
    end
    
    subgraph Stage2[Stage 2: Testing]
        Unit[Unit Tests]
        Int[Integration Tests]
        E2E[End-to-End Tests]
    end
    
    subgraph Stage3[Stage 3: Build]
        Build[Build Package]
        Docker[Docker Build]
        Docs[Build Docs]
    end
    
    subgraph Stage4[Stage 4: Deploy]
        Publish[PyPI Release]
        Container[Container Registry]
        HF[HuggingFace Space]
    end
    
    Stage1 --> Stage2 --> Stage3 --> Stage4
```

### 6.3 Deployment Environments

| Environment | Trigger | Description |
|-------------|---------|-------------|
| **Development** | Push to `develop` | Auto-deploy to dev environment |
| **Staging** | Push to `main` | Pre-production testing |
| **Production** | Release tag | Production deployment |
| **Demo** | Model update | HuggingFace Space |

---

## 7. Model & Dataset Management

### 7.1 Model Registry Architecture

```mermaid
graph TB
    subgraph "Model Lifecycle"
        Train[Training] --> Register[Register]
        Register --> Validate[Validate]
        Validate --> Stage[Stage]
        Stage --> Deploy[Deploy]
        Deploy --> Monitor[Monitor]
        Monitor --> Archive[Archive]
    end
    
    subgraph "Storage"
        Local[Local Disk]
        S3[S3/MinIO]
        HF[🤗 Hub]
    end
    
    subgraph "Metadata"
        MLflow[MLflow Server]
        DB[(PostgreSQL)]
    end
    
    Register --> Local
    Register --> S3
    Register --> HF
    Register --> MLflow
    Register --> DB
```

### 7.2 MLflow Integration

#### Configuration (`configs/mlflow.yaml`)

```yaml
mlflow:
  tracking_uri: http://localhost:5000
  registry_uri: mysql://mlflow:mlflow@localhost:3306/mlflow
  artifact_root: s3://bas-ai-mlflow/artifacts
  
  experiment:
    name: "bas-industrial-ai"
    tracking_enabled: true
  
  model:
    registered_model_name: "bas-defect-detector"
    stages:
      - None
      - Staging
      - Production
      - Archived
```

#### Model Registration Script

```python
"""
Model registration with MLflow
"""

import mlflow
from mlflow.tracking import MlflowClient
import yaml

def load_config():
    with open('configs/mlflow.yaml') as f:
        return yaml.safe_load(f)

def register_model(
    model_path: str,
    model_name: str,
    metrics: dict,
    params: dict,
    tags: dict,
    stage: str = "Staging"
):
    """Register model to MLflow"""
    config = load_config()
    
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    client = MlflowClient()
    
    # Create or get experiment
    experiment = mlflow.get_or_create_experiment(
        config['mlflow']['experiment']['name']
    )
    
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        # Log parameters
        mlflow.log_params(params)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log tags
        mlflow.set_tags(tags)
        
        # Log model
        model_info = mlflow.pytorch.log_model(
            artifact_path="model",
            pytorch_model=model_path,
            registered_model_name=model_name
        )
        
        # Transition to stage
        if stage != "None":
            client.transition_model_version_stage(
                name=model_name,
                version=model_info.version,
                stage=stage
            )
    
    return model_info

# Example usage
if __name__ == "__main__":
    register_model(
        model_path="weights/best.pt",
        model_name="bas-defect-detector",
        metrics={
            "precision": 0.94,
            "recall": 0.92,
            "mAP50": 0.91,
            "mAP50-95": 0.85
        },
        params={
            "model": "yolo11s-seg",
            "image_size": 640,
            "batch_size": 16,
            "epochs": 100
        },
        tags={
            "dataset": "industrial-defects-v1",
            "framework": "ultralytics",
            "version": "2.0.0"
        }
    )
```

### 7.3 Dataset Structure

#### Standard Dataset Format

```
dataset/
├── dataset.yaml              # Dataset configuration
├── README.md                 # Dataset documentation
├── LICENSE                   # Data license
├── images/
│   ├── train/
│   │   ├── defect_001.jpg
│   │   ├── defect_002.jpg
│   │   └── ...
│   ├── val/
│   │   ├── defect_001.jpg
│   │   └── ...
│   └── test/
│       ├── defect_001.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── defect_001.txt
│   │   ├── defect_002.txt
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
├── classes.txt               # Class definitions
└── metadata/
    ├── train.csv             # Metadata with file info
    ├── val.csv
    └── test.csv
```

#### Dataset Configuration (`dataset.yaml`)

```yaml
# YOLO Dataset Configuration
# Enterprise Vision AI Dataset

path: ./dataset
train: images/train
val: images/val
test: images/test

# Classes
names:
  0: crack           # Çatlak
  1: scratch         # Çizik
  2: hole            # Delik
  3: stain           # Leke
  4: deformation     # Deformasyon
  5: magnetite       # Manyetit
  6: chrome          # Krom
  7: waste           # Atık
  8: low_grade       # Düşük tenör

# Dataset info
dataset_info:
  version: "2.0.0"
  created: "2026-03-08"
  total_images: 10000
  train_split: 0.7
  val_split: 0.2
  test_split: 0.1
  annotations: "COCO format"
  source: "Industrial manufacturing lines"

# Data augmentation
augmentation:
  enabled: true
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 0.0
  translate: 0.1
  scale: 0.5
  shear: 0.0
  perspective: 0.0
  flipud: 0.0
  fliplr: 0.5
  mosaic: 1.0
  mixup: 0.0
```

#### Label Format (YOLO Segmentation)

```
# Format: class_id cx cy w h polygon_points...
# Example for defect detection

0 0.5 0.5 0.2 0.3 0.4,0.3 0.6,0.3 0.6,0.7 0.4,0.7
1 0.3 0.4 0.15 0.2 0.25,0.3 0.35,0.3 0.35,0.5 0.25,0.5
```

### 7.4 Dataset Management Service

```python
"""
Dataset management service
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import yaml
import shutil

@dataclass
class DatasetConfig:
    name: str
    path: Path
    train_split: float
    val_split: float
    test_split: float
    classes: List[str]

class DatasetManager:
    """Manage datasets for training and evaluation"""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.datasets_path = base_path / "data" / "datasets"
    
    def create_dataset(
        self,
        name: str,
        source_path: Path,
        classes: List[str],
        split_ratios: tuple = (0.7, 0.2, 0.1)
    ) -> DatasetConfig:
        """Create a new dataset from source images"""
        
        # Create dataset directory structure
        dataset_path = self.datasets_path / name
        dataset_path.mkdir(parents=True)
        
        for split in ['train', 'val', 'test']:
            (dataset_path / 'images' / split).mkdir(parents=True)
            (dataset_path / 'labels' / split).mkdir(parents=True)
        
        # Create dataset.yaml
        config = DatasetConfig(
            name=name,
            path=dataset_path,
            train_split=split_ratios[0],
            val_split=split_ratios[1],
            test_split=split_ratios[2],
            classes=classes
        )
        
        self._save_config(dataset_path, config)
        
        return config
    
    def validate_dataset(self, dataset_path: Path) -> dict:
        """Validate dataset integrity"""
        issues = []
        
        # Check directory structure
        required_dirs = [
            'images/train', 'images/val', 'images/test',
            'labels/train', 'labels.val', 'labels/test'
        ]
        
        for dir_path in required_dirs:
            if not (dataset_path / dir_path).exists():
                issues.append(f"Missing directory: {dir_path}")
        
        # Check image/label pairs
        for split in ['train', 'val', 'test']:
            images = set(p.stem for p in (dataset_path / 'images' / split).glob('*.jpg'))
            labels = set(p.stem for p in (dataset_path / 'labels' / split).glob('*.txt'))
            
            missing_labels = images - labels
            missing_images = labels - images
            
            if missing_labels:
                issues.append(f"{split}: {len(missing_labels)} images without labels")
            if missing_images:
                issues.append(f"{split}: {len(missing_images)} labels without images")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
    
    def export_to_yolo(self, dataset_path: Path, output_path: Path):
        """Export dataset to YOLO format for training"""
        # Create zip file or copy to training location
        pass
```

---

## 8. API Reference

### 8.1 REST API Endpoints

| Method | Endpoint | Description | Authentication |
|--------|----------|-------------|----------------|
| `GET` | `/health` | Health check | None |
| `POST` | `/api/v1/detect` | Run defect detection | Optional |
| `POST` | `/api/v1/classify` | Classify ore | Optional |
| `POST` | `/api/v1/detect/batch` | Batch detection | Required |
| `GET` | `/api/v1/models` | List models | Optional |
| `GET` | `/api/v1/models/{name}` | Get model info | Optional |
| `POST` | `/api/v1/models/{name}/predict` | Model inference | Required |
| `GET` | `/api/v1/datasets` | List datasets | Optional |
| `GET` | `/api/v1/history` | Get inference history | Required |

### 8.2 API Schemas

#### Inference Request

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class InferenceRequest(BaseModel):
    """Base inference request"""
    model_name: str = Field(..., description="Model to use")
    confidence: float = Field(0.25, ge=0.0, le=1.0)
    image_url: Optional[str] = None
    image_data: Optional[str] = None  # Base64 encoded

class BatchInferenceRequest(BaseModel):
    """Batch inference request"""
    model_name: str
    confidence: float = 0.25
    image_urls: List[str]
```

#### Inference Response

```python
class DetectionResult(BaseModel):
    """Single detection result"""
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    mask: Optional[List[List[int]]] = None
    area: Optional[float] = None

class InferenceResponse(BaseModel):
    """Inference response"""
    request_id: str
    model_name: str
    inference_time: float
    results: List[DetectionResult]
    metadata: dict
```

---

## 9. Security & Compliance

### 9.1 Security Measures

| Area | Implementation |
|------|----------------|
| **Authentication** | OAuth2 with JWT tokens |
| **Authorization** | Role-based access control (RBAC) |
| **API Security** | Rate limiting, input validation |
| **Data Encryption** | TLS 1.3, AES-256 at rest |
| **Secrets Management** | HashiCorp Vault or AWS Secrets Manager |
| **Audit Logging** | Immutable audit trail |

### 9.2 Data Privacy

- All inference data can be processed locally
- No mandatory cloud data transfer
- GDPR-compliant data handling options
- Anonymization capabilities for sensitive data

---

## 10. Contributing & Community

### 10.1 Contribution Guidelines

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for detailed guidelines on:
- Code style and conventions
- Pull request process
- Testing requirements
- Documentation standards

### 10.2 Code of Conduct

See [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) for community guidelines.

### 10.3 License

This project is licensed under the **Apache License 2.0** - see [`LICENSE`](LICENSE) for details.

---

## 11. Quick Start

### 11.1 Local Development

```bash
# Clone repository
git clone https://github.com/bas-ai/bas-endustriyel-ai.git
cd bas-endustriyel-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run Streamlit app
streamlit run app/main.py
```

### 11.2 Docker Deployment

```bash
# Build Docker image
docker build -t bas-ai:latest -f docker/Dockerfile .

# Run container
docker run -p 8501:8501 bas-ai:latest
```

### 11.3 HuggingFace Demo

Visit our HuggingFace Space to try the demo:
- **URL**: https://huggingface.co/spaces/basai/industrial-ai
- **Features**: Defect detection, Ore classification

---

## 12. Appendix

### A. Environment Variables

```bash
# .env.example

# Application
APP_ENV=development
APP_DEBUG=true
APP_HOST=0.0.0.0
APP_PORT=8501

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/bas_ai

# Redis
REDIS_URL=redis://localhost:6379/0

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000

# HuggingFace
HF_TOKEN=your_huggingface_token

# Model Storage
S3_ENDPOINT=http://localhost:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_BUCKET=bas-ai-models
```

### B. Model Performance Benchmarks

| Model | mAP50 | mAP50-95 | FPS | Parameters |
|-------|-------|-----------|-----|------------|
| YOLO11n-seg | 0.85 | 0.72 | 120 | 2.6M |
| YOLO11s-seg | 0.91 | 0.78 | 85 | 9.4M |
| YOLO11m-seg | 0.94 | 0.82 | 45 | 25.9M |
| YOLO11l-seg | 0.95 | 0.85 | 30 | 43.7M |

### C. Support & Resources

- **Documentation**: https://docs.bas-ai.io
- **Discord**: https://discord.gg/bas-ai
- **Twitter**: @BASIndustrialAI
- **Email**: contact@bas-ai.io

---

*This specification is a living document and will be updated as the project evolves.*
