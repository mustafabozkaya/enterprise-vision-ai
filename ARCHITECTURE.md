# Enterprise Vision AI - System Architecture

## Table of Contents

1. [System Overview](#system-overview)
2. [Layer Architecture](#layer-architecture)
3. [Component Architecture](#component-architecture)
4. [Deployment Architecture](#deployment-architecture)
5. [Data Flow Diagrams](#data-flow-diagrams)
6. [Technology Stack](#technology-stack)
7. [Security Considerations](#security-considerations)
8. [Scalability Patterns](#scalability-patterns)

---

## System Overview

Enterprise Vision AI is an industrial computer vision system designed for mineral defect detection and ore classification in mining operations. The system leverages YOLO11 (Ultralytics v8+) for instance segmentation, providing pixel-level masks for precise defect localization.

### High-Level Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        Streamlit[Streamlit Web UI]
        HFGradio[HuggingFace Gradio Space]
    end

    subgraph "API Gateway Layer"
        Auth[Authentication Service]
        Router[Request Router]
        RateLimiter[Rate Limiter]
    end

    subgraph "Service Layer"
        YOLO[YOLO Inference Engine]
        ModelMgr[Model Manager]
        Preproc[Image Preprocessor]
        Postproc[Result Post-processor]
    end

    subgraph "Data Layer"
        Models[Model Storage]
        Datasets[Dataset Storage]
        Cache[Result Cache]
    end

    Streamlit --> Auth
    HFGradio --> Auth
    Auth --> Router
    Router --> RateLimiter
    RateLimiter --> YOLO
    YOLO --> ModelMgr
    YOLO --> Preproc
    Preproc --> Models
    Postproc --> Cache
```

### Core Capabilities

| Capability | Description |
|------------|-------------|
| **Defect Detection** | Real-time surface defect detection (cracks, scratches, dents, discoloration, contamination) |
| **Ore Classification** | Mineral ore classification (magnetite, chromite, waste, low-grade) |
| **Instance Segmentation** | Pixel-level mask generation for defect localization |
| **Severity Assessment** | Automatic defect severity classification (low, medium, high) |
| **Maintenance Recommendations** | AI-driven maintenance suggestions based on detected defects |

---

## Layer Architecture

### 1. Client Layer

The Client Layer provides user interfaces for interacting with the vision AI system.

#### Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Streamlit Web UI** | Streamlit (Python) | Primary production application with multi-page navigation |
| **HuggingFace Gradio** | Gradio (Python) | Cloud-based demo space for public access |
| **Mobile Interface** | Responsive Design | Mobile-friendly access via PWA |

#### Streamlit UI Structure

```
app.py                 # Main entry point with page routing
├── pages/
│   ├── 01_Defekt_Tespiti.py    # Defect Detection page
│   ├── 02_Cevher_On_Secimi.py   # Ore Classification page
│   ├── defect.py               # Defect detection implementation
│   └── ore.py                  # Ore classification implementation
```

#### HuggingFace Space Structure

```
huggingface_space/
├── app.py                 # Gradio application
├── requirements.txt       # Python dependencies
├── hardware.yaml         # Hardware configuration
└── README.md             # Space documentation
```

### 2. API Gateway Layer

The API Gateway Layer handles request routing, authentication, and rate limiting.

```mermaid
graph LR
    Client[Client Request]
    Auth[Auth Middleware]
    Router[Request Router]
    Rate[Rate Limiter]
    Cache[Response Cache]
    Backend[Backend Service]

    Client --> Auth
    Auth --> Router
    Router --> Rate
    Rate --> Cache
    Cache --> Backend
```

#### Responsibilities

| Responsibility | Implementation |
|----------------|----------------|
| **Authentication** | Session-based auth for Streamlit, API keys for HF Spaces |
| **Request Routing** | Path-based routing to appropriate service endpoints |
| **Rate Limiting** | Per-user rate limits to prevent abuse |
| **Response Caching** | LRU cache for frequently requested results |
| **Request Validation** | Input validation and sanitization |

### 3. Service Layer

The Service Layer contains the core business logic for AI inference.

```mermaid
graph TB
    subgraph "Service Layer"
        Preproc[Image Preprocessor]
        YOLO[YOLO Inference Engine]
        Postproc[Result Post-processor]
        ModelMgr[Model Manager]
    end

    subgraph "Processing Flow"
        Input[Image Input]
        Resize[Resize & Normalize]
        Infer[Model Inference]
        Annotate[Draw Annotations]
        Output[Structured Output]
    end

    Input --> Resize
    Resize --> Preproc
    Preproc --> Infer
    Infer --> YOLO
    YOLO --> Postproc
    Postproc --> Annotate
    Annotate --> Output
```

#### Core Services

| Service | File | Responsibility |
|---------|------|----------------|
| **YOLO Inference** | [`pages/defect.py`](pages/defect.py:33), [`pages/ore.py`](pages/ore.py:32) | Load and run YOLO models |
| **Model Management** | [`models/registry.yaml`](models/registry.yaml:1) | Model versioning and metadata |
| **Image Preprocessing** | [`utils.py`](utils.py:72) | Image resize, normalization, augmentation |
| **Result Post-processing** | [`utils.py`](utils.py:50) | Result formatting, annotation drawing |

### 4. Data Layer

The Data Layer manages models, datasets, and caching.

```mermaid
graph LR
    subgraph "Data Layer"
        ModelStore[(Model Storage)]
        DatasetStore[(Dataset Storage)]
        ResultCache[(Result Cache)]
        MetadataDB[(Metadata DB)]
    end

    subgraph "Access Patterns"
        Download[Model Download]
        CacheHit[Cache Hit]
        CacheMiss[Cache Miss]
    end

    Download --> ModelStore
    CacheHit --> ResultCache
    CacheMiss --> DatasetStore
```

#### Storage Structure

```
models/
├── checkpoints/           # Trained model weights
├── .cache/              # Model cache directory
├── registry.yaml         # Model registry configuration
└── download_models.py   # Model download script

datasets/
├── defect_detection/    # YOLO format dataset
│   ├── train/          # Training images & labels
│   ├── val/            # Validation images & labels
│   ├── test/           # Test images & labels
│   └── dataset.yaml   # Dataset configuration
└── README.md           # Dataset documentation
```

---

## Component Architecture

### 1. Model Service

The Model Service manages YOLO model lifecycle.

```mermaid
classDiagram
    class ModelService {
        +load_model(model_id) YOLOModel
        +unload_model(model_id) None
        +get_model_info(model_id) dict
        +list_available_models() list
    }

    class YOLOModel {
        +predict(image, conf, iou) Results
        +export(format) bytes
        +get_names() dict
    }

    class ModelRegistry {
        +models: dict
        +get_metadata(model_id) dict
        +register_model(model) None
    }

    ModelService --> ModelRegistry
    ModelService --> YOLOModel
```

#### Model Configuration ([`models/registry.yaml`](models/registry.yaml:1))

```yaml
models:
  yolov8n-defect:
    type: "detection"
    framework: "ultralytics"
    input_size: 640
    classes: ["crack", "scratch", "dent", "discoloration", "contamination"]
  yolov8s-defect:
    type: "segmentation"
    framework: "ultralytics"
    input_size: 640
    classes: ["çatlak", "çizik", "delik", "leke", "deformasyon"]
```

#### Model Loading Pattern

```python
# From pages/defect.py:33
@st.cache_resource
def load_model():
    """Load YOLO model with fallback chain"""
    from ultralytics import YOLO
    
    try:
        model = YOLO('yolo26-seg.pt')
    except:
        try:
            model = YOLO('yolo11s-seg.pt')
        except:
            model = YOLO('yolo11n-seg.pt')
    
    return model
```

### 2. Dataset Service

The Dataset Service manages training and inference datasets.

```mermaid
graph TB
    subgraph "Dataset Service"
        Loader[Dataset Loader]
        Validator[Dataset Validator]
        Augmenter[Data Augmenter]
        Splitter[Train/Val/Test Splitter]
    end

    subgraph "YOLO Dataset Format"
        Images[Images Directory]
        Labels[Labels Directory]
        YAML[Dataset YAML]
    end

    Loader --> Validator
    Validator --> Augmenter
    Augmenter --> Splitter
    Splitter --> Images
    Splitter --> Labels
```

#### Dataset Configuration ([`datasets/defect_detection/dataset.yaml`](datasets/defect_detection/dataset.yaml:1))

```yaml
name: "BAS Defect Detection Dataset"
nc: 5  # Number of classes
names:
  0: crack
  1: scratch
  2: dent
  3: discoloration
  4: contamination
```

### 3. Inference Pipeline

The Inference Pipeline processes images through the complete detection workflow.

```mermaid
flowchart LR
    subgraph Input
        Image[Image Input]
    end

    subgraph Preprocessing
        Decode[Decode]
        Resize[Resize to 640x640]
        Normalize[Normalize pixel values]
    end

    subgraph Inference
        YOLO[YOLO11 Model]
        NMS[Non-Max Suppression]
    end

    subgraph Postprocessing
        Filter[Filter by confidence]
        Mask[Extract segmentation masks]
        Annotate[Draw bounding boxes]
    end

    subgraph Output
        JSON[JSON Results]
        Visual[Visual Output]
    end

    Image --> Decode
    Decode --> Resize
    Resize --> Normalize
    Normalize --> YOLO
    YOLO --> NMS
    NMS --> Filter
    Filter --> Mask
    Mask --> Annotate
    Annotate --> JSON
    Annotate --> Visual
```

#### Inference Pipeline Implementation

```python
# From utils.py:72
def draw_annotations(image, results, class_colors, show_labels, show_confidence):
    """Draw detection annotations on image"""
    # Extract detections
    for result in results:
        boxes = result.boxes
        masks = result.masks
        
        # Draw bounding boxes
        for box in boxes:
            # Draw box with class color
            # Add label with confidence
            pass
        
        # Draw segmentation masks
        if masks is not None:
            # Overlay colored masks
            pass
    
    return annotated_image
```

---

## Deployment Architecture

### 1. Docker Containerization

The project uses multi-stage Docker builds for optimized production images.

```mermaid
graph TB
    subgraph "Docker Build Stages"
        Builder[Builder Stage<br/>python:3.11-slim-bookworm]
        Production[Production Stage<br/>python:3.11-slim-bookworm]
    end

    subgraph "Builder Stage Actions"
        InstallDeps[Install Dependencies]
        CreateVenv[Create Virtual Environment]
    end

    subgraph "Production Stage Actions"
        CopyVenv[Copy Virtual Environment]
        CopyApp[Copy Application Code]
        SetupUser[Setup Non-root User]
    end

    Builder --> InstallDeps
    InstallDeps --> CreateVenv
    CreateVenv --> CopyVenv
    CopyVenv --> CopyApp
    CopyApp --> SetupUser
```

#### Dockerfile Stages ([`Dockerfile`](Dockerfile:1))

| Stage | Base Image | Purpose |
|-------|------------|---------|
| **Builder** | `python:3.11-slim-bookworm` | Install dependencies in isolated venv |
| **Production** | `python:3.11-slim-bookworm` | Minimal runtime image |
| **Development** | `python:3.11-slim-bookworm` | Development with live reload |

#### Docker Compose Configuration

```yaml
# From docker-compose.yml
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
      target: development
    ports:
      - "8501:8501"
    volumes:
      - .:/app
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - STREAMLIT_SERVER_PORT=8501
    restart: unless-stopped
```

### 2. HuggingFace Space Deployment

The HuggingFace Space provides a cloud-based demo environment.

```mermaid
graph TB
    subgraph "HuggingFace Infrastructure"
        HFProxy[HF Proxy / CDN]
        GPU[GPU Hardware]
        ModelHub[Model Hub]
    end

    subgraph "Gradio Application"
        Upload[File Upload]
        Preproc[Preprocessing]
        Infer[Inference]
        Display[Result Display]
    end

    HFProxy --> Upload
    Upload --> Preproc
    Preproc --> Infer
    Infer --> GPU
    GPU --> ModelHub
    ModelHub --> Display
```

#### Hardware Configuration ([`huggingface_space/hardware.yaml`](huggingface_space/hardware.yaml:1))

```yaml
# Hardware specifications for HF Space
hardware:
  cpu: 2
  memory: 16GB
  gpu: T4  # Optional GPU for faster inference
  enable_accelerate: true
```

#### Gradio Application Structure ([`huggingface_space/app.py`](huggingface_space/app.py:1))

```python
# Key components in Gradio app
DEFECT_MODEL_ID = "bas-industriel/yolo-defect-detection"
ORE_MODEL_ID = "bas-industriel/yolo-ore-classification"

DEFECT_CLASSES = ["çizik", "çatlak", "delik", "ezilme", "yanık", "pas", "diğer"]
ORE_CLASSES = ["manyetit", "kromit", "pirit", "kalkopirit", "atık", "düşük tenörlü"]
```

### 3. Streamlit Cloud Deployment

The Streamlit application can be deployed to Streamlit Cloud.

```mermaid
graph LR
    GitHub[GitHub Repository]
    StreamlitCloud[Streamlit Cloud]
    ModelStorage[Model Storage]

    GitHub --> StreamlitCloud
    ModelStorage -.-> StreamlitCloud
```

#### Deployment Configuration

| Environment | Platform | Configuration |
|-------------|----------|---------------|
| **Local** | Docker Compose | [`docker-compose.yml`](docker-compose.yml:1) |
| **Cloud** | Streamlit Cloud | GitHub integration |
| **Demo** | HuggingFace Space | Gradio interface |

---

## Data Flow Diagrams

### 1. Defect Detection Flow

```mermaid
sequenceDiagram
    participant User
    participant Streamlit
    participant Model
    participant Cache
    participant Utils

    User->>Streamlit: Upload image
    Streamlit->>Model: Load YOLO model
    Model-->>Streamlit: Model loaded
    
    Streamlit->>Model: Run inference (image, conf=0.25)
    Model->>Utils: Preprocess image (resize, normalize)
    Utils-->>Model: Preprocessed image
    
    alt Cache hit
        Cache-->>Streamlit: Return cached results
    else Cache miss
        Model->>Model: YOLO inference
        Model-->>Streamlit: Detection results
        Streamlit->>Cache: Store results
    end
    
    Streamlit->>Utils: Draw annotations
    Utils-->>Streamlit: Annotated image
    
    Streamlit->>User: Display results
```

### 2. Ore Classification Flow

```mermaid
sequenceDiagram
    participant User
    participant Streamlit
    participant OreModel
    participant Cache
    participant Utils

    User->>Streamlit: Upload ore image
    Streamlit->>OreModel: Load classification model
    
    Streamlit->>OreModel: Run inference
    OreModel->>OreModel: Classify ore type
    
    alt High confidence
        OreModel-->>Streamlit: Classification results
    else Low confidence
        Streamlit->>User: Request re-upload
    end
    
    Streamlit->>Utils: Calculate ore metrics
    Utils-->>Streamlit: Metal ratio, recommendations
    
    Streamlit->>User: Display classification + recommendations
```

### 3. Batch Processing Flow

```mermaid
flowchart TB
    subgraph Input
        Batch[Image Batch]
    end

    subgraph Processing
        Queue[Processing Queue]
        Worker1[Worker 1]
        Worker2[Worker 2]
        WorkerN[Worker N]
    end

    subgraph Output
        Results[Results Aggregation]
        Export[Export to CSV/JSON]
    end

    Batch --> Queue
    Queue --> Worker1
    Queue --> Worker2
    Queue --> WorkerN
    
    Worker1 --> Results
    Worker2 --> Results
    WorkerN --> Results
    
    Results --> Export
```

---

## Technology Stack

### Core Technologies

| Category | Technology | Version | Purpose |
|----------|------------|---------|---------|
| **AI Framework** | Ultralytics | v8+ | YOLO11 model inference |
| **Deep Learning** | PyTorch | 2.x | Neural network backend |
| **Computer Vision** | OpenCV | 4.x | Image processing |
| **Web Framework** | Streamlit | 1.x | Primary UI framework |
| **Demo Framework** | Gradio | 4.x | HF Space interface |

### Infrastructure

| Category | Technology | Purpose |
|----------|------------|---------|
| **Containerization** | Docker | Application packaging |
| **Orchestration** | Docker Compose | Local development |
| **Cloud Platform** | Streamlit Cloud | Production deployment |
| **Model Hub** | HuggingFace Hub | Model storage & distribution |

### Supporting Libraries

| Library | Purpose |
|---------|---------|
| NumPy | Numerical operations |
| Pandas | Data manipulation |
| Pillow (PIL) | Image handling |
| Numba | Performance optimization |

### Development Tools

| Tool | Purpose |
|------|---------|
| Python 3.11 | Runtime environment |
| Make | Build automation |
| pytest | Testing framework |
| Git | Version control |

---

## Security Considerations

### Authentication

```mermaid
graph LR
    subgraph "Authentication Flow"
        Login[Login Form]
        Session[Session Management]
        Token[JWT Token]
        Validate[Token Validation]
    end

    Login --> Session
    Session --> Token
    Token --> Validate
```

### Security Measures

| Measure | Implementation |
|---------|----------------|
| **Input Validation** | File type checking, size limits |
| **Output Sanitization** | HTML escaping in UI |
| **Secure Defaults** | Non-root container user |
| **Secrets Management** | Environment variables |

---

## Scalability Patterns

### Horizontal Scaling

```mermaid
graph TB
    LB[Load Balancer]
    
    subgraph "Instance 1"
        App1[Streamlit App]
        Model1[YOLO Model]
    end
    
    subgraph "Instance 2"
        App2[Streamlit App]
        Model2[YOLO Model]
    end
    
    subgraph "Instance N"
        AppN[Streamlit App]
        ModelN[YOLO Model]
    end
    
    LB --> App1
    LB --> App2
    LB --> AppN
    
    App1 --> Model1
    App2 --> Model2
    AppN --> ModelN
```

### Caching Strategy

| Cache Layer | Technology | TTL | Purpose |
|-------------|------------|-----|---------|
| **Model Cache** | `@st.cache_resource` | Persistent | Model weights |
| **Result Cache** | LRU Cache | 1 hour | Inference results |
| **Static Assets** | CDN | Long-term | CSS, JS, images |

---

## Appendix: File Reference

| File | Description |
|------|-------------|
| [`app.py`](app.py:1) | Main Streamlit application entry point |
| [`utils.py`](utils.py:1) | Shared utility functions |
| [`pages/defect.py`](pages/defect.py:1) | Defect detection page logic |
| [`pages/ore.py`](pages/ore.py:1) | Ore classification page logic |
| [`huggingface_space/app.py`](huggingface_space/app.py:1) | Gradio application |
| [`models/registry.yaml`](models/registry.yaml:1) | Model registry configuration |
| [`datasets/defect_detection/dataset.yaml`](datasets/defect_detection/dataset.yaml:1) | Dataset configuration |
| [`Dockerfile`](Dockerfile:1) | Docker image definition |
| [`docker-compose.yml`](docker-compose.yml:1) | Local deployment configuration |

---

*Last Updated: 2026-03-08*
*Architecture Version: 1.0.0*
