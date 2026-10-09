# Deepfake Detection Application (MesoNet + XceptionNet + Vision Transformer + Random Forest)

An end-to-end **Deepfake Detection Application** integrated into a **Flask-based social media platform (`VeriShare`)** that automatically inspects uploaded photos and videos and prevents detected deepfake content from being shared while allowing genuine media to be published.

---

## Key Features & Architecture

1. **Preprocessing (`models/feature_extractors.py` -> `VideoImagePreprocessor`)**:
   - Converts uploaded videos (`.mp4`, `.avi`, `.mov`, `.webm`) into frames using **OpenCV (`cv2.VideoCapture`)**.
   - Performs facial region localization (`cv2.CascadeClassifier`), resizes frames (`256x256` for MesoNet, `224x224` for XceptionNet and ViT), and normalizes pixel values to `[0.0, 1.0]`.
2. **Multi-Layer Feature Extraction**:
   - **MesoNet (`MesoNetExtractor` / `build_mesonet4_keras`)**: Extracts `64-D` mesoscopic facial manipulation and high-frequency Laplacian boundary features.
   - **XceptionNet (`XceptionFeatureExtractor` / `build_xception_extractor_keras`)**: Extracts `128-D` depthwise separable convolutional texture and cross-channel correlation features.
   - **Vision Transformer (`VisionTransformerExtractor` / `build_vit_extractor_keras`)**: Extracts `128-D` `16x16` patch embeddings and multi-head self-attention global spatial consistency features.
3. **Feature Fusion**:
   - Concatenates extracted features from MesoNet (`64-D`), XceptionNet (`128-D`), and Vision Transformer (`128-D`) into a unified **`320-D` feature vector**.
4. **Random Forest Classification (`train_classifier.py`)**:
   - Trains a **Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`)** on the fused `320-D` feature vectors for binary classification (`Real` vs. `Fake`).
5. **Flask Social Media Web Application (`app.py`)**:
   - **User Authentication**: User registration, login, logout, and session management.
   - **Content Posting**: Upload photos and videos to the community feed, like and comment on posts.
   - **Deepfake Detection Integration**: Automatically scans every uploaded image or video before publishing—blocking detected deepfakes with a detailed forensic telemetry report while verifying and publishing genuine content.

---

## Project Structure

```text
DeepFakeDetection/
├── app.py                      # Flask Social Media Web Application + Deepfake Gatekeeper
├── train_classifier.py         # Training & evaluation script for the Random Forest Classifier
├── requirements.txt            # Python dependencies
├── models/
│   ├── __init__.py
│   ├── feature_extractors.py   # Preprocessing, MesoNet, XceptionNet, ViT, Feature Fusion & RF pipeline
│   └── saved/                  # Saved Random Forest model & StandardScaler artifacts
├── templates/
│   ├── base.html               # Main layout & navigation
│   ├── index.html              # Social feed, post upload & live gatekeeper logs
│   ├── scanner.html            # Standalone AI Forensic Analyzer page
│   ├── login.html              # User login page
│   └── register.html           # User registration page
└── static/
    └── uploads/                # Verified user media uploads
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train / Initialize the Random Forest Classifier
```bash
python train_classifier.py
```

### 3. Run the Flask Web Application
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser.
- **Demo Account**: Username `praveen` / Password `praveen123`

---

## Author
**Pidamarthi Praveen Kumar**
- B.Tech in Artificial Intelligence and Machine Learning — Chaitanya Bharathi Institute of Technology (CBIT)
- GitHub: [@praveen-79](https://github.com/praveen-79)
