# Traffic Sign Recognition System

An intelligent Traffic Sign Recognition (TSR) system that automatically detects and classifies traffic signs from images using computer vision and deep learning. Built as a graduation project for the Digital Egypt Pioneers Initiative (DEPI) - AI/ML Track.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Performance Metrics](#performance-metrics)
- [Deployment](#deployment)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

This project implements a comprehensive deep learning solution for real-time traffic sign recognition, enabling vehicles to automatically identify and classify 43 different traffic sign categories. The system processes input images through trained neural networks (CNN and YOLO) to predict traffic sign types with high accuracy, supporting critical safety functions in modern automotive applications.

**Key Applications:**
- 🚗 Autonomous and semi-autonomous driving systems
- 🛡️ Advanced Driver Assistance Systems (ADAS)
- 📊 Road safety monitoring and automation
- 🚦 Smart city traffic management

The system enhances driving safety and significantly reduces human error in traffic sign interpretation.
## ✨ Features

- **🔄 Advanced Data Preprocessing**: Comprehensive pipeline for image cleaning, normalization, and augmentation techniques
- **🧠 Dual Model Approach**: CNN for classification + YOLOv8 for real-time detection
- **🎯 43-Class Classification**: Complete coverage of GTSRB traffic sign categories including speed limits, warnings, and prohibitions
- **📊 Comprehensive Evaluation**: Performance analysis using accuracy, precision, recall, and F1-score metrics
- **⚡ Real-Time Processing**: Optimized for video streams with ≥20 FPS performance
- **🌐 Web API Deployment**: RESTful API built with Flask/FastAPI for easy integration
- **🖥️ Interactive GUI**: User-friendly interface for testing images and videos (Streamlit/Tkinter/Web-based)
- **📈 Performance Monitoring**: Detailed logging and metrics tracking throughout training and inference
- **🐳 Docker Support**: Containerized deployment for easy scaling

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.x |
| **Deep Learning** | TensorFlow, Keras |
| **Object Detection** | YOLOv8 |
| **Computer Vision** | OpenCV |
| **Data Processing** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **Backend API** | Flask  |
| **Frontend UI** | Streamlit / Tkinter / HTML/CSS/JS |
| **Containerization** | Docker |
| **Version Control** | Git, GitHub |
| **Development** | Jupyter Notebook, Visual Studio Code |
| **Hardware** | GPU-enabled (Google Colab / Local GPU) |

## 📁 Project Structure

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- GPU (recommended): CUDA-capable GPU for faster training
- Git

## 📊 Dataset

### German Traffic Sign Recognition Benchmark (GTSRB)

This project uses the GTSRB dataset, which contains:

- **43 traffic sign classes** covering:
  - Speed limits (20, 30, 50, 60, 70, 80, 100, 120 km/h)
  - Warning signs (dangerous curves, slippery road, etc.)
  - Mandatory signs (turn right, go straight, etc.)
  - Prohibitory signs (no entry, no overtaking, etc.)
  
- **50,000+ images** with varying:
  - Lighting conditions
  - Weather conditions
  - Viewing angles
  - Image quality



### Data Quality Metrics

| Metric | Value |
|--------|-------|
| Missing values handled | 100% |
| Data accuracy after preprocessing | ≥98% |
| Dataset diversity | 43 classes (100% represented) |

## 💻 Usage

### Training the CNN Model

```bash
python src/train.py --model cnn --epochs 50 --batch_size 32 --learning_rate 0.001
```

**Arguments:**
- `--model`: Model type (cnn, resnet, mobilenet)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 32)
- `--learning_rate`: Learning rate for optimizer (default: 0.001)
- `--data_path`: Path to dataset directory
- `--save_path`: Path to save trained model

### Training YOLO Model

```bash
python src/train.py --model yolo --epochs 100 --img_size 640
```

### Evaluating the Model

```bash
python src/evaluate.py --model_path models/cnn_model.h5 --test_data data/test/
```

### Making Predictions

**Single Image:**
```bash
python src/predict.py --image path/to/sign.jpg --model models/cnn_model.h5
```

**Video Stream:**
```bash
python src/predict.py --video path/to/video.mp4 --model models/yolo_model.pt
```

### Running the API

```bash
# Using Flask
python api/app.py

# Using FastAPI
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

### Using Jupyter Notebooks

```bash
jupyter notebook
# Navigate to notebooks/ directory
```

### YOLO Integration

- **YOLOv8** for real-time traffic sign detection
- **Input**: 640x640 RGB images
- **Output**: Bounding boxes + class predictions for 43 sign types
- **Performance**: ≥20 FPS on standard hardware

## 📈 Performance Metrics

### Target KPIs

#### Model Performance
| Metric | Target | Status |
|--------|--------|--------|
| F1-Score | ≥96% | 🎯 |
| Prediction Latency | ≤50 ms/image | 🎯 |
| Error Rate (FP/FN) | ≤4% | 🎯 |
| Model Accuracy | ≥96% | 🎯 |

#### Deployment & Scalability
| Metric | Target | Status |
|--------|--------|--------|
| API Uptime | ≥99% | 🎯 |
| Response Time | ≤200 ms/request | 🎯 |
| Real-time Processing | ≥20 FPS | 🎯 |

#### Business Impact
| Metric | Target |
|--------|--------|
| Reduction in manual detection effort | 80% |
| Expected cost savings | 50% |
| User satisfaction | ≥90% |

### Confusion Matrix & Results

Detailed performance analysis, confusion matrices, and visualization plots are available in the `results/` directory after training.

## 🚀 Deployment

### Local Deployment

```bash
# Run the API server
python app.py
```

### Docker Deployment

```bash
# Build the Docker image
docker build -t traffic-sign-recognition .

# Run the container
docker run -p 8000:8000 traffic-sign-recognition
```

### Cloud Deployment

The API is designed to be easily deployed on:
- AWS (EC2, Lambda, SageMaker)
- Google Cloud Platform (Compute Engine, Cloud Run)
- Azure (VM, Container Instances)
- Heroku

## 🗓️ Roadmap

### Project Milestones

| Week | Milestone | Status |
|------|-----------|--------|
| Week 1 | Dataset Understanding & Setup | ✅ |
| Week 2 | Data Preprocessing (cleaning, normalization, augmentation) | ✅ |
| Week 3 | Model Development (CNN architecture design and training) | ✅ |
| Week 4 | Model Evaluation (validation and test analysis) | 🔄 |
| Week 5 | Deployment & YOLO Integration (API + YOLO model) | 🔄 |
| Week 6 | GUI Implementation & Testing (end-to-end system testing) | 📋 |
| Week 7 | YOLO Integration Completion (merge CNN + YOLO) | 📋 |

**Legend:** ✅ Completed | 🔄 In Progress | 📋 Planned

### Future Enhancements

- [ ] Mobile application (Android/iOS)
- [ ] Edge device deployment (Raspberry Pi, NVIDIA Jetson)
- [ ] Multi-language support for sign names
- [ ] Night vision and adverse weather handling
- [ ] Integration with GPS for location-based validation
- [ ] Extended dataset support (US, EU, Asian traffic signs)

## 🤝 Contributing

We welcome contributions from the community! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code:
- Follows PEP 8 style guidelines
- Includes appropriate documentation
- Passes all existing tests
- Includes new tests for new features


## 🙏 Acknowledgments

- **Digital Egypt Pioneers Initiative (DEPI)** for providing the opportunity and support
- **German Traffic Sign Recognition Benchmark (GTSRB)** for the comprehensive dataset
- **TensorFlow and Keras** communities for excellent deep learning frameworks
- **Ultralytics** for the YOLOv8 implementation
- **OpenCV** contributors for computer vision tools
- All open-source contributors who made this project possible

## 📞 Contact

**Team Leader:** Abdelrahman Mohamed Abdelhaleem

- **GitHub**: [@AbdoOsary0](https://github.com/AbdoOsary0)
- **Project Repository**: [https://github.com/AbdoOsary0/Ml-project](https://github.com/AbdoOsary0/Ml-project)
- **Issues**: [Report a bug or request a feature](https://github.com/AbdoOsary0/Ml-project/issues)

---

**⭐ If you find this project useful, please consider giving it a star!**

**🎓 Developed as a graduation project for DEPI - AI/ML Track**
