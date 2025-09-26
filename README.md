# Zero-Shot Speech Recognition Research

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive research project implementing zero-shot spoken language identification using phonological features and deep learning techniques.

## 🎯 Project Overview

This repository contains a state-of-the-art implementation of zero-shot language identification that can recognize spoken languages without prior training on those specific languages. The system leverages cross-linguistic phonological features to enable transfer learning across language families.

## 🏗️ Architecture

```
Audio Input → Wav2Vec2 → Audio Embeddings → ProjectionHead → Phonological Space
                                                                     ↓
                                             Cosine Similarity → Language Prediction
                                                                     ↑
                                       Target Phonological Vectors ← Panphon
```

## 📁 Repository Structure

```
Zero-Shot-Speech-Recognition/
├── zero-shot-lid/              # Main project directory
│   ├── src/                    # Source code modules
│   │   ├── config.py          # Configuration and hyperparameters
│   │   ├── data_prep.py       # Data loading and preprocessing
│   │   ├── features.py        # Audio and phonological features
│   │   ├── model.py           # Neural network architectures
│   │   ├── train.py           # Training pipeline
│   │   └── evaluate.py        # Evaluation and metrics
│   ├── .devcontainer/         # Development environment
│   ├── .github/workflows/     # CI/CD pipelines
│   ├── main.py               # Main execution script
│   ├── requirements.txt      # Python dependencies
│   └── README.md            # Detailed project documentation
├── models/                   # Model checkpoints (generated)
└── README.md                # This file
```

## 🚀 Quick Start

### Option 1: GitHub Codespaces (Recommended)
1. Click "Code" → "Codespaces" → "Create codespace on main"
2. Wait for environment setup (automatic)
3. Run the project:
   ```bash
   cd zero-shot-lid
   python main.py
   ```

### Option 2: Local Development
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yaswanthsetty/Zero-Shot-Speech-Recognition.git
   cd Zero-Shot-Speech-Recognition/zero-shot-lid
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the system:**
   ```bash
   python main.py
   ```

## 🧪 Key Features

- **🎵 Advanced Audio Processing**: Wav2Vec2-based feature extraction
- **🧬 Phonological Intelligence**: Cross-linguistic feature representation
- **🧠 Neural Projection Learning**: Deep learning for embedding alignment
- **🌍 Zero-Shot Capability**: Recognition of completely unseen languages
- **📊 Comprehensive Evaluation**: Multi-metric performance analysis
- **🔧 Production Ready**: Robust error handling and logging

## 📈 Performance

### System Capabilities
- **Languages Supported**: 20 total (15 for training, 5 for zero-shot testing)
- **Model Size**: 667K parameters (lightweight and efficient)
- **Processing Speed**: Real-time capable on modern hardware
- **Memory Usage**: ~2GB RAM for inference

### Expected Results (with real data)
- **Cross-family Transfer**: 45-65% Top-1 accuracy
- **Within-family Transfer**: 65-80% Top-1 accuracy
- **Top-3 Performance**: +15-25% improvement over Top-1

## 🔬 Research Applications

This implementation is suitable for:
- **Academic Research**: Cross-lingual transfer learning studies
- **Industry Applications**: Multilingual speech systems
- **Educational Use**: Understanding phonological features in ML
- **Baseline Development**: Comparison with other approaches

## 🛠️ Development

### Contributing
We welcome contributions! See [CONTRIBUTING.md](zero-shot-lid/CONTRIBUTING.md) for guidelines.

### Key Areas for Enhancement
- **Dataset Integration**: Real multilingual speech corpora
- **Architecture Innovation**: Attention mechanisms and transformers  
- **Evaluation Expansion**: More languages and metrics
- **Production Optimization**: Speed and memory improvements

## 📚 Technical Details

### Dependencies
- **PyTorch 2.8+**: Deep learning framework
- **Transformers 4.56+**: Pre-trained model access
- **Panphon 0.22+**: Phonological feature extraction
- **Librosa 0.11+**: Audio processing utilities

### Model Architecture
- **Input**: Variable-length audio (up to 30s, 16kHz)
- **Encoder**: Wav2Vec2-base-960h (frozen)
- **Projection**: 2-layer MLP (768→512→22 dimensions)
- **Output**: Phonological feature space for similarity comparison

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](zero-shot-lid/LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Research** for FLEURS multilingual speech corpus
- **Facebook AI Research** for Wav2Vec2 pre-trained models  
- **David R. Mortensen** for Panphon phonological features library
- **Hugging Face** for transformers ecosystem

## 📞 Contact

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and research collaboration
- **Author**: Yaswanth Setty

---

**🌍 Enabling cross-lingual speech understanding through computational phonology 🎤**