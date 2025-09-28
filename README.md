# Zero-Shot Speech Recognition Research

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive research project implementing zero-shot spoken language identification using phonological features and deep learning techniques.

## � Recent Updates (September 2025)

- ⚡ **8x Performance Boost**: Implemented batch processing for feature extraction
- 🔧 **Fixed Dataset Loading**: Updated for latest HuggingFace datasets API
- 🛡️ **Enhanced Reliability**: Added robust error handling and fallback mechanisms
- 📊 **Better Progress Tracking**: Real-time processing updates
- 🐍 **Type Safety**: Fixed all type annotations and linting issues
- 🔄 **Automatic Fallbacks**: Synthetic data generation when real datasets fail

## �🎯 Project Overview

This repository contains a state-of-the-art implementation of zero-shot language identification that can recognize spoken languages without prior training on those specific languages. The system leverages cross-linguistic phonological features to enable transfer learning across language families.

## 🏗️ How It Works

```
Speech Audio → Audio Features → Phonological Space → Language Match
```

The system converts speech into phonological features (sound patterns) that are similar across related languages, enabling zero-shot transfer.

## 📁 Repository Structure

```
Zero-Shot-Speech-Recognition/
├── zero-shot-lid/              # 🎯 Main project (detailed docs inside)
│   ├── src/                    # Source code modules
│   ├── main.py                 # Run the complete system
│   ├── requirements.txt        # Dependencies
│   └── README.md              # 📖 Detailed technical documentation
├── models/                     # Generated model files
└── README.md                  # 👈 This overview file
```

**📖 For detailed technical documentation, installation instructions, and API reference, see [`zero-shot-lid/README.md`](zero-shot-lid/README.md)**

## 🚀 Quick Start

```bash
# Clone and run
git clone https://github.com/yaswanthsetty/Zero-Shot-Speech-Recognition.git
cd Zero-Shot-Speech-Recognition/zero-shot-lid
pip install -r requirements.txt
python main.py
```

**🔧 For detailed setup instructions, troubleshooting, and advanced usage, see [`zero-shot-lid/README.md`](zero-shot-lid/README.md)**

## ✨ Key Capabilities

- **� Zero-Shot Recognition**: Identify languages never seen during training
- **⚡ High Performance**: 8x faster processing with batch optimization
- **🛡️ Robust & Reliable**: Automatic fallbacks and error handling
- **� Research Ready**: Complete pipeline for academic and industry use
- **📊 Comprehensive**: Supports 20 languages (15 training + 5 testing)

## � Performance Highlights

| Metric | Value |
|--------|-------|
| **Languages** | 20 total (15 train + 5 test) |
| **Model Size** | 667K parameters (lightweight) |
| **Speed** | 8x faster (batch processing) |
| **Memory** | ~2GB RAM |
| **Expected Accuracy** | 45-80% (depending on language similarity) |

**🎯 Recent Updates**: 8x performance boost, fixed dataset loading, enhanced reliability

## 🔬 Research Applications

This implementation is suitable for:
- **Academic Research**: Cross-lingual transfer learning studies
- **Industry Applications**: Multilingual speech systems
- **Educational Use**: Understanding phonological features in ML
- **Baseline Development**: Comparison with other approaches



## � Research Applications

- **🎓 Academic Research**: Cross-lingual transfer learning studies
- **🏢 Industry Applications**: Multilingual speech systems
- **📚 Educational**: Understanding phonological features in ML
- **📊 Benchmarking**: Baseline for other approaches

## 🤝 Contributing

Contributions welcome! See [`zero-shot-lid/CONTRIBUTING.md`](zero-shot-lid/CONTRIBUTING.md) for guidelines.

## � Documentation

- **📋 Detailed Setup**: [`zero-shot-lid/README.md`](zero-shot-lid/README.md)
- **🔧 Technical Details**: Architecture, dependencies, configuration
- **🚨 Troubleshooting**: Common issues and solutions
- **📚 API Reference**: Function and class documentation
- **🧪 Experiments**: Performance analysis and research insights

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](zero-shot-lid/LICENSE) file for details.

## � License

MIT License - see [LICENSE](zero-shot-lid/LICENSE) for details.

## 📞 Contact

- **🐛 Issues**: [GitHub Issues](https://github.com/yaswanthsetty/Zero-Shot-Speech-Recognition/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/yaswanthsetty/Zero-Shot-Speech-Recognition/discussions)
- **👤 Author**: [Yaswanth Setty](https://github.com/yaswanthsetty)

---

<div align="center">

**🌍 Zero-shot cross-lingual understanding through phonological intelligence 🎤**

[📖 Detailed Documentation](zero-shot-lid/README.md) • [🚀 Quick Start](#-quick-start) • [🤝 Contributing](#-contributing)

</div>