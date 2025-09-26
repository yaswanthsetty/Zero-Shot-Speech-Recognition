# Contributing to Zero-Shot Spoken Language Identification

We welcome contributions to improve this zero-shot language identification system! This document provides guidelines for contributing.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/Zero-Shot-Speech-Recognition.git
   cd Zero-Shot-Speech-Recognition/zero-shot-lid
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 📋 Types of Contributions

### 🐛 Bug Reports
- Use the GitHub issue tracker
- Include system information, error messages, and reproduction steps
- Provide minimal code examples when possible

### ✨ Feature Requests
- Open an issue with detailed description
- Explain the use case and expected behavior
- Consider implementation complexity and scope

### 🔧 Code Contributions
- Follow the existing code style
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

## 💻 Development Guidelines

### Code Style
- Use **Black** for Python code formatting
- Follow **PEP 8** conventions
- Use type hints where appropriate
- Write descriptive docstrings

### Testing
- Add unit tests for new functions
- Ensure existing tests continue to pass
- Test with synthetic and real data when possible

### Documentation
- Update README.md for user-facing changes
- Add docstrings to new functions and classes
- Include code examples in docstrings

## 🏗️ Architecture Areas for Contribution

### High-Priority Areas
1. **Real Dataset Integration**: Fix FLEURS dataset loading
2. **Phonological Features**: Improve panphon integration
3. **Model Architectures**: Experiment with attention mechanisms
4. **Evaluation Metrics**: Add more comprehensive analysis

### Research Areas
1. **Cross-lingual Transfer**: Language family analysis
2. **Few-shot Learning**: Adaptation with minimal data
3. **Prosodic Features**: Beyond phonological characteristics
4. **Multi-modal Fusion**: Text + audio approaches

## 📝 Pull Request Process

1. **Update documentation** for any user-facing changes
2. **Add or update tests** for modified functionality
3. **Run the full test suite**:
   ```bash
   python -m pytest tests/
   python main.py  # Integration test
   ```
4. **Create descriptive PR title** and detailed description
5. **Link related issues** using keywords (fixes #123)

## 🔍 Code Review Guidelines

### For Contributors
- Keep PRs focused and reasonably sized
- Respond promptly to review feedback
- Be open to suggestions and improvements

### For Reviewers
- Be constructive and respectful
- Focus on code quality, maintainability, and correctness
- Test the changes locally when possible

## 📊 Performance Benchmarking

When contributing performance improvements:

1. **Benchmark before and after** changes
2. **Include timing and accuracy metrics**
3. **Test on different hardware** (CPU vs GPU)
4. **Document trade-offs** (speed vs accuracy)

## 🌍 Language Support

We encourage contributions for:
- Additional language phonological mappings
- Support for low-resource languages
- Cross-linguistic evaluation datasets
- Regional dialect handling

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Email**: For sensitive security issues

## 🏆 Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes for significant contributions
- Academic papers (for research contributions)

Thank you for helping improve zero-shot language identification! 🙏