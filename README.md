# SubFarsiPro - Persian Video Subtitle Translator

A professional tool to translate video subtitles from English to Persian (Farsi) using Whisper AI and advanced models like Ollama or Gemini.

## 🚀 Quick Start

See the [subfarsipro/](subfarsipro/) directory for the main application.

```bash
cd subfarsipro
python3 subfarsipro_v3.py
```

For detailed documentation, installation instructions, and usage guide, see [subfarsipro/README.md](subfarsipro/README.md).

## 📁 Project Structure

- `subfarsipro/` - Main application directory
  - `subfarsipro_v3.py` - Main application script (Version 3.0 with advanced CUDA support)
  - `README.md` - Complete documentation
  - `CUDA_INSTALLATION_GUIDE.md` - CUDA setup guide
  - `requirements.txt` - Python dependencies
- `run_subfarsipro.sh` - Convenience script to run with virtualenv

## ✨ Latest Features (Version 3.0)

- 🎮 Advanced CUDA detection and GPU compatibility checks
- 🛡️ Automatic fallback to CPU mode if GPU issues detected
- 🔍 Version mismatch detection for CUDA/PyTorch compatibility
- 📊 Smart GPU memory-based Whisper model selection

## 📖 Documentation

Full documentation is available in [subfarsipro/README.md](subfarsipro/README.md).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available for use.

---

Made with ❤️ for the Persian-speaking community
