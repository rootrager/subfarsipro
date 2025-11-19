#!/usr/bin/env python3
"""
Persian Video Subtitle Translator - Professional Edition
A professional tool for translating video subtitles to Persian using Whisper + Ollama/Gemini
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import time

try:
    import whisper
    import google.generativeai as genai
    import requests
    import torch
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("📦 Please install: pip install git+https://github.com/openai/whisper.git google-generativeai requests torch")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def show_logo():
    """Display the Persian Translator logo"""
    logo = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║        ███████╗██╗   ██╗██████╗ ███████╗ █████╗ ██████╗ ███████╗██╗          ║
║        ██╔════╝██║   ██║██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔════╝██║          ║
║        ███████╗██║   ██║██████╔╝█████╗  ███████║██████╔╝███████╗██║          ║
║        ╚════██║██║   ██║██╔══██╗██╔══╝  ██╔══██║██╔══██╗╚════██║██║          ║
║        ███████║╚██████╔╝██████╔╝██║     ██║  ██║██║  ██║███████║██║          ║
║        ╚══════╝ ╚═════╝ ╚═════╝ ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝          ║
║                                                                              ║
║                            ██████╗ ██████╗  ██████╗                          ║
║                            ██╔══██╗██╔══██╗██╔═══██╗                         ║
║                            ██████╔╝██████╔╝██║   ██║                         ║
║                            ██╔═══╝ ██╔══██╗██║   ██║                         ║
║                            ██║     ██║  ██║╚██████╔╝                         ║
║                            ╚═╝     ╚═╝  ╚═╝ ╚═════╝                          ║
║                                                                              ║
║                    🎬 Professional Video Subtitle Translator 🎬              ║
║                          English → Persian (Farsi)                           ║ 
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(logo)
    print("🚀 Welcome to Persian Translator - Your Professional Subtitle Solution!")
    print("📝 Powered by Whisper AI + Ollama/Gemini for accurate Persian translations")
    print("=" * 80)

def show_features():
    """Display tool features and capabilities"""
    print("\n✨ FEATURES & CAPABILITIES:")
    print("  🎥 Extract audio from any video format (MP4, AVI, MKV, etc.)")
    print("  🎤 Generate accurate English subtitles using OpenAI Whisper")
    print("  🌍 Translate to natural Persian (Farsi) using AI models")
    print("  🤖 Support for Ollama (local) and Gemini API models")
    print("  📄 Export professional SRT subtitle files")
    print("  🔧 Automatic timing adjustment and error handling")
    print("  📊 Detailed progress tracking and quality reports")
    
def get_system_cuda_version() -> Optional[str]:
    """Get system CUDA version from nvcc (if available)"""
    try:
        result = subprocess.run(
            ['nvcc', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Parse version from output like "release 11.5, V11.5.119"
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.lower() == 'release':
                            if i + 1 < len(parts):
                                version = parts[i + 1].rstrip(',')
                                return version
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logger.debug(f"Could not detect system CUDA version: {e}")
    return None

def check_cuda_status() -> dict:
    """Check CUDA installation status and provide information"""
    status = {
        'cuda_available': False,
        'cuda_version': None,
        'system_cuda_version': None,
        'pytorch_cuda_version': None,
        'gpu_name': None,
        'gpu_memory_gb': None,
        'pytorch_cuda': False,
        'pytorch_version': None,
        'version_mismatch': False,
        'cuda_test_error': None
    }
    
    # Get system CUDA version
    status['system_cuda_version'] = get_system_cuda_version()
    
    try:
        # Check PyTorch CUDA support
        status['pytorch_cuda'] = torch.cuda.is_available()
        status['pytorch_version'] = torch.__version__
        
        if status['pytorch_cuda']:
            status['pytorch_cuda_version'] = torch.version.cuda
            status['cuda_version'] = status['pytorch_cuda_version']
            
            # Try to get GPU info (might fail if there's a version mismatch)
            try:
                status['gpu_name'] = torch.cuda.get_device_name(0)
                status['gpu_memory_gb'] = get_gpu_memory_gb()
                status['cuda_available'] = True
            except Exception as e:
                status['cuda_test_error'] = str(e)
                logger.warning(f"CUDA available but GPU access failed: {e}")
            
            # Check for version mismatch
            if status['system_cuda_version'] and status['pytorch_cuda_version']:
                sys_version = float('.'.join(status['system_cuda_version'].split('.')[:2]))
                pytorch_version = float('.'.join(status['pytorch_cuda_version'].split('.')[:2]))
                
                # Allow minor version differences (e.g., 11.5 vs 11.8)
                if abs(sys_version - pytorch_version) > 0.5:
                    status['version_mismatch'] = True
                    logger.warning(f"CUDA version mismatch detected: System={sys_version}, PyTorch={pytorch_version}")
        
        # Test CUDA with a simple operation
        if status['pytorch_cuda']:
            try:
                test_tensor = torch.tensor([1.0], device='cuda')
                test_result = test_tensor * 2
                del test_tensor, test_result
                torch.cuda.empty_cache()
            except Exception as e:
                status['cuda_test_error'] = str(e)
                logger.warning(f"CUDA test operation failed: {e}")
                status['cuda_available'] = False
                
    except Exception as e:
        logger.debug(f"CUDA check failed: {e}")
        status['cuda_test_error'] = str(e)
    
    return status

def show_cuda_info():
    """Display CUDA information and installation guide"""
    print("\n" + "="*70)
    print("🎮 CUDA INFORMATION & INSTALLATION GUIDE")
    print("="*70)
    
    status = check_cuda_status()
    
    # Display version information
    if status['system_cuda_version']:
        print(f"\n📋 System CUDA Version (nvcc): {status['system_cuda_version']}")
    if status['pytorch_cuda_version']:
        print(f"📋 PyTorch CUDA Version: {status['pytorch_cuda_version']}")
    if status['pytorch_version']:
        print(f"📋 PyTorch Version: {status['pytorch_version']}")
    
    # Check for version mismatch
    if status['version_mismatch']:
        print("\n⚠️  WARNING: CUDA Version Mismatch Detected!")
        print(f"   • System CUDA: {status['system_cuda_version']}")
        print(f"   • PyTorch CUDA: {status['pytorch_cuda_version']}")
        print("\n   This mismatch may cause runtime errors!")
        print("   See solution guide below.")
    
    if status['cuda_available'] and not status['version_mismatch']:
        print("\n✅ CUDA is installed and working!")
        print(f"  • CUDA Version: {status['cuda_version']}")
        if status['gpu_name']:
            print(f"  • GPU: {status['gpu_name']}")
        if status['gpu_memory_gb']:
            print(f"  • GPU Memory: {status['gpu_memory_gb']:.2f} GB")
        print("\n💡 Your GPU will be used for faster Whisper processing!")
    elif status['pytorch_cuda'] and status['cuda_test_error']:
        print("\n⚠️  CUDA is detected but not working properly!")
        print(f"   Error: {status['cuda_test_error']}")
        if status['version_mismatch']:
            print("\n   🔧 SOLUTION: Install PyTorch compatible with your CUDA version")
            if status['system_cuda_version']:
                sys_version = status['system_cuda_version']
                print(f"\n   For CUDA {sys_version}:")
                if sys_version.startswith('11.5') or sys_version.startswith('11.4'):
                    print("   pip uninstall torch torchvision torchaudio")
                    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
                    print("   (Note: CUDA 11.5 is compatible with cu118 builds)")
                elif sys_version.startswith('11.8'):
                    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
                elif sys_version.startswith('12.1') or sys_version.startswith('12.0'):
                    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("\n   Alternatively, use CPU mode (option 3 in Whisper configuration)")
    else:
        print("\n⚠️  CUDA is NOT installed or not available")
        print("\n📖 WHAT IS CUDA?")
        print("   CUDA (Compute Unified Device Architecture) is NVIDIA's parallel")
        print("   computing platform that allows programs to use GPU for")
        print("   acceleration. It makes AI/ML tasks much faster!")
        print("\n🔍 WHY DO YOU NEED IT?")
        print("   • Without CUDA: Whisper runs on CPU (slow, but works)")
        print("   • With CUDA: Whisper runs on GPU (10-50x faster!)")
        print("\n📦 INSTALLATION GUIDE:")
        print("\n   1️⃣  Check if you have NVIDIA GPU:")
        print("      Run: nvidia-smi")
        print("      If you see GPU info → You have NVIDIA GPU")
        print("      If error → You don't have NVIDIA GPU (use CPU mode)")
        print("\n   2️⃣  Install CUDA Toolkit:")
        print("      • Visit: https://developer.nvidia.com/cuda-downloads")
        print("      • Select your OS (Linux/Windows)")
        print("      • Download and install CUDA Toolkit (11.8, 12.1, or 12.4 recommended)")
        print("\n   3️⃣  Install PyTorch with CUDA support:")
        if status['system_cuda_version']:
            sys_version = status['system_cuda_version']
            print(f"\n      For CUDA {sys_version} on your system:")
            if sys_version.startswith('11.5') or sys_version.startswith('11.4'):
                print("      pip uninstall torch torchvision torchaudio")
                print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
                print("      (CUDA 11.5 works with cu118 builds)")
            elif sys_version.startswith('11.8'):
                print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            elif sys_version.startswith('12.1') or sys_version.startswith('12.0'):
                print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            else:
                print("      Check PyTorch website for compatible version:")
                print("      https://pytorch.org/get-started/locally/")
        
        print("\n      General options:")
        print("      For CUDA 11.8:")
        print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("\n      For CUDA 12.1:")
        print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("\n   4️⃣  Verify installation:")
        print("      python -c \"import torch; print(torch.cuda.is_available())\"")
        print("      Should print: True")
        print("\n💡 NOTE: You can use this tool WITHOUT CUDA (CPU mode)")
        print("   It will be slower but will work perfectly fine!")
    print("="*70)

def show_requirements():
    """Display system requirements"""
    os_name = platform.system()
    print("\n📋 SYSTEM REQUIREMENTS:")
    print(f"  • Operating System: {os_name}")
    print("  • FFmpeg installed (for audio extraction)")
    print("  • Python 3.7+ with required packages")
    print("  • Ollama running locally (recommended) OR Gemini API key")
    print("  • CUDA (optional - for GPU acceleration, much faster)")
    print("  • Sufficient disk space for temporary audio files")

def find_ffmpeg() -> Optional[str]:
    """Find FFmpeg executable path (cross-platform)"""
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        return ffmpeg_path
    
    # Try common Windows locations
    if platform.system() == 'Windows':
        common_paths = [
            r'C:\ffmpeg\bin\ffmpeg.exe',
            r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
            r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe',
        ]
        for path in common_paths:
            if os.path.exists(path):
                return path
    
    return None

def get_gpu_memory_gb() -> Optional[float]:
    """Get available GPU memory in GB"""
    try:
        if torch.cuda.is_available():
            # Get total GPU memory
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return total_memory
    except Exception as e:
        logger.debug(f"Could not detect GPU memory: {e}")
    return None

def test_cuda_operation() -> Tuple[bool, Optional[str]]:
    """Test if CUDA operations actually work"""
    try:
        # Test tensor creation and operation
        test_tensor = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        result = test_tensor * 2.0
        result_cpu = result.cpu()
        del test_tensor, result
        torch.cuda.empty_cache()
        return True, None
    except Exception as e:
        error_msg = str(e)
        return False, error_msg

def load_whisper_safely(model_preference: Optional[str] = None, force_cpu: bool = False) -> whisper.Whisper:
    """
    Load Whisper model with automatic GPU memory detection and smart model selection.
    Includes CUDA compatibility checks and error handling.
    
    Args:
        model_preference: Preferred model size ('tiny', 'base', 'small', 'medium', 'large') or None for auto
        force_cpu: Force CPU usage even if GPU is available
    
    Returns:
        Loaded Whisper model
    """
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available() and not force_cpu
    
    if force_cpu or not cuda_available:
        print("💻 Using CPU for Whisper (slower but no VRAM issues)")
        if model_preference:
            print(f"📦 Loading Whisper model: {model_preference} (CPU)")
            return whisper.load_model(model_preference, device="cpu")
        else:
            print("📦 Loading Whisper model: base (CPU - recommended for CPU)")
            return whisper.load_model("base", device="cpu")
    
    # Test CUDA before proceeding
    print("🔍 Testing CUDA compatibility...")
    cuda_works, cuda_error = test_cuda_operation()
    
    if not cuda_works:
        print(f"⚠️  CUDA test failed: {cuda_error}")
        print("   This usually indicates a CUDA version mismatch or driver issue.")
        
        # Check CUDA status for diagnostic info
        status = check_cuda_status()
        if status['version_mismatch']:
            print(f"\n   🔧 DETECTED: CUDA Version Mismatch!")
            print(f"   • System CUDA: {status['system_cuda_version']}")
            print(f"   • PyTorch CUDA: {status['pytorch_cuda_version']}")
            print("\n   SOLUTION: Install PyTorch compatible with your CUDA version")
            if status['system_cuda_version'] and status['system_cuda_version'].startswith('11.5'):
                print("\n   For CUDA 11.5, run:")
                print("   pip uninstall torch torchvision torchaudio")
                print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        else:
            print("\n   💡 TIP: Check your NVIDIA driver and CUDA installation")
            print("   Run: nvidia-smi to verify driver is working")
        
        print("\n   Falling back to CPU mode...")
        if model_preference:
            print(f"📦 Loading Whisper model: {model_preference} (CPU)")
            return whisper.load_model(model_preference, device="cpu")
        else:
            print("📦 Loading Whisper model: base (CPU)")
            return whisper.load_model("base", device="cpu")
    
    print("✅ CUDA test passed!")
    
    # GPU available - detect memory and select appropriate model
    gpu_memory = get_gpu_memory_gb()
    
    if gpu_memory is None:
        print("⚠️  Could not detect GPU memory. Using base model on CPU for safety.")
        return whisper.load_model("base", device="cpu")
    
    print(f"🎮 GPU detected: {gpu_memory:.2f} GB VRAM")
    
    # Model selection based on GPU memory
    if model_preference:
        # User specified a model - try to use it
        model_size_map = {
            'tiny': 1.0,   # ~1GB
            'base': 1.5,   # ~1.5GB
            'small': 2.5,  # ~2.5GB
            'medium': 5.0, # ~5GB
            'large': 10.0  # ~10GB
        }
        
        required_memory = model_size_map.get(model_preference.lower(), 1.5)
        
        if gpu_memory < required_memory:
            print(f"⚠️  GPU memory ({gpu_memory:.2f} GB) may be insufficient for '{model_preference}' model")
            print(f"   Required: ~{required_memory} GB. Trying anyway...")
        
        try:
            # Try loading with FP16 disabled for low VRAM
            use_fp16 = gpu_memory >= 4.0
            print(f"📦 Loading Whisper model: {model_preference} (GPU, FP16={'enabled' if use_fp16 else 'disabled'})")
            model = whisper.load_model(model_preference, device="cuda", fp16=use_fp16)
            print("✅ Model loaded successfully on GPU!")
            return model
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg:
                print(f"❌ Out of memory! Falling back to smaller model...")
                try:
                    print("📦 Loading Whisper model: base (GPU, FP16=False)")
                    model = whisper.load_model("base", device="cuda", fp16=False)
                    print("✅ Model loaded successfully on GPU!")
                    return model
                except RuntimeError as e2:
                    print(f"❌ Still out of memory. Falling back to CPU...")
                    print(f"📦 Loading Whisper model: {model_preference} (CPU)")
                    return whisper.load_model(model_preference, device="cpu")
            elif "cuda" in error_msg or "gpu" in error_msg:
                print(f"❌ CUDA error detected: {e}")
                print("   Falling back to CPU mode...")
                print(f"📦 Loading Whisper model: {model_preference} (CPU)")
                return whisper.load_model(model_preference, device="cpu")
            else:
                # Unknown error, try CPU fallback
                print(f"⚠️  Error loading model: {e}")
                print("   Attempting CPU fallback...")
                try:
                    return whisper.load_model(model_preference, device="cpu")
                except Exception:
                    raise
    
    # Auto-select model based on GPU memory
    try:
        if gpu_memory < 2.0:
            print("⚠️  Very low VRAM detected. Using 'tiny' model (fastest, lower accuracy)")
            model = whisper.load_model("tiny", device="cuda", fp16=False)
            print("✅ Model loaded successfully on GPU!")
            return model
        elif gpu_memory < 4.0:
            print("⚠️  Low VRAM detected. Using 'base' model (good balance)")
            model = whisper.load_model("base", device="cuda", fp16=False)
            print("✅ Model loaded successfully on GPU!")
            return model
        elif gpu_memory < 6.0:
            print("✅ Using 'small' model (better accuracy)")
            model = whisper.load_model("small", device="cuda", fp16=True)
            print("✅ Model loaded successfully on GPU!")
            return model
        elif gpu_memory < 10.0:
            print("✅ Using 'medium' model (high accuracy)")
            model = whisper.load_model("medium", device="cuda", fp16=True)
            print("✅ Model loaded successfully on GPU!")
            return model
        else:
            print("✅ Using 'large' model (best accuracy)")
            model = whisper.load_model("large", device="cuda", fp16=True)
            print("✅ Model loaded successfully on GPU!")
            return model
    except RuntimeError as e:
        error_msg = str(e).lower()
        if "cuda" in error_msg or "gpu" in error_msg or "out of memory" in error_msg:
            print(f"❌ GPU loading failed: {e}")
            print("   Falling back to CPU mode with base model...")
            return whisper.load_model("base", device="cpu")
        else:
            raise

def get_user_choice(prompt: str, options: List[str], default: int = 0) -> str:
    """Get user choice from a list of options"""
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        marker = " (default)" if i-1 == default else ""
        print(f"  {i}. {option}{marker}")
    
    while True:
        try:
            choice = input(f"\nEnter choice (1-{len(options)}) or press Enter for default: ").strip()
            if not choice:
                return options[default]
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(options):
                return options[choice_num - 1]
            else:
                print(f"❌ Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("❌ Please enter a valid number")

def get_video_path() -> str:
    """Get and validate video file path (cross-platform)"""
    while True:
        print("\n📁 VIDEO FILE SELECTION:")
        video_path = input("Enter the path to your video file: ").strip().strip('"\'')
        
        if not video_path:
            print("❌ Please enter a video file path")
            continue
        
        # Normalize path for cross-platform compatibility
        try:
            video_path = str(Path(video_path).expanduser().resolve())
        except Exception as e:
            print(f"❌ Invalid path format: {e}")
            continue
            
        if not os.path.exists(video_path):
            print(f"❌ File not found: {video_path}")
            print("💡 Tip: You can drag and drop the file into the terminal")
            if platform.system() == 'Windows':
                print("   On Windows, make sure to use backslashes or forward slashes")
            continue
        
        if not os.path.isfile(video_path):
            print(f"❌ Path is not a file: {video_path}")
            continue
            
        # Check if it's likely a video file
        video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v']
        if not any(video_path.lower().endswith(ext) for ext in video_extensions):
            confirm = input("⚠️  This doesn't look like a video file. Continue anyway? (y/N): ").strip().lower()
            if confirm != 'y':
                continue
                
        return video_path

def get_output_path(video_path: str) -> str:
    """Get output subtitle file path (cross-platform)"""
    # Normalize video path first
    video_path = str(Path(video_path).resolve())
    default_output = str(Path(video_path).with_suffix('.persian.srt'))
    print(f"\n💾 OUTPUT FILE:")
    print(f"Default output: {default_output}")
    
    custom_path = input("Enter custom output path (or press Enter for default): ").strip().strip('"\'')
    
    if custom_path:
        # Normalize custom path
        try:
            custom_path = str(Path(custom_path).expanduser().resolve())
        except Exception:
            # If resolution fails, use as-is but ensure .srt extension
            pass
        
        if not custom_path.endswith('.srt'):
            custom_path += '.srt'
        return custom_path
    
    return default_output

def test_ollama_model(model_name: str, ollama_url: str = "http://localhost:11434/api/generate") -> bool:
    """Test if Ollama model is available"""
    try:
        payload = {
            "model": model_name,
            "prompt": "Hello",
            "stream": False
        }
        response = requests.post(ollama_url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except Exception:
        return False

def get_ollama_model() -> Optional[str]:
    """Get and validate Ollama model"""
    print("\n🤖 OLLAMA MODEL SELECTION:")
    print("Common Persian-capable models: gemma3, llama4, qwen, deepseek, etc.")
    
    while True:
        model_name = input("Enter your Ollama model name (e.g., gemma3): ").strip().lower()
        
        if not model_name:
            print("❌ Please enter a model name")
            continue
            
        print(f"🔍 Testing connection to model '{model_name}'...")
        if test_ollama_model(model_name):
            print(f"✅ Model '{model_name}' is available!")
            return model_name
        else:
            print(f"❌ Model '{model_name}' not found or Ollama not running")
            print("💡 Make sure Ollama is running and the model is installed")
            
            retry = input("Try another model? (Y/n): ").strip().lower()
            if retry == 'n':
                return None

def get_gemini_config() -> Optional[str]:
    """Get Gemini API configuration"""
    print("\n🌟 GEMINI API (Optional):")
    print("Gemini can be used as fallback if Ollama fails")
    
    use_gemini = input("Do you want to configure Gemini API? (y/N): ").strip().lower()
    if use_gemini != 'y':
        return None
        
    api_key = input("Enter your Gemini API key: ").strip()
    if api_key:
        return api_key
    return None

def get_gemini_config_required() -> Optional[str]:
    """Get Gemini API configuration (required mode)"""
    print("\n🌟 GEMINI API CONFIGURATION:")
    print("Please enter your Gemini API key to continue")
    
    while True:
        api_key = input("Enter your Gemini API key: ").strip()
        
        if api_key:
            return api_key
        else:
            print("❌ API key is required for Gemini-only mode")
            retry = input("Try again? (Y/n): ").strip().lower()
            if retry == 'n':
                return None

def get_whisper_config() -> Tuple[Optional[str], bool]:
    """
    Get Whisper model configuration from user
    
    Returns:
        Tuple of (model_preference, force_cpu)
    """
    print("\n🎤 WHISPER MODEL CONFIGURATION:")
    
    # Show GPU info if available
    if torch.cuda.is_available():
        gpu_memory = get_gpu_memory_gb()
        if gpu_memory:
            print(f"🎮 GPU detected: {gpu_memory:.2f} GB VRAM")
        else:
            print("🎮 GPU detected (memory info unavailable)")
    else:
        print("💻 No GPU detected - will use CPU")
    
    print("\nOptions:")
    print("  1. Auto-select based on GPU memory (recommended)")
    print("  2. Choose specific model size")
    print("  3. Force CPU usage (no VRAM issues, but slower)")
    
    choice = input("\nEnter choice (1-3, default=1): ").strip()
    
    if not choice or choice == '1':
        return (None, False)  # Auto-select
    elif choice == '2':
        print("\nAvailable Whisper models:")
        print("  tiny   - ~1GB VRAM, fastest, lower accuracy")
        print("  base   - ~1.5GB VRAM, good balance (recommended for 4GB GPU)")
        print("  small  - ~2.5GB VRAM, better accuracy")
        print("  medium - ~5GB VRAM, high accuracy")
        print("  large  - ~10GB VRAM, best accuracy")
        
        model = input("\nEnter model name (tiny/base/small/medium/large, default=base): ").strip().lower()
        valid_models = ['tiny', 'base', 'small', 'medium', 'large']
        
        if not model or model not in valid_models:
            model = 'base'
            print(f"Using default: {model}")
        
        use_cpu = input("Force CPU usage? (y/N): ").strip().lower() == 'y'
        return (model, use_cpu)
    elif choice == '3':
        print("💻 CPU mode selected (slower but no VRAM issues)")
        model = input("Model size for CPU (tiny/base/small, default=base): ").strip().lower()
        valid_models = ['tiny', 'base', 'small']
        if not model or model not in valid_models:
            model = 'base'
        return (model, True)
    else:
        print("Invalid choice, using auto-select")
        return (None, False)

class PersianSubtitleTranslator:
    def __init__(self, ollama_model: Optional[str] = None, gemini_api_key: Optional[str] = None, 
                 ollama_url: str = "http://localhost:11434/api/generate", timing_offset: float = 0.5,
                 whisper_model: Optional[str] = None, force_cpu: bool = False):
        """
        Initialize the professional Persian subtitle translator
        
        Args:
            ollama_model: Ollama model name for translation
            gemini_api_key: Gemini API key for translation
            ollama_url: Ollama API URL
            timing_offset: Timing offset for subtitles (seconds)
            whisper_model: Whisper model size ('tiny', 'base', 'small', 'medium', 'large') or None for auto
            force_cpu: Force CPU usage for Whisper (slower but no VRAM issues)
        """
        self.ollama_model = ollama_model
        self.gemini_api_key = gemini_api_key
        self.ollama_url = ollama_url
        self.timing_offset = timing_offset
        
        # Initialize Gemini if API key provided
        if gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("✅ Gemini API configured successfully")
            except Exception as e:
                logger.warning(f"⚠️  Gemini API configuration failed: {e}")
                self.gemini_model = None
        else:
            self.gemini_model = None
        
        # Load Whisper model with smart GPU memory detection
        print("\n🎤 Loading Whisper AI model...")
        try:
            self.whisper_model = load_whisper_safely(model_preference=whisper_model, force_cpu=force_cpu)
            print("✅ Whisper model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            print("⚠️  Falling back to CPU mode with base model...")
            try:
                self.whisper_model = whisper.load_model("base", device="cpu")
                print("✅ Whisper model loaded on CPU!")
            except Exception as e2:
                raise Exception(f"❌ Failed to load Whisper model: {e2}")
        
    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video file using FFmpeg (cross-platform)"""
        # Normalize video path for cross-platform compatibility
        video_path = str(Path(video_path).resolve())
        
        # Create audio path in same directory as video
        audio_path = str(Path(video_path).with_suffix('.wav'))
        print(f"\n🎵 Extracting audio from video...")
        
        # Find FFmpeg executable
        ffmpeg_exe = find_ffmpeg()
        if not ffmpeg_exe:
            raise Exception("❌ FFmpeg not found. Please install FFmpeg first.\n"
                          f"   On Linux/Mac: sudo apt install ffmpeg  or  brew install ffmpeg\n"
                          f"   On Windows: Download from https://ffmpeg.org/download.html")
        
        try:
            # Build FFmpeg command (cross-platform)
            cmd = [
                ffmpeg_exe, '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1', '-y', audio_path
            ]
            
            # Run FFmpeg with proper encoding for error messages
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True,
                encoding='utf-8',
                errors='replace'
            )
            print(f"✅ Audio extracted successfully!")
            return audio_path
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            logger.error(f"FFmpeg error: {error_msg}")
            raise Exception(f"❌ Failed to extract audio.\n"
                          f"   Error: {error_msg[:200]}\n"
                          f"   Make sure FFmpeg is installed and the video file is valid.")
        except Exception as e:
            logger.error(f"Unexpected error during audio extraction: {e}")
            raise Exception(f"❌ Unexpected error: {e}")
        
    def transcribe_audio(self, audio_path: str) -> List[Dict]:
        """Convert audio to English subtitles using Whisper"""
        # Validate audio file exists
        if not os.path.exists(audio_path):
            raise Exception(f"❌ Audio file not found: {audio_path}")
        
        print("\n📝 Transcribing audio to English subtitles...")
        
        try:
            result = self.whisper_model.transcribe(audio_path)
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            raise Exception(f"❌ Failed to transcribe audio: {e}")
        
        # Validate result structure
        if not isinstance(result, dict) or 'segments' not in result:
            raise Exception("❌ Invalid transcription result format")
        
        subtitles = []
        segments = result.get('segments', [])
        
        if not segments:
            logger.warning("⚠️  No segments found in transcription. Video might be silent or too short.")
            return subtitles
        
        for i, segment in enumerate(segments):
            # Validate segment structure
            if not isinstance(segment, dict):
                logger.warning(f"⚠️  Skipping invalid segment at index {i}")
                continue
            
            text = segment.get('text', '').strip()
            # Skip empty segments
            if not text:
                continue
            
            start_time = segment.get('start', 0)
            end_time = segment.get('end', start_time + 1)
            
            # Ensure valid timing
            adjusted_start = max(0, float(start_time) + self.timing_offset)
            adjusted_end = max(adjusted_start, float(end_time) + self.timing_offset)
            
            subtitle = {
                'index': len(subtitles) + 1,
                'start': adjusted_start,
                'end': adjusted_end,
                'text': text
            }
            subtitles.append(subtitle)
        
        if not subtitles:
            logger.warning("⚠️  No valid subtitle segments generated")
        
        print(f"✅ Generated {len(subtitles)} subtitle segments")
        return subtitles

    def translate_with_ollama(self, text: str) -> Optional[str]:
        """Translate text using Ollama"""
        # Validate inputs
        if not self.ollama_model:
            logger.warning("⚠️  Ollama model not configured")
            return None
        
        if not text or not text.strip():
            logger.warning("⚠️  Empty text provided for translation")
            return None
        
        prompt = f"""Translate this English text to natural Persian (Farsi). 
Make it conversational and suitable for video subtitles. 
Return only the Persian translation, nothing else.

English: {text.strip()}

Persian:"""

        try:
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            }
            response = requests.post(self.ollama_url, json=payload, timeout=120)
            response.raise_for_status()
            
            # Safely parse JSON response
            try:
                result = response.json()
            except (ValueError, requests.exceptions.JSONDecodeError) as e:
                logger.error(f"Invalid JSON response from Ollama: {e}")
                logger.debug(f"Response content: {response.text[:200]}")
                return None
            
            # Safely extract translation
            if not isinstance(result, dict):
                logger.error(f"Unexpected response format from Ollama: {type(result)}")
                return None
                
            translation = result.get('response', '').strip()
            
            # Clean up response
            if translation:
                prefixes = ['Persian:', 'Farsi:', 'Translation:', 'فارسی:']
                for prefix in prefixes:
                    if translation.startswith(prefix):
                        translation = translation[len(prefix):].strip()
                
                if translation.startswith('"') and translation.endswith('"'):
                    translation = translation[1:-1].strip()
                
                return translation
                    
        except Exception as e:
            logger.error(f"Ollama translation failed: {e}")
            return None

    def translate_with_gemini(self, text: str) -> Optional[str]:
        """Translate text using Gemini API"""
        if not self.gemini_model:
            return None
        
        # Validate input
        if not text or not text.strip():
            logger.warning("⚠️  Empty text provided for translation")
            return None
            
        prompt = f"""Translate this English text to natural Persian (Farsi).
Make it conversational and suitable for video subtitles.
Return only the Persian translation, nothing else.

English: {text.strip()}

Persian:"""

        try:
            response = self.gemini_model.generate_content(prompt)
            
            # Validate response
            if not response or not hasattr(response, 'text'):
                logger.warning("⚠️  Invalid response from Gemini API")
                return None
            
            translation = response.text.strip()
            
            # Return None if translation is empty
            if not translation:
                logger.warning("⚠️  Empty translation received from Gemini")
                return None
            
            return translation
        except Exception as e:
            logger.error(f"Gemini translation failed: {e}")
            return None

    def translate_subtitles(self, subtitles: List[Dict]) -> List[Dict]:
        """Translate all subtitles to Persian"""
        # Validate input
        if not subtitles:
            logger.warning("⚠️  No subtitles to translate")
            return []
        
        print(f"\n🌍 Translating {len(subtitles)} subtitles to Persian...")
        
        # Determine translation strategy
        if self.ollama_model and self.gemini_model:
            strategy_msg = f"Using: {self.ollama_model} (Ollama) + Gemini fallback"
        elif self.ollama_model:
            strategy_msg = f"Using: {self.ollama_model} (Ollama only)"
        elif self.gemini_model:
            strategy_msg = "Using: Gemini API only (Google AI)"
        else:
            raise Exception("No translation service available")
        
        print(strategy_msg)
        
        translated_subtitles = []
        successful_translations = 0
        ollama_successes = 0
        gemini_successes = 0
        
        for i, subtitle in enumerate(subtitles):
            # Validate subtitle structure
            if not isinstance(subtitle, dict):
                logger.warning(f"⚠️  Invalid subtitle at index {i}, skipping")
                continue
            
            subtitle_text = subtitle.get('text', '').strip()
            
            # Skip empty subtitles
            if not subtitle_text:
                logger.debug(f"⚠️  Empty subtitle at index {i}, skipping translation")
                translated_subtitle = subtitle.copy()
                translated_subtitle['text'] = ""
                translated_subtitles.append(translated_subtitle)
                continue
            # Progress indicator
            if i % 10 == 0 or i == len(subtitles) - 1:
                progress = ((i + 1) / len(subtitles)) * 100
                print(f"📊 Progress: {progress:.1f}% ({i+1}/{len(subtitles)})")
            
            persian_text = None
            
            # Translation logic based on available services
            if self.ollama_model:
                # Try Ollama first (if available)
                persian_text = self.translate_with_ollama(subtitle_text)
                if persian_text:
                    ollama_successes += 1
                elif self.gemini_model:
                    # Fallback to Gemini
                    persian_text = self.translate_with_gemini(subtitle_text)
                    if persian_text:
                        gemini_successes += 1
            elif self.gemini_model:
                # Gemini-only mode
                persian_text = self.translate_with_gemini(subtitle_text)
                if persian_text:
                    gemini_successes += 1
            
            # Create translated subtitle
            translated_subtitle = subtitle.copy()
            if persian_text:
                translated_subtitle['text'] = persian_text
                successful_translations += 1
            else:
                # Keep original text if translation fails (better than showing error message)
                translated_subtitle['text'] = subtitle_text
                logger.warning(f"⚠️  Translation failed for subtitle {i+1}: {subtitle_text[:50]}...")
            
            translated_subtitles.append(translated_subtitle)
            time.sleep(0.1)  # Small delay to be gentle on APIs
        
        # Show translation summary
        print(f"\n📈 TRANSLATION SUMMARY:")
        print(f"  ✅ Total subtitles: {len(subtitles)}")
        print(f"  ✅ Successful: {successful_translations}")
        if ollama_successes > 0:
            print(f"  🤖 Ollama translations: {ollama_successes}")
        if gemini_successes > 0:
            print(f"  🌟 Gemini translations: {gemini_successes}")
        print(f"  ❌ Failed: {len(subtitles) - successful_translations}")
        
        return translated_subtitles

    def format_srt_time(self, seconds: float) -> str:
        """Format seconds to SRT time format"""
        # Handle edge cases
        if seconds < 0:
            seconds = 0
        if seconds > 359999.999:  # Max SRT time (99:59:59,999)
            seconds = 359999.999
        
        # Ensure seconds is a valid number
        try:
            seconds = float(seconds)
        except (ValueError, TypeError):
            seconds = 0.0
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds - int(seconds)) * 1000)
        
        # Ensure millisecs is in valid range
        millisecs = max(0, min(999, millisecs))
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

    def save_srt_file(self, subtitles: List[Dict], output_path: str):
        """Save subtitles to SRT file (cross-platform)"""
        # Normalize output path for cross-platform compatibility
        output_path = str(Path(output_path).resolve())
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Write with UTF-8 encoding and BOM for Windows compatibility
            with open(output_path, 'w', encoding='utf-8-sig') as f:
                for sub in sorted(subtitles, key=lambda x: x['index']):
                    f.write(f"{sub['index']}\n")
                    f.write(f"{self.format_srt_time(sub['start'])} --> {self.format_srt_time(sub['end'])}\n")
                    f.write(f"{sub['text']}\n\n")
            print(f"💾 Persian subtitles saved: {output_path}")
        except PermissionError:
            raise Exception(f"❌ Permission denied: Cannot write to {output_path}\n"
                          f"   Make sure you have write permissions for this location.")
        except Exception as e:
            raise Exception(f"❌ Failed to save subtitle file: {e}")

    def process_video(self, video_path: str, output_path: str):
        """Complete translation pipeline"""
        print("\n🚀 STARTING TRANSLATION PROCESS")
        print("=" * 50)
        
        audio_path = self.extract_audio(video_path)
        
        try:
            english_subtitles = self.transcribe_audio(audio_path)
            persian_subtitles = self.translate_subtitles(english_subtitles)
            self.save_srt_file(persian_subtitles, output_path)
            
            print("\n🎉 SUCCESS! Translation completed successfully!")
            print(f"📄 Output file: {output_path}")
            print("=" * 50)
            
        finally:
            # Clean up temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
                print("🧹 Temporary files cleaned up")

def main():
    """Main interactive function"""
    # Show logo and introduction
    show_logo()
    show_features()
    show_requirements()
    
    # Show CUDA status and installation guide
    show_cuda_info()
    
    print("\n🎯 Let's get started with your subtitle translation!")
    
    # Get translation method preference
    method = get_user_choice(
        "🔧 Choose your preferred translation method:",
        [
            "Ollama only (Local, Private, Free)", 
            "Ollama + Gemini fallback (Best quality)",
            "Gemini only (Google AI, No Ollama needed)"
        ],
        default=0
    )
    
    # Configure based on method
    ollama_model = None
    gemini_api_key = None
    
    if "Gemini only" in method:
        # Gemini-only mode - skip Ollama configuration
        print("\n" + "="*50)
        print("🌟 GEMINI-ONLY MODE")
        print("="*50)
        print("Skipping Ollama configuration - using Gemini API only")
        
        gemini_api_key = get_gemini_config_required()
        if not gemini_api_key:
            print("❌ Cannot proceed without Gemini API key")
            sys.exit(1)
    else:
        # Configure Ollama
        print("\n" + "="*50)
        print("🤖 OLLAMA CONFIGURATION")
        print("="*50)
        
        ollama_model = get_ollama_model()
        if not ollama_model:
            print("❌ Cannot proceed without a working Ollama model")
            sys.exit(1)
        
        # Configure Gemini if requested
        if "Gemini" in method:
            gemini_api_key = get_gemini_config()
    
    # Get Whisper configuration
    whisper_model, force_cpu = get_whisper_config()
    
    # Get video file path
    print("\n" + "="*50)
    print("📁 FILE SELECTION")
    print("="*50)
    
    video_path = get_video_path()
    output_path = get_output_path(video_path)
    
    # Confirm settings
    print("\n" + "="*50)
    print("⚙️  CONFIGURATION SUMMARY")
    print("="*50)
    print(f"📹 Video file: {video_path}")
    print(f"💾 Output file: {output_path}")
    if ollama_model:
        print(f"🤖 Ollama model: {ollama_model}")
    if gemini_api_key:
        print(f"🌟 Gemini API: Configured")
    if whisper_model:
        print(f"🎤 Whisper model: {whisper_model}")
    else:
        print(f"🎤 Whisper model: Auto-select (based on GPU memory)")
    if force_cpu:
        print(f"💻 Whisper device: CPU (forced)")
    elif torch.cuda.is_available():
        print(f"💻 Whisper device: GPU (auto)")
    else:
        print(f"💻 Whisper device: CPU (no GPU)")
    if not ollama_model and not gemini_api_key:
        print("❌ No translation service configured!")
        sys.exit(1)
    
    confirm = input("\n✅ Start translation? (Y/n): ").strip().lower()
    if confirm == 'n':
        print("👋 Translation cancelled. Goodbye!")
        sys.exit(0)
    
    # Start translation process
    try:
        translator = PersianSubtitleTranslator(
            ollama_model=ollama_model,
            gemini_api_key=gemini_api_key,
            whisper_model=whisper_model,
            force_cpu=force_cpu
        )
        translator.process_video(video_path, output_path)
        
        print("\n🎊 All done! Your Persian subtitles are ready!")
        print("💡 You can now use the .srt file with any video player")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Translation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during translation: {e}")
        logger.error(f"Translation error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()