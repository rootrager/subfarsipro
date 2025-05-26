#!/usr/bin/env python3
"""
Persian Video Subtitle Translator - Professional Edition
A professional tool for translating video subtitles to Persian using Whisper + Ollama/Gemini
"""

import os
import sys
import subprocess
from pathlib import Path
import re
from typing import List, Dict, Optional
import logging
import time

try:
    import whisper
    import google.generativeai as genai
    import requests
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("📦 Please install: pip install git+https://github.com/openai/whisper.git google-generativeai requests")
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
    
def show_requirements():
    """Display system requirements"""
    print("\n📋 SYSTEM REQUIREMENTS:")
    print("  • FFmpeg installed (for audio extraction)")
    print("  • Python 3.7+ with required packages")
    print("  • Ollama running locally (recommended) OR Gemini API key")
    print("  • Sufficient disk space for temporary audio files")

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
    """Get and validate video file path"""
    while True:
        print("\n📁 VIDEO FILE SELECTION:")
        video_path = input("Enter the path to your video file: ").strip().strip('"\'')
        
        if not video_path:
            print("❌ Please enter a video file path")
            continue
            
        if not os.path.exists(video_path):
            print(f"❌ File not found: {video_path}")
            print("💡 Tip: You can drag and drop the file into the terminal")
            continue
            
        # Check if it's likely a video file
        video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v']
        if not any(video_path.lower().endswith(ext) for ext in video_extensions):
            confirm = input("⚠️  This doesn't look like a video file. Continue anyway? (y/N): ").strip().lower()
            if confirm != 'y':
                continue
                
        return video_path

def get_output_path(video_path: str) -> str:
    """Get output subtitle file path"""
    default_output = str(Path(video_path).with_suffix('.persian.srt'))
    print(f"\n💾 OUTPUT FILE:")
    print(f"Default output: {default_output}")
    
    custom_path = input("Enter custom output path (or press Enter for default): ").strip().strip('"\'')
    
    if custom_path:
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

class PersianSubtitleTranslator:
    def __init__(self, ollama_model: str = None, gemini_api_key: str = None, 
                 ollama_url: str = "http://localhost:11434/api/generate", timing_offset: float = 0.5):
        """Initialize the professional Persian subtitle translator"""
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
        
        # Load Whisper model
        print("\n🎤 Loading Whisper AI model...")
        self.whisper_model = whisper.load_model("base")
        print("✅ Whisper model loaded successfully!")
        
    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video file using FFmpeg"""
        audio_path = str(Path(video_path).with_suffix('.wav'))
        print(f"\n🎵 Extracting audio from video...")
        
        try:
            cmd = [
                'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1', '-y', audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ Audio extracted successfully!")
            return audio_path
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            raise Exception("❌ Failed to extract audio. Make sure FFmpeg is installed.")
        except FileNotFoundError:
            raise Exception("❌ FFmpeg not found. Please install FFmpeg first.")
        
    def transcribe_audio(self, audio_path: str) -> List[Dict]:
        """Convert audio to English subtitles using Whisper"""
        print("\n📝 Transcribing audio to English subtitles...")
        result = self.whisper_model.transcribe(audio_path)
        
        subtitles = []
        for i, segment in enumerate(result['segments']):
            adjusted_start = max(0, segment['start'] + self.timing_offset)
            adjusted_end = max(adjusted_start, segment['end'] + self.timing_offset)
            subtitle = {
                'index': i + 1,
                'start': adjusted_start,
                'end': adjusted_end,
                'text': segment['text'].strip()
            }
            subtitles.append(subtitle)
        
        print(f"✅ Generated {len(subtitles)} subtitle segments")
        return subtitles

    def translate_with_ollama(self, text: str) -> Optional[str]:
        """Translate text using Ollama"""
        prompt = f"""Translate this English text to natural Persian (Farsi). 
Make it conversational and suitable for video subtitles. 
Return only the Persian translation, nothing else.

English: {text}

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
            result = response.json()
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
            
        prompt = f"""Translate this English text to natural Persian (Farsi).
Make it conversational and suitable for video subtitles.
Return only the Persian translation, nothing else.

English: {text}

Persian:"""

        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini translation failed: {e}")
            return None

    def translate_subtitles(self, subtitles: List[Dict]) -> List[Dict]:
        """Translate all subtitles to Persian"""
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
            # Progress indicator
            if i % 10 == 0 or i == len(subtitles) - 1:
                progress = ((i + 1) / len(subtitles)) * 100
                print(f"📊 Progress: {progress:.1f}% ({i+1}/{len(subtitles)})")
            
            persian_text = None
            
            # Translation logic based on available services
            if self.ollama_model:
                # Try Ollama first (if available)
                persian_text = self.translate_with_ollama(subtitle['text'])
                if persian_text:
                    ollama_successes += 1
                elif self.gemini_model:
                    # Fallback to Gemini
                    persian_text = self.translate_with_gemini(subtitle['text'])
                    if persian_text:
                        gemini_successes += 1
            elif self.gemini_model:
                # Gemini-only mode
                persian_text = self.translate_with_gemini(subtitle['text'])
                if persian_text:
                    gemini_successes += 1
            
            # Create translated subtitle
            translated_subtitle = subtitle.copy()
            if persian_text:
                translated_subtitle['text'] = persian_text
                successful_translations += 1
            else:
                translated_subtitle['text'] = f"[Translation Failed: {subtitle['text']}]"
            
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
        if seconds < 0:
            seconds = 0
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

    def save_srt_file(self, subtitles: List[Dict], output_path: str):
        """Save subtitles to SRT file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for sub in sorted(subtitles, key=lambda x: x['index']):
                f.write(f"{sub['index']}\n")
                f.write(f"{self.format_srt_time(sub['start'])} --> {self.format_srt_time(sub['end'])}\n")
                f.write(f"{sub['text']}\n\n")
        print(f"💾 Persian subtitles saved: {output_path}")

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
            gemini_api_key=gemini_api_key
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