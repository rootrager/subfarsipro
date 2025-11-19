#!/usr/bin/env python3
"""
DeepSeek AI Test Script
A comprehensive testing tool for DeepSeek AI API
"""

import os
import sys
import time
import json
from typing import Dict, Optional, List
from datetime import datetime

try:
    import requests
except ImportError:
    print("❌ Missing required package: requests")
    print("📦 Please install: pip install requests")
    sys.exit(1)

# DeepSeek API Configuration
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"

class DeepSeekTester:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize DeepSeek AI tester"""
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            print("⚠️  No API key provided. Set DEEPSEEK_API_KEY environment variable or pass it as argument.")
            print("💡 You can get your API key from: https://platform.deepseek.com/")
        
        self.base_url = f"{DEEPSEEK_API_BASE}/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        }
        
    def test_connection(self) -> bool:
        """Test basic API connection"""
        print("\n🔍 Testing DeepSeek API Connection...")
        print("=" * 60)
        
        if not self.api_key:
            print("❌ No API key available. Cannot test connection.")
            return False
        
        test_prompt = "Say 'Hello, DeepSeek is working!' in one sentence."
        
        try:
            payload = {
                "model": DEFAULT_MODEL,
                "messages": [
                    {"role": "user", "content": test_prompt}
                ],
                "max_tokens": 50,
                "temperature": 0.7
            }
            
            start_time = time.time()
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                message = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                print(f"✅ Connection successful!")
                print(f"⏱️  Response time: {elapsed_time:.2f} seconds")
                print(f"📝 Response: {message.strip()}")
                return True
            else:
                print(f"❌ Connection failed!")
                print(f"Status code: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            print("❌ Connection timeout. API may be slow or unreachable.")
            return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
    
    def test_text_generation(self, prompt: str, max_tokens: int = 200) -> Optional[Dict]:
        """Test text generation with a custom prompt"""
        print(f"\n📝 Testing Text Generation...")
        print(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        print("-" * 60)
        
        if not self.api_key:
            print("❌ No API key available.")
            return None
        
        try:
            payload = {
                "model": DEFAULT_MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            
            start_time = time.time()
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                message = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                usage = result.get("usage", {})
                
                print(f"✅ Generation successful!")
                print(f"⏱️  Response time: {elapsed_time:.2f} seconds")
                print(f"📊 Tokens used: {usage.get('total_tokens', 'N/A')}")
                print(f"   - Prompt: {usage.get('prompt_tokens', 'N/A')}")
                print(f"   - Completion: {usage.get('completion_tokens', 'N/A')}")
                print(f"\n📝 Response:\n{message.strip()}\n")
                
                return {
                    "success": True,
                    "response": message.strip(),
                    "time": elapsed_time,
                    "usage": usage
                }
            else:
                print(f"❌ Generation failed!")
                print(f"Status code: {response.status_code}")
                print(f"Response: {response.text}")
                return {"success": False, "error": response.text}
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return {"success": False, "error": str(e)}
    
    def test_translation(self, text: str, target_language: str = "Persian") -> Optional[Dict]:
        """Test translation capability"""
        print(f"\n🌍 Testing Translation to {target_language}...")
        print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print("-" * 60)
        
        prompt = f"Translate the following English text to {target_language}. Return only the translation, nothing else.\n\nEnglish: {text}\n\n{target_language}:"
        
        return self.test_text_generation(prompt, max_tokens=300)
    
    def test_code_generation(self, task: str) -> Optional[Dict]:
        """Test code generation capability"""
        print(f"\n💻 Testing Code Generation...")
        print(f"Task: {task[:100]}{'...' if len(task) > 100 else ''}")
        print("-" * 60)
        
        prompt = f"Write a Python function to {task}. Include comments and make it production-ready."
        
        return self.test_text_generation(prompt, max_tokens=500)
    
    def test_conversation(self, messages: List[Dict]) -> Optional[Dict]:
        """Test multi-turn conversation"""
        print(f"\n💬 Testing Multi-turn Conversation...")
        print(f"Messages: {len(messages)} turns")
        print("-" * 60)
        
        if not self.api_key:
            print("❌ No API key available.")
            return None
        
        try:
            payload = {
                "model": DEFAULT_MODEL,
                "messages": messages,
                "max_tokens": 300,
                "temperature": 0.7
            }
            
            start_time = time.time()
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                message = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                usage = result.get("usage", {})
                
                print(f"✅ Conversation successful!")
                print(f"⏱️  Response time: {elapsed_time:.2f} seconds")
                print(f"📊 Tokens used: {usage.get('total_tokens', 'N/A')}")
                print(f"\n📝 Response:\n{message.strip()}\n")
                
                return {
                    "success": True,
                    "response": message.strip(),
                    "time": elapsed_time,
                    "usage": usage
                }
            else:
                print(f"❌ Conversation failed!")
                print(f"Status code: {response.status_code}")
                return {"success": False, "error": response.text}
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return {"success": False, "error": str(e)}
    
    def run_comprehensive_test(self):
        """Run a comprehensive test suite"""
        print("\n" + "=" * 60)
        print("🚀 DEEPSEEK AI COMPREHENSIVE TEST SUITE")
        print("=" * 60)
        print(f"📅 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔑 API Key: {'✅ Set' if self.api_key else '❌ Not set'}")
        print(f"🌐 API Endpoint: {self.base_url}")
        print(f"🤖 Model: {DEFAULT_MODEL}")
        
        results = {
            "connection": False,
            "text_generation": False,
            "translation": False,
            "code_generation": False,
            "conversation": False
        }
        
        # Test 1: Connection
        results["connection"] = self.test_connection()
        
        if not results["connection"]:
            print("\n❌ Cannot proceed with other tests. Please check your API key and connection.")
            return results
        
        # Test 2: Text Generation
        result = self.test_text_generation(
            "Explain quantum computing in simple terms for a beginner.",
            max_tokens=300
        )
        results["text_generation"] = result.get("success", False) if result else False
        
        # Test 3: Translation
        result = self.test_translation(
            "Hello, how are you today? I hope you're having a wonderful day!",
            "Persian"
        )
        results["translation"] = result.get("success", False) if result else False
        
        # Test 4: Code Generation
        result = self.test_code_generation("calculate the factorial of a number")
        results["code_generation"] = result.get("success", False) if result else False
        
        # Test 5: Conversation
        conversation_messages = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a high-level programming language known for its simplicity and readability."},
            {"role": "user", "content": "What are its main advantages?"}
        ]
        result = self.test_conversation(conversation_messages)
        results["conversation"] = result.get("success", False) if result else False
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        for test_name, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"  {test_name.replace('_', ' ').title()}: {status}")
        
        total_tests = len(results)
        passed_tests = sum(results.values())
        print(f"\n📈 Results: {passed_tests}/{total_tests} tests passed")
        print(f"📅 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return results

def main():
    """Main function"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║              🧪 DEEPSEEK AI TEST SCRIPT 🧪                     ║
║                                                                ║
║        Comprehensive testing tool for DeepSeek AI API         ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Get API key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("💡 No DEEPSEEK_API_KEY environment variable found.")
        api_key_input = input("Enter your DeepSeek API key (or press Enter to skip): ").strip()
        if api_key_input:
            api_key = api_key_input
        else:
            print("⚠️  Continuing without API key (some tests will be skipped)")
    
    tester = DeepSeekTester(api_key)
    
    # Menu
    while True:
        print("\n" + "=" * 60)
        print("📋 TEST MENU")
        print("=" * 60)
        print("1. Test API Connection")
        print("2. Test Text Generation")
        print("3. Test Translation")
        print("4. Test Code Generation")
        print("5. Test Multi-turn Conversation")
        print("6. Run Comprehensive Test Suite")
        print("7. Exit")
        
        choice = input("\nSelect an option (1-7): ").strip()
        
        if choice == "1":
            tester.test_connection()
        elif choice == "2":
            prompt = input("Enter your prompt: ").strip()
            if prompt:
                tester.test_text_generation(prompt)
            else:
                print("❌ Prompt cannot be empty")
        elif choice == "3":
            text = input("Enter text to translate: ").strip()
            if text:
                target = input("Target language (default: Persian): ").strip() or "Persian"
                tester.test_translation(text, target)
            else:
                print("❌ Text cannot be empty")
        elif choice == "4":
            task = input("Enter coding task: ").strip()
            if task:
                tester.test_code_generation(task)
            else:
                print("❌ Task cannot be empty")
        elif choice == "5":
            print("Enter conversation messages (type 'done' when finished):")
            messages = []
            while True:
                role = input("Role (user/assistant/system): ").strip().lower()
                if role == "done":
                    break
                if role not in ["user", "assistant", "system"]:
                    print("❌ Role must be 'user', 'assistant', or 'system'")
                    continue
                content = input("Content: ").strip()
                if content:
                    messages.append({"role": role, "content": content})
            if messages:
                tester.test_conversation(messages)
            else:
                print("❌ No messages provided")
        elif choice == "6":
            tester.run_comprehensive_test()
        elif choice == "7":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid option. Please select 1-7.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

