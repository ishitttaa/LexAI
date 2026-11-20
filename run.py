"""
Galileo-Style Hallucination Audit System — Launcher
"""
import subprocess
import sys
import os
import shutil

def check_env():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        example = os.path.join(os.path.dirname(__file__), ".env.example")
        if os.path.exists(example):
            shutil.copy(example, env_path)
        print("\n" + "="*60)
        print("  ACTION REQUIRED: API keys not configured!")
        print("="*60)
        print("\n  A .env file has been created for you.")
        print("  Please open it and fill in your API keys:")
        print(f"\n  -> {env_path}\n")
        print("  GROQ_API_KEY    : https://console.groq.com")
        print("  SERPER_API_KEY  : https://serper.dev\n")
        sys.exit(0)

    with open(env_path) as f:
        content = f.read()

    missing = []
    if "your_groq_api_key_here" in content or "GROQ_API_KEY=" not in content:
        missing.append("GROQ_API_KEY")
    if "your_serper_api_key_here" in content or "SERPER_API_KEY=" not in content:
        missing.append("SERPER_API_KEY")

    if missing:
        print("\n" + "="*60)
        print("  ACTION REQUIRED: Missing API keys in .env!")
        print("="*60)
        for key in missing:
            print(f"  [X]  {key} is not set")
        print(f"\n  Edit:  {env_path}\n")
        sys.exit(1)


def install_deps():
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    print("\nInstalling dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_path])
    print("Dependencies ready.\n")


def start_server():
    host = "127.0.0.1"
    port = 8000
    print("="*60)
    print("  Galileo-Style Hallucination Audit System")
    print("="*60)
    print(f"\n  Server running at:  http://{host}:{port}")
    print("\n  Press Ctrl+C to stop the server.\n")
    print("="*60 + "\n")

    subprocess.run([sys.executable, "-m", "uvicorn", "main:app", "--host", host, "--port", str(port), "--reload"])


if __name__ == "__main__":
    check_env()
    install_deps()
    start_server()
