"""
Script to download and save the Blenderbot model to a local directory.
This allows the model to be stored in the project directory instead of the HuggingFace cache.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

def download_model_to_directory(model_name: str, local_dir: str):
    """
    Download model and tokenizer to a local directory.
    
    Args:
        model_name: HuggingFace model name (e.g., "facebook/blenderbot-400M-distill")
        local_dir: Local directory to save the model
    """
    print("=" * 60)
    print(f"Downloading model: {model_name}")
    print(f"Save location: {local_dir}")
    print("=" * 60)
    
    # Create directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    # Download tokenizer
    print("\n1. Downloading tokenizer...")
    print("   This may take a few minutes...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(local_dir)
        print(f"   ✓ Tokenizer saved to {local_dir}")
    except Exception as e:
        print(f"   ✗ Error downloading tokenizer: {e}")
        raise
    
    # Download model
    print("\n2. Downloading model...")
    print("   This may take several minutes (~1.5GB)...")
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.save_pretrained(local_dir)
        print(f"   ✓ Model saved to {local_dir}")
    except Exception as e:
        print(f"   ✗ Error downloading model: {e}")
        raise
    
    print("\n" + "=" * 60)
    print("✓ Model download complete!")
    print(f"✓ Model saved to: {local_dir}")
    print("=" * 60)
    print("\nYou can now use the model by setting LOCAL_MODEL_DIR in companion_bot.py")


if __name__ == "__main__":
    model_name = "facebook/blenderbot-400M-distill"
    local_dir = "./models/blenderbot-400M-distill"
    
    try:
        download_model_to_directory(model_name, local_dir)
        print("\n✅ Success! The model is now stored locally.")
        print(f"   You can find it in: {os.path.abspath(local_dir)}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure you have:")
        print("  1. Internet connection")
        print("  2. transformers library installed: pip install transformers torch")
        print("  3. Enough disk space (~2GB)")

