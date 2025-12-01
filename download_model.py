"""
Script to download and save the model to a local directory.
This allows the model to be stored in the project directory instead of the HuggingFace cache.
Configuration is loaded from config.yaml.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import os
import yaml

def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def download_model_to_directory(model_name: str, local_dir: str, model_type: str = "causal"):
    """
    Download model and tokenizer to a local directory.
    
    Args:
        model_name: HuggingFace model name (e.g., "EleutherAI/gpt-neo-1.3B")
        local_dir: Local directory to save the model
        model_type: Type of model - "causal" or "seq2seq"
    """
    print("=" * 60)
    print(f"Downloading model: {model_name}")
    print(f"Model type: {model_type}")
    print(f"Save location: {local_dir}")
    print("=" * 60)
    
    # Create directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    # Download tokenizer
    print("\n1. Downloading tokenizer...")
    print("   This may take a few minutes...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set pad_token if it doesn't exist (needed for GPT-Neo)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(local_dir)
        print(f"   ✓ Tokenizer saved to {local_dir}")
    except Exception as e:
        print(f"   ✗ Error downloading tokenizer: {e}")
        raise
    
    # Download model based on type
    print("\n2. Downloading model...")
    print("   This may take several minutes...")
    try:
        if model_type == "causal":
            model = AutoModelForCausalLM.from_pretrained(model_name)
        elif model_type == "seq2seq":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Must be 'causal' or 'seq2seq'")
        
        model.save_pretrained(local_dir)
        print(f"   ✓ Model saved to {local_dir}")
    except Exception as e:
        print(f"   ✗ Error downloading model: {e}")
        raise
    
    print("\n" + "=" * 60)
    print("✓ Model download complete!")
    print(f"✓ Model saved to: {local_dir}")
    print("=" * 60)
    print("\nYou can now use the model with companion_bot.py")


if __name__ == "__main__":
    try:
        # Load configuration from YAML
        config = load_config()
        download_config = config.get("download", {})
        
        model_name = download_config.get("model_name", "EleutherAI/gpt-neo-1.3B")
        local_dir = download_config.get("local_dir", "./models/gpt-neo-1.3B")
        model_type = config.get("model", {}).get("type", "causal")
        
        download_model_to_directory(model_name, local_dir, model_type)
        print("\n Success! The model is now stored locally.")
        print(f"   You can find it in: {os.path.abspath(local_dir)}")
    except FileNotFoundError as e:
        print(f"\n Error: {e}")
        print("Please create a config.yaml file with model configuration.")
    except Exception as e:
        print(f"\n Error: {e}")
        print("Make sure you have:")
        print("  1. Internet connection")
        print("  2. transformers library installed: pip install transformers torch")
        print("  3. PyYAML installed: pip install pyyaml")
        print("  4. Enough disk space (~5GB for GPT-Neo-1.3B)")

