"""
Quick test to verify if the model can be loaded and run.
This is a minimal test script.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

print("=" * 60)
print("QUICK MODEL TEST")
print("=" * 60)

# Check if transformers is installed
try:
    print("✓ transformers library is installed")
except ImportError:
    print("✗ transformers library not found. Run: pip install transformers torch")
    exit(1)

# Configuration
LOCAL_MODEL_DIR = "./models/blenderbot-400M-distill"
MODEL_NAME = "facebook/blenderbot-400M-distill"

# Check if local model exists
use_local = os.path.exists(LOCAL_MODEL_DIR) and os.path.isdir(LOCAL_MODEL_DIR)

if use_local:
    print(f"\n✓ Found local model directory: {LOCAL_MODEL_DIR}")
    model_source = LOCAL_MODEL_DIR
    source_type = "local directory"
else:
    print(f"\n⚠ Local model not found at: {LOCAL_MODEL_DIR}")
    print(f"  Will load from HuggingFace: {MODEL_NAME}")
    model_source = MODEL_NAME
    source_type = "HuggingFace (will download if not cached)"

print(f"\n1. Loading tokenizer from: {source_type}")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    print(f"   ✓ Tokenizer loaded successfully from {source_type}")
except Exception as e:
    print(f"   ✗ Error loading tokenizer: {e}")
    if not use_local:
        print("   Make sure you have internet connection for first download")
    exit(1)

print(f"\n2. Loading model from: {source_type}")
try:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_source)
    print(f"   ✓ Model loaded successfully from {source_type}")
except Exception as e:
    print(f"   ✗ Error loading model: {e}")
    if not use_local:
        print("   Make sure you have internet connection for first download")
    else:
        print(f"   Local model may be corrupted. Try running: python download_model.py")
    exit(1)

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n3. Device detection: {device}")
model = model.to(device)
print(f"   ✓ Model moved to {device}")

# Quick generation test
print(f"\n4. Testing generation...")
test_input = "Hello, how are you?"
print(f"   Input: '{test_input}'")

try:
    inputs = tokenizer(test_input, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   ✓ Generated response: {response}")
    print("\n" + "=" * 60)
    print("🎉 SUCCESS! The model is working correctly on your laptop.")
    print("=" * 60)
    print(f"\nYou can now run the full bot with: python companion_bot.py")
    
except Exception as e:
    print(f"   ✗ Error during generation: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

