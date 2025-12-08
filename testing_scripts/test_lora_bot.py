"""
Test script to load LoRA fine-tuned model and use it with CompanionBot for end-to-end testing.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from companion_bot import CompanionBot

# Paths - adjust these to match your cluster paths
BASE_MODEL = "/projects/project1/laceytt/companion_bot/models/opt-1.3b"
LORA_PATH = "/projects/project1/laceytt/companion_bot/code/lora-tuned/opt-1.3b-empathetic-lora-exp5-r32"

def load_lora_model():
    """Load base model with LoRA adapter."""
    print("Loading LoRA fine-tuned model...")
    print(f"Base model: {BASE_MODEL}")
    print(f"LoRA adapter: {LORA_PATH}")
    
    # Load tokenizer from LoRA path (contains fine-tuned tokenizer if needed)
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
    print("   Tokenizer loaded!")
    
    # Load base model
    print("\n2. Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        device_map="auto"
    )
    print("   Base model loaded!")
    
    # Load LoRA adapter
    print("\n3. Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()
    print("   LoRA adapter loaded!")
    
    return model, tokenizer


def test_interactive():
    """Interactive testing with CompanionBot."""
    # Load model
    model, tokenizer = load_lora_model()
    
    # Initialize CompanionBot with LoRA model
    print("\n" + "="*60)
    print("Initializing CompanionBot with LoRA fine-tuned model...")
    print("="*60)
    
    bot = CompanionBot(
        model=model,
        tokenizer=tokenizer,
        model_type="causal",
        tone="empathetic",
        use_emotion_detection=True
    )
    
    print("\n=== LoRA Fine-tuned Companion Bot is ready! ===")
    print("Commands:")
    print("  - Type your message to chat")
    print("  - 'tone <name>' to change tone (cheerful, supportive, empathetic, calm, friendly, angry)")
    print("  - 'debug on/off' to toggle debug mode")
    print("  - 'reset' to reset conversation")
    print("  - 'exit' or 'quit' to quit")
    print()
    
    debug_mode = False
    
    # Interactive chat loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            # Handle tone change
            if user_input.lower().startswith("tone "):
                new_tone = user_input[5:].strip().lower()
                valid_tones = ["cheerful", "supportive", "empathetic", "calm", "friendly", "angry"]
                if new_tone in valid_tones:
                    bot.set_tone(new_tone)
                else:
                    print(f"Invalid tone. Choose from: {', '.join(valid_tones)}")
                continue
            
            # Handle debug mode
            if user_input.lower() == "debug on":
                debug_mode = True
                print("Debug mode ON")
                continue
            elif user_input.lower() == "debug off":
                debug_mode = False
                print("Debug mode OFF")
                continue
            
            # Handle reset
            if user_input.lower() == "reset":
                bot.reset_conversation()
                continue
            
            # Normal chat
            try:
                reply = bot.chat(user_input, debug=debug_mode)
                print(f"Bot: {reply}\n")
            except Exception as e:
                print(f"Error: {e}\n")
                
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'exit' to quit.")
            continue


def test_batch():
    """Batch testing with predefined test cases."""
    # Load model
    model, tokenizer = load_lora_model()
    
    # Initialize CompanionBot
    bot = CompanionBot(
        model=model,
        tokenizer=tokenizer,
        model_type="causal",
        tone="empathetic",
        use_emotion_detection=True
    )
    
    print("\n" + "="*60)
    print("Batch Testing with LoRA Fine-tuned Model")
    print("="*60)
    
    test_cases = [
        ("I just lost my job today.", "sad"),
        ("I got promoted at work!", "joyful"),
        ("I'm feeling really anxious about my exam.", "anxious"),
        ("I'm grateful for everything I have.", "grateful"),
    ]
    
    for i, (text, expected_emotion) in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Expected Emotion: {expected_emotion}")
        print(f"User: {text}")
        
        response = bot.chat(text, debug=False)
        detected_emotion = bot.get_last_detected_emotion()
        
        print(f"Detected Emotion: {detected_emotion}")
        print(f"Bot: {response}")
        print("-" * 60)
        
        # Reset conversation between test cases
        bot.reset_conversation()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        test_batch()
    else:
        test_interactive()

