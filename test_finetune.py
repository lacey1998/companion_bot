import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Path
BASE_MODEL = "/projects/project1/laceytt/companion_bot/models/opt-1.3b"
LORA_PATH = "/projects/project1/laceytt/companion_bot/checkpoints/opt-1.3b-empathetic-lora"

# Load the model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()
print("Model loaded!")

# test function
def chat(user_input, emotion="sad"):
    prompt = f"[{emotion}]\nContext: {user_input}\nUser: {user_input}\nChatbot:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # get Chatbot part
    if "Chatbot:" in response:
        response = response.split("Chatbot:")[-1].strip()
    if "User:" in response:
        response = response.split("User:")[0].strip()
    return response

# test
print("\======start test======\n")

test_cases = [
    ("I just lost my job today.", "sad"),
    ("I got promoted at work!", "joyful"),
    ("I'm feeling really anxious about my exam.", "anxious"),
]

for text, emotion in test_cases:
    print(f"[{emotion}] User: {text}")
    response = chat(text, emotion)
    print(f"Chatbot: {response}")
    print("-" * 50)