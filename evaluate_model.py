"""
Evaluation script for companion bot.
Compares model performance before and after fine-tuning.

Metrics:
- Perplexity (lower is better)
- BLEU score (higher is better)
- Emotion-appropriate response rate (custom metric)
- Response quality (qualitative examples)
"""

import argparse
import json
import math
import torch
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_test_data(json_path: str, max_samples: int = None) -> List[Dict]:
    """
    Load test data from JSON file.
    Expected format: List of dicts with "text" field containing:
    "[emotion]\nContext:...\nUser:...\nChatbot:..."
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    return data


def parse_training_format(text: str) -> Dict[str, str]:
    """
    Parse training data format: "[emotion]\nContext:...\nUser:...\nChatbot:..."
    
    Returns:
        dict with keys: emotion, context, user_input, target_response
    """
    lines = text.strip().split('\n')
    
    # Extract emotion (first line, in brackets)
    emotion = "neutral"
    if lines and lines[0].startswith('[') and lines[0].endswith(']'):
        emotion = lines[0][1:-1].strip()
        lines = lines[1:]
    
    # Find Context, User, and Chatbot lines.
    # We specifically want the *last* chatbot turn as the target,
    # and the last user turn before that as the user_input.
    context = ""
    user_input = ""
    target_response = ""
    
    user_indices = []
    chatbot_indices = []
    
    for i, line in enumerate(lines):
        if line.startswith("Context:") and not context:
            context = line.replace("Context:", "").strip()
        elif line.startswith("User:"):
            user_indices.append(i)
        elif line.startswith("Chatbot:"):
            chatbot_indices.append(i)
    
    # Determine last chatbot segment (if any)
    if chatbot_indices:
        last_chatbot_idx = chatbot_indices[-1]
        # Target response is everything from the last "Chatbot:" onward
        first_line = lines[last_chatbot_idx].replace("Chatbot:", "").strip()
        if last_chatbot_idx + 1 < len(lines):
            target_response = "\n".join([first_line] + lines[last_chatbot_idx + 1 :]).strip()
        else:
            target_response = first_line
        
        # User input: last "User:" line that appears before this chatbot line
        prior_users = [idx for idx in user_indices if idx < last_chatbot_idx]
        if prior_users:
            last_user_idx = prior_users[-1]
            user_input = lines[last_user_idx].replace("User:", "").strip()
    else:
        # Fallback: no Chatbot line; use last User as input if present
        if user_indices:
            last_user_idx = user_indices[-1]
            user_input = lines[last_user_idx].replace("User:", "").strip()
    
    return {
        "emotion": emotion,
        "context": context,
        "user_input": user_input,
        "target_response": target_response
    }


def compute_perplexity(model, tokenizer, texts: List[str], device: torch.device) -> float:
    """
    Compute perplexity on a list of texts.
    Lower perplexity = better language model quality.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            labels = inputs["input_ids"].clone()
            
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss.item()
            n_tokens = labels.numel()
            
            total_loss += loss * n_tokens
            total_tokens += n_tokens
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity


def compute_bleu(predictions: List[str], references: List[str]) -> float:
    """
    Compute BLEU score using sacrebleu.
    Higher BLEU = better generation quality.
    """
    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(predictions, [references])
        return bleu.score
    except ImportError:
        print("Warning: sacrebleu not installed. Install with: pip install sacrebleu")
        return 0.0


def generate_responses(
    model,
    tokenizer,
    test_data: List[Dict],
    device: torch.device,
    use_emotion_labels: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Generate responses for test data.
    
    Returns:
        (predictions, references, detected_emotions)
    """
    predictions: List[str] = []
    references: List[str] = []
    
    for item in test_data:
        parsed = parse_training_format(item["text"])
        user_input = parsed["user_input"]
        target_response = parsed["target_response"]
        emotion_label = parsed["emotion"]
        
        # Build evaluation prompt directly from dataset fields.
        if use_emotion_labels and emotion_label and emotion_label != "neutral":
            # Match training format, but using ground-truth emotion from the dataset.
            prompt = f"[{emotion_label}]\nContext: {parsed['context']}\nUser: {user_input}\nChatbot:"
        else:
            # Ignore emotion label; simple context + user prompt.
            prompt = f"Context: {parsed['context']}\nUser: {user_input}\nChatbot:"
        
        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        input_length = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # For causal LMs, drop the prompt tokens and decode only the continuation.
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Keep only the first chatbot turn: cut off anything after a new "User:" or
        # another "Chatbot:" marker that may appear in the continuation.
        if "User:" in response:
            response = response.split("User:")[0].strip()
        if "Chatbot:" in response:
            response = response.split("Chatbot:")[0].strip()

        predictions.append(response)
        references.append(target_response)
    
    return predictions, references


def main():
    parser = argparse.ArgumentParser(description="Evaluate companion bot model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model (base model or fine-tuned checkpoint)"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to LoRA adapter (if using LoRA fine-tuning)"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to test JSON file"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Maximum number of samples to evaluate"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--use_emotion",
        action="store_true",
        help="Use emotion detection (matches training format)"
    )
    
    args = parser.parse_args()
    
    # Load device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    
    if args.lora_path:
        # Match test_finetune.py: tokenizer from LoRA path, base model from BASE_MODEL
        print(f"Loading tokenizer from LoRA path: {args.lora_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.lora_path)
        
        print(f"Loading base model for LoRA from: {args.model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
        )
        print(f"Loading LoRA adapter from: {args.lora_path}")
        model = PeftModel.from_pretrained(base_model, args.lora_path)
    else:
        # Base model only (no LoRA)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    print("Model loaded!")
    
    # Load test data
    print(f"Loading test data from: {args.test_data}")
    test_data = load_test_data(args.test_data, max_samples=args.max_samples)
    print(f"Loaded {len(test_data)} test samples")
    
    # Parse test data
    parsed_data = [parse_training_format(item["text"]) for item in test_data]
    
    # 1. Compute Perplexity
    print("\n" + "="*60)
    print("Computing Perplexity...")
    print("="*60)
    full_texts = [item["text"] for item in test_data]
    perplexity = compute_perplexity(model, tokenizer, full_texts, device)
    print(f"Perplexity: {perplexity:.2f}")
    
    # 2. Generate responses and compute BLEU
    print("\n" + "="*60)
    print("Generating responses and computing BLEU...")
    print("="*60)
    predictions, references = generate_responses(
        model,
        tokenizer,
        test_data,
        device,
        use_emotion_labels=args.use_emotion
    )
    bleu_score = compute_bleu(predictions, references)
    print(f"BLEU Score: {bleu_score:.2f}")
    
    # 3. Show sample outputs
    print("\n" + "="*60)
    print("Sample Outputs (first 3):")
    print("="*60)
    for i in range(min(3, len(parsed_data))):
        print(f"\nSample {i+1}:")
        print(f"  Emotion (GT): {parsed_data[i]['emotion']}")
        print(f"  User Input: {parsed_data[i]['user_input']}")
        print(f"  Target: {parsed_data[i]['target_response']}")
        print(f"  Generated: {predictions[i]}")
        print("-" * 60)
    
    # Compile results
    results = {
        "model_path": args.model_path,
        "lora_path": args.lora_path,
        "num_samples": len(test_data),
        "perplexity": perplexity,
        "bleu_score": bleu_score,
    }
    
    # Save results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Perplexity: {perplexity:.2f} (lower is better)")
    print(f"BLEU Score: {bleu_score:.2f} (higher is better)")
    print("="*60)
    
    return results


if __name__ == "__main__":
    main()

