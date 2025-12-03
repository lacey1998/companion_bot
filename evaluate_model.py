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
from emotion_detector import EmotionDetector
import sys
import os

# Add current directory to path to import companion_bot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from companion_bot import CompanionBot


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
    
    # Find User and Chatbot lines
    user_input = ""
    target_response = ""
    context = ""
    
    for i, line in enumerate(lines):
        if line.startswith("Context:"):
            context = line.replace("Context:", "").strip()
        elif line.startswith("User:"):
            user_input = line.replace("User:", "").strip()
        elif line.startswith("Chatbot:"):
            target_response = line.replace("Chatbot:", "").strip()
            # Everything after Chatbot: is the response
            if i + 1 < len(lines):
                target_response = '\n'.join([target_response] + lines[i+1:]).strip()
            break
    
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


def evaluate_emotion_appropriateness(
    bot: CompanionBot,
    test_data: List[Dict],
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate if responses are appropriate for detected emotions.
    
    Returns:
        dict with metrics:
        - emotion_detection_accuracy: How often emotion matches ground truth
        - appropriate_response_rate: Subjective measure (placeholder for now)
    """
    emotion_detector = EmotionDetector()
    correct_detections = 0
    total = 0
    
    for item in test_data:
        parsed = parse_training_format(item["text"])
        ground_truth_emotion = parsed["emotion"]
        user_input = parsed["user_input"]
        
        # Detect emotion
        detected_emotion = emotion_detector.detect_emotion(user_input)
        
        if detected_emotion == ground_truth_emotion:
            correct_detections += 1
        total += 1
    
    accuracy = correct_detections / total if total > 0 else 0.0
    
    return {
        "emotion_detection_accuracy": accuracy,
        "correct_detections": correct_detections,
        "total_samples": total
    }


def generate_responses(
    bot: CompanionBot,
    test_data: List[Dict],
    device: torch.device
) -> Tuple[List[str], List[str], List[str]]:
    """
    Generate responses for test data.
    
    Returns:
        (predictions, references, detected_emotions)
    """
    predictions = []
    references = []
    detected_emotions = []
    
    for item in test_data:
        parsed = parse_training_format(item["text"])
        user_input = parsed["user_input"]
        target_response = parsed["target_response"]
        
        # Generate response
        response = bot.chat(user_input, debug=False)
        
        # Get detected emotion
        detected_emotion = bot.get_last_detected_emotion() or "neutral"
        
        predictions.append(response)
        references.append(target_response)
        detected_emotions.append(detected_emotion)
    
    return predictions, references, detected_emotions


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
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    if args.lora_path:
        # Load base model + LoRA adapter
        print(f"Loading LoRA adapter from: {args.lora_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None
        )
        model = PeftModel.from_pretrained(base_model, args.lora_path)
    else:
        # Load base model only
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    print("Model loaded!")
    
    # Initialize bot
    bot = CompanionBot(
        model=model,
        tokenizer=tokenizer,
        model_type="causal",
        use_emotion_detection=args.use_emotion
    )
    
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
    predictions, references, detected_emotions = generate_responses(bot, test_data, device)
    bleu_score = compute_bleu(predictions, references)
    print(f"BLEU Score: {bleu_score:.2f}")
    
    # 3. Emotion detection accuracy
    if args.use_emotion:
        print("\n" + "="*60)
        print("Evaluating Emotion Detection...")
        print("="*60)
        emotion_metrics = evaluate_emotion_appropriateness(bot, test_data, device)
        print(f"Emotion Detection Accuracy: {emotion_metrics['emotion_detection_accuracy']:.2%}")
        print(f"Correct: {emotion_metrics['correct_detections']}/{emotion_metrics['total_samples']}")
    
    # 4. Show sample outputs
    print("\n" + "="*60)
    print("Sample Outputs (first 3):")
    print("="*60)
    for i in range(min(3, len(parsed_data))):
        print(f"\nSample {i+1}:")
        print(f"  Emotion (GT): {parsed_data[i]['emotion']}")
        if args.use_emotion:
            print(f"  Emotion (Detected): {detected_emotions[i]}")
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
    
    if args.use_emotion:
        results["emotion_detection_accuracy"] = emotion_metrics['emotion_detection_accuracy']
        results["emotion_detections"] = {
            "correct": emotion_metrics['correct_detections'],
            "total": emotion_metrics['total_samples']
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
    if args.use_emotion:
        print(f"Emotion Detection Accuracy: {emotion_metrics['emotion_detection_accuracy']:.2%}")
    print("="*60)
    
    return results


if __name__ == "__main__":
    main()

