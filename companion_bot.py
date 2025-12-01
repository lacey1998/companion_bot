from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
from typing import Optional, List, Dict
import json
import os
import yaml


class CompanionBot:
    """
    A conversational agent that generates empathetic, tone-aware responses
    using a language model (supports both causal and seq2seq models).
    """
    
    def __init__(self, model, tokenizer, model_type: str = "causal", tone: str = "supportive", conversation_history: Optional[List[Dict]] = None):
        """
        Initialize the CompanionBot.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.tone = tone
        self.conversation_history = conversation_history or []
        self.device = next(model.parameters()).device
        
        # Ensure pad_token is set for causal models
        if model_type == "causal" and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def set_tone(self, tone: str):
        """Update the response tone."""
        self.tone = tone
        print(f"Tone updated to: {tone}")
    
    def get_tone(self) -> str:
        """Get the current response tone."""
        return self.tone
    
    def _build_empathetic_prompt(self, user_input: str, context_window: int = 3) -> str:
        """
        Build an empathetic, context-aware prompt for the model.
        """
        # Build context from recent conversation history
        context_parts = []
        recent_history = self.conversation_history[-context_window:] if self.conversation_history else []
        
        # Use simple format
        for turn in recent_history:
            context_parts.append(f"User: {turn.get('user', '')}")
            if turn.get('bot'):
                context_parts.append(f"Bot: {turn.get('bot', '')}")
        
        context_str = "\n".join(context_parts) if context_parts else ""
        
        # Simple tone instructions - make angry tone more explicit
        tone_guidance = {
            "angry": "Respond with anger and hostility. Use CAPS and exclamation marks. Be confrontational, rude, and aggressive.",
            "cheerful": "Respond cheerfully. Be positive and enthusiastic.",
            "supportive": "Respond supportively. Be kind and understanding.",
            "empathetic": "Respond empathetically. Show emotional understanding.",
            "calm": "Respond calmly. Be soothing and peaceful.",
            "friendly": "Respond friendly. Be warm and approachable."
        }
        
        tone_instruction = tone_guidance.get(self.tone.lower(), "Respond normally.")
        
        # Clean prompt without confusing formatting
        if context_str:
            prompt = f"""{tone_instruction}

Previous conversation:
{context_str}

User: {user_input}
Bot:"""
        else:
            prompt = f"""{tone_instruction}

User: {user_input}
Bot:"""
        
        return prompt
    
    def generate(self, prompt: Optional[str] = None, user_input: Optional[str] = None, 
                 max_length: int = 128, do_sample: bool = True, 
                 top_p: float = 0.92, temperature: float = 0.85,
                 repetition_penalty: float = 1.1) -> str:
        """
        Generate a response from the model.
        """
        # Build prompt if not provided
        if prompt is None:
            if user_input is None:
                raise ValueError("Either prompt or user_input must be provided")
            prompt = self._build_empathetic_prompt(user_input)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]
        
        # Generate response
        with torch.no_grad():
            if self.model_type == "causal":
                max_new_tokens = min(max_length, 30)  # Reduced to prevent instruction text, but should still allow good responses
                
                # Use stop sequences to prevent instruction text generation
                # This is better than post-processing pattern matching
                # TODO: Fine-tuning (Phase 2) will reduce need for this
                stop_strings = ["\n\nUser:", "\n\nBot:", "\nUser:", "\nBot:", 
                               "User:", "Bot:", "when you", "do not use", 
                               "please note", "in the event"]
                stopping_criteria = StoppingCriteriaList([
                    InstructionStoppingCriteria(self.tokenizer, stop_strings)
                ])
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=5,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=2,
                    stopping_criteria=stopping_criteria,
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=10,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        
        # Decode response
        if self.model_type == "causal":
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Clean up response - split by newlines first
            lines = response.split('\n')
            clean_lines = []
            for line in lines:
                line = line.strip()
                # Stop if line starts with User/Bot
                if line.lower().startswith('user:') or line.lower().startswith('bot:'):
                    break
                # Stop if line starts with instruction patterns
                line_lower = line.lower()
                if any(line_lower.startswith(phrase) for phrase in [
                    "do not use", "do not respond", "use your best", "do not believe",
                    "anything that suggests", "anything other than"
                ]):
                    break
                clean_lines.append(line)
            response = ' '.join(clean_lines).strip()
            
            # Remove any Bot: prefix
            if response.lower().startswith('bot:'):
                response = response[4:].strip()
            
            # Remove instruction-like text - balanced approach
            response_lower = response.lower()
            
            # Check for inappropriate content first
            inappropriate_patterns = [
                "sex", "sexual", "nude", "naked", "porn", "fuck", "shit", "damn"
            ]
            # Only filter if it's clearly inappropriate (not just containing the word in context)
            # For now, we'll be lenient and only catch obvious cases
            
            # Remove prompt patterns that might leak through (only at start)
            prompt_leakage_patterns = [
                "you are a chatbot",
                "your response tone is set to",
            ]
            
            # Check if response STARTS with exact prompt leakage (be more specific)
            for pattern in prompt_leakage_patterns:
                if response_lower.startswith(pattern):
                    # Find the first sentence that doesn't start with a prompt pattern
                    sentences = response.split('. ')
                    found_valid = False
                    for i, sentence in enumerate(sentences):
                        sentence_lower = sentence.strip().lower()
                        if not any(sentence_lower.startswith(p) for p in prompt_leakage_patterns):
                            response = '. '.join(sentences[i:]).strip()
                            found_valid = True
                            break
                    if not found_valid:
                        # If all sentences are prompt leakage, try to salvage something
                        # Take the last sentence even if it has prompt text
                        if sentences:
                            response = sentences[-1].strip()
                        else:
                            response = ""
                    break
            
            # TODO: TEMPORARY - Pattern-based instruction detection
            # This is a band-aid solution. Proper fixes:
            # 1. Fine-tune model on conversational data (see PROJECT_ROADMAP.md Phase 2)
            # 2. Use stop sequences in generation (implemented below)
            # 3. Train a classifier to detect instruction-like text (more robust)
            # 
            # For now, use a simple heuristic: stop at common instruction patterns
            # This catches the most common cases but is not comprehensive
            
            # Heuristic: If response contains instruction-like patterns, stop there
            # Common patterns that indicate the model is generating instructions/training data
            instruction_keywords = [
                "when you respond", "you must use", "important instructions",
                "do not use", "do not respond", "use your best judgment",
                "please note that", "in the event that", "this will allow"
            ]
            
            # Find the earliest occurrence of any instruction keyword
            earliest_instruction_idx = len(response)
            for keyword in instruction_keywords:
                idx = response_lower.find(keyword)
                if idx != -1 and idx < earliest_instruction_idx:
                    earliest_instruction_idx = idx
            
            # If we found instruction text, cut it off
            if earliest_instruction_idx < len(response):
                response = response[:earliest_instruction_idx].strip()
                # Take only complete sentences before the cut
                sentences = response.split('. ')
                if len(sentences) > 1:
                    response = '. '.join(sentences[:-1]).strip() + '.'
                response_lower = response.lower()
            
            # Limit to first 2 sentences maximum for better control
            sentences = response.split('. ')
            
            # Stop at instruction-like phrases
            clean_sentences = []
            instruction_starters = [
                "in the event", "when you", "this will", "once you",
                "please note", "they can", "it is difficult",
                "please report", "we have been", "this behavior",
                "it is an extremely", "this type of", "previous conversation:",
                "do not use", "do not respond", "use your best", "do not believe",
                "anything that suggests", "anything other than"
            ]
            
            for sentence in sentences:
                sentence_lower = sentence.strip().lower()
                # Stop if sentence starts with instruction-like phrases
                if any(sentence_lower.startswith(phrase) for phrase in instruction_starters):
                    break
                # Skip empty sentences
                if sentence.strip():
                    clean_sentences.append(sentence)
                # Limit to 2 sentences max
                if len(clean_sentences) >= 2:
                    break
            
            if clean_sentences:
                response = '. '.join(clean_sentences).strip()
                # Only add period if we have a complete sentence and it doesn't end with punctuation
                if response and not response[-1] in '.!?':
                    response += '.'
            else:
                # If we removed everything, try to keep at least the first sentence
                if sentences:
                    first = sentences[0].strip()
                    # Only use it if it's substantial and not clearly prompt leakage
                    if first and len(first) > 5 and not any(first.lower().startswith(p) for p in prompt_leakage_patterns):
                        response = first
                        if not response[-1] in '.!?':
                            response += '.'
            
            # Clean extra whitespace and newlines
            response = ' '.join(response.split())
            
        else:
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "Bot:" in response:
                response = response.split("Bot:")[-1].strip()
        
        # Ensure we have a response
        if not response or len(response.strip()) < 3:
            # If we got here, the cleaning was too aggressive or generation failed
            # Try generating again with simpler prompt or return a default
            return "I'm not sure what to say."
        
        # TODO: TEMPORARY basic safety check - will be replaced with proper Safety Layer
        # See PROJECT_ROADMAP.md Phase 3 for full safety layer implementation
        # This is a minimal filter until we implement the proper safety layer using Real Toxic dataset
        response_lower = response.lower()
        # Basic inappropriate content filtering (temporary measure)
        inappropriate_patterns = ["sex with", "sexual act", "want sex", "have sex"]
        if any(pattern in response_lower for pattern in inappropriate_patterns):
            # Return safe response - proper safety layer will handle this better
            return "I can't respond to that."
        
        return response
    
    def chat(self, user_input: str, debug: bool = False) -> str:
        """
        Generate a response and update conversation history.
        
        Args:
            user_input: The user's message
            debug: If True, print the prompt for debugging
        """
        # Build the prompt
        prompt = self._build_empathetic_prompt(user_input)
        
        # Debug: print the prompt
        if debug:
            print("\n" + "="*60)
            print("DEBUG - PROMPT BEING SENT TO MODEL:")
            print("-"*60)
            print(prompt)
            print("="*60 + "\n")
        
        # Generate response
        response = self.generate(prompt=prompt)
        
        # Update conversation history
        self.conversation_history.append({
            "user": user_input,
            "bot": response
        })
        
        # Keep history manageable
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        return response
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []
        print("Conversation reset.")
    
    def save_conversation(self, filepath: str):
        """Save conversation history to a file."""
        data = {
            "tone": self.tone,
            "conversation_history": self.conversation_history
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_conversation(self, filepath: str):
        """Load conversation history from a file."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.tone = data.get("tone", "supportive")
                self.conversation_history = data.get("conversation_history", [])
                print(f"Loaded conversation with tone: {self.tone}")
        else:
            print(f"File {filepath} not found.")


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


if __name__ == "__main__":
    # Load configuration
    try:
        config = load_config()
        model_config = config.get("model", {})
        MODEL_NAME = model_config.get("name", "EleutherAI/gpt-neo-1.3B")
        MODEL_TYPE = model_config.get("type", "causal")
        LOCAL_MODEL_DIR = model_config.get("local_dir", "./models/gpt-neo-1.3B")
    except FileNotFoundError:
        print("Warning: config.yaml not found. Using default values.")
        MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
        MODEL_TYPE = "causal"
        LOCAL_MODEL_DIR = "./models/gpt-neo-1.3B"
    
    # Load model & tokenizer
    print("Loading model...")
    
    try:
        # Check if local model directory exists
        if LOCAL_MODEL_DIR and os.path.exists(LOCAL_MODEL_DIR):
            print(f"Loading model from local directory: {LOCAL_MODEL_DIR}")
            tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
            if MODEL_TYPE == "causal":
                model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_DIR)
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_MODEL_DIR)
        else:
            if LOCAL_MODEL_DIR:
                print(f"Local model not found at {LOCAL_MODEL_DIR}")
                print("Downloading from HuggingFace...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            if MODEL_TYPE == "causal":
                model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        
        # Set pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Move to GPU if available
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        
        model = model.to(device)
        model.eval()
        print(f"Model loaded on device: {device}")
        
        # Initialize bot
        default_tone = "supportive"
        bot = CompanionBot(model, tokenizer, model_type=MODEL_TYPE, tone=default_tone)
        
        print("\n=== Companion Bot is ready! ===")
        print(f"Current tone: {default_tone}")
        print("\nCommands:")
        print("  - 'tone <name>' to change tone")
        print("  - 'debug on/off' to toggle debug mode")
        print("  - 'reset' to reset conversation")
        print("  - 'save' to save conversation")
        print("  - 'load' to load conversation")
        print("  - 'exit' or 'quit' to quit")
        print()
        
        # Variables for the chat loop
        debug_mode = False
        
        # Simple REPL loop
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
            
                # Handle save
            if user_input.lower() == "save":
                bot.save_conversation("conversation.json")
                print("Conversation saved to conversation.json")
                continue
            
                # Handle load
            if user_input.lower() == "load":
                bot.load_conversation("conversation.json")
                continue
            
                # Handle test commands
                if user_input.lower() == "test params":
                    # Test with different parameters
                    print("Testing with varied parameters...")
                    prompt = bot._build_empathetic_prompt(user_input)
                    response = bot.generate(
                        prompt=prompt,
                        temperature=0.9,
                        repetition_penalty=1.05,
                        top_p=0.95
                    )
                    print(f"Bot: {response}\n")
                    continue
                
                # Normal chat
                try:
                    print("Generating response...")
                    reply = bot.chat(user_input, debug=debug_mode)
                    
                if reply and reply.strip():
                    print(f"Bot: {reply}\n")
                else:
                        print("Bot: (No response generated)\n")
                        
                except torch.cuda.OutOfMemoryError:
                    print("Error: Out of GPU memory. Try reducing max_length.\n")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                        print("Error: Out of memory. Try using CPU or smaller model.\n")
                else:
                        print(f"Error: {e}\n")
            except Exception as e:
                    print(f"Error: {e}\n")
                    
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'exit' to quit.")
                continue
                
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure transformers is installed: pip install transformers torch")
        print("2. Check your internet connection for model download")
        print("3. Verify the model path is correct")