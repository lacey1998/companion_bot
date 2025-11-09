from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import Optional, List, Dict
import json
import os


class BlenderbotResponder:
    """
    A conversational agent that generates empathetic, tone-aware responses
    using the Blenderbot model.
    """
    
    def __init__(self, model, tokenizer, tone: str = "supportive", conversation_history: Optional[List[Dict]] = None):
        """
        Initialize the BlenderbotResponder.
        
        Args:
            model: The loaded Blenderbot model
            tokenizer: The loaded tokenizer
            tone: The response tone (e.g., "cheerful", "supportive", "empathetic")
            conversation_history: Optional list of previous conversation turns
        """
        self.model = model
        self.tokenizer = tokenizer
        self.tone = tone
        self.conversation_history = conversation_history or []
        self.device = next(model.parameters()).device
        
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
        
        Args:
            user_input: The current user message
            context_window: Number of recent conversation turns to include
            
        Returns:
            A formatted prompt string
        """
        # Build context from recent conversation history
        context_parts = []
        recent_history = self.conversation_history[-context_window:] if self.conversation_history else []
        
        for turn in recent_history:
            context_parts.append(f"User: {turn.get('user', '')}")
            if turn.get('bot'):
                context_parts.append(f"Bot: {turn.get('bot', '')}")
        
        context_str = "\n".join(context_parts) if context_parts else ""
        
        # Build empathetic prompt with tone guidance
        tone_guidance = {
            "cheerful": """You MUST respond in a cheerful, upbeat, and positive manner. Use enthusiastic language, exclamation points, and show optimism. Be energetic and happy. Do NOT be neutral or serious.""",
            "supportive": """You MUST respond in a supportive, understanding, and encouraging manner. Show care, empathy, and offer help. Be warm and reassuring. Do NOT be dismissive or uncaring.""",
            "empathetic": """You MUST respond with deep empathy and emotional understanding. Acknowledge the user's feelings explicitly, show compassion, and validate their emotions. Be emotionally attuned. Do NOT be emotionally distant.""",
            "calm": """You MUST respond in a calm, soothing, and reassuring manner. Use peaceful language, provide comfort, and maintain a stable tone. Be serene and composed. Do NOT be anxious or rushed.""",
            "friendly": """You MUST respond in a friendly, warm, and approachable manner. Be conversational, personable, and use casual, welcoming language. Do NOT be formal or distant.""",
            "angry": """You MUST respond in an angry, frustrated, and assertive manner."""
        }
        
        tone_instruction = tone_guidance.get(self.tone.lower(), tone_guidance["supportive"])
        
        # Construct the prompt with stronger tone enforcement
        if context_str:
            prompt = f"""You are a chatbot. Your response tone is set to: {self.tone.upper()}

IMPORTANT INSTRUCTIONS:
{tone_instruction}

You MUST maintain this tone in your response. The tone is non-negotiable.

Previous conversation:
{context_str}

User: {user_input}
Bot:"""
        else:
            prompt = f"""You are a chatbot. Your response tone is set to: {self.tone.upper()}

IMPORTANT INSTRUCTIONS:
{tone_instruction}

You MUST maintain this tone in your response. The tone is non-negotiable.

User: {user_input}
Bot:"""
        
        return prompt
    
    def generate(self, prompt: Optional[str] = None, user_input: Optional[str] = None, 
                 max_length: int = 128, do_sample: bool = True, 
                 top_p: float = 0.9, temperature: float = 0.8) -> str:
        """
        Generate a response from the model.
        
        Args:
            prompt: Optional explicit prompt. If not provided, will build from user_input
            user_input: The user's message (used if prompt is None)
            max_length: Maximum length of generated response
            do_sample: Whether to use sampling
            top_p: Nucleus sampling parameter
            temperature: Sampling temperature
            
        Returns:
            The generated response
        """
        # Build prompt if not provided
        if prompt is None:
            if user_input is None:
                raise ValueError("Either prompt or user_input must be provided")
            prompt = self._build_empathetic_prompt(user_input)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        # Generate response
        with torch.no_grad():
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
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up response (remove prompt if it got included)
        if "Bot:" in response:
            response = response.split("Bot:")[-1].strip()
        
        return response
    
    def chat(self, user_input: str) -> str:
        """
        Generate a response and update conversation history.
        
        Args:
            user_input: The user's message
            
        Returns:
            The bot's response
        """
        response = self.generate(user_input=user_input)
        
        # Update conversation history
        self.conversation_history.append({
            "user": user_input,
            "bot": response
        })
        
        # Keep history manageable (last 10 turns)
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        return response
    
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


if __name__ == "__main__":
    # Configuration: Set local model directory (None to use HuggingFace cache)
    LOCAL_MODEL_DIR = "./models/blenderbot-400M-distill"  # Set to None to use HuggingFace cache
    MODEL_NAME = "facebook/blenderbot-400M-distill"
    
    # Load model & tokenizer
    print("Loading model...")
    
    try:
        # Check if local model directory exists
        if LOCAL_MODEL_DIR and os.path.exists(LOCAL_MODEL_DIR):
            print(f"Loading model from local directory: {LOCAL_MODEL_DIR}")
            tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
            model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_MODEL_DIR)
        else:
            if LOCAL_MODEL_DIR:
                print(f"Local model not found at {LOCAL_MODEL_DIR}")
                print("Downloading from HuggingFace (this may take a few minutes)...")
                print("Tip: Run 'python download_model.py' to download model locally")
            else:
                print("Loading from HuggingFace cache...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"Model loaded on device: {device}")
        
        # Initialize bot with default tone
        default_tone = "supportive"
        bot = BlenderbotResponder(model, tokenizer, tone=default_tone)
        
        print("=== CompanionBot is ready! ===")
        print(f"Current tone: {default_tone}")
        print("Commands:")
        print("  - 'tone <name>' to change tone (cheerful, supportive, empathetic, calm, friendly, angry)")
        print("  - 'save' to save conversation")
        print("  - 'load' to load conversation")
        print("  - 'exit' or 'quit' to quit")
        print()
        
        # Simple REPL loop
        while True:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye! 👋")
                break
            
            # Handle tone change command
            if user_input.lower().startswith("tone "):
                new_tone = user_input[5:].strip()
                bot.set_tone(new_tone)
                continue
            
            # Handle save command
            if user_input.lower() == "save":
                bot.save_conversation("conversation.json")
                print("Conversation saved to conversation.json")
                continue
            
            # Handle load command
            if user_input.lower() == "load":
                bot.load_conversation("conversation.json")
                continue
            
            # Generate and print response
            try:
                reply = bot.chat(user_input)
                print(f"Bot: {reply}\n")
            except Exception as e:
                print(f"Error generating response: {e}\n")
                
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure you have internet connection for first-time download")
        print("  2. Or run 'python download_model.py' to download model to local directory")
        print("  3. Make sure transformers library is installed: pip install transformers torch")

