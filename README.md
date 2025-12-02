# Companion Bot

A conversational AI agent that generates empathetic, tone-aware responses using the Blenderbot model. This bot can adapt its communication style based on different tones (cheerful, supportive, empathetic, calm, friendly, angry) to provide more personalized interactions.

## Features

- **Tone-aware responses**: Adjust the bot's communication style with different tones
- **Conversation history**: Maintains context across multiple conversation turns
- **Multiple tones**: Support for cheerful, supportive, empathetic, calm, friendly, and angry tones
- **Conversation persistence**: Save and load conversation history
- **Easy to use**: Simple command-line interface

## Installation

1. Clone this repository:
```bash
git clone https://github.com/lacey1998/companion_bot.git
cd companion_bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the model (optional - will download automatically on first run):
```bash
python download_model.py
```

## Usage

Run the bot:
```bash
python companion_bot.py
```

### Commands

- `tone <name>` - Change the bot's tone (cheerful, supportive, empathetic, calm, friendly, angry)
- `save` - Save conversation to `conversation.json`
- `load` - Load conversation from `conversation.json`
- `exit` or `quit` - Exit the bot

### Example

```
You: Hello!
Bot: Hello! How are you doing today?

tone angry
Tone updated to: angry

You: I'm having a bad day
Bot: [Response in angry tone]
```

## Code Example

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from companion_bot import BlenderbotResponder

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")

# Initialize bot with a specific tone
bot = BlenderbotResponder(model, tokenizer, tone="supportive")

# Chat with the bot
response = bot.chat("I'm feeling stressed about work")
print(response)
```

## Available Tones

- **cheerful**: Upbeat, enthusiastic, and positive
- **supportive**: Understanding, encouraging, and caring
- **empathetic**: Emotionally attuned, compassionate, and validating
- **calm**: Soothing, peaceful, and reassuring
- **friendly**: Warm, approachable, and conversational
- **angry**: Frustrated, assertive, and direct

## Requirements

- Python 3.7+
- torch >= 1.9.0
- transformers >= 4.20.0
- tokenizers >= 0.13.0

## Model

This project uses the `facebook/gpt-neo-1.3B` model from Hugging Face. The model will be automatically downloaded on first run if not already present locally.

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

