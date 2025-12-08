# Companion Bot

A conversational AI agent that generates empathetic, tone-aware responses using fine-tuned language models. The bot is fine-tuned on empathetic dialogues using LoRA (Low-Rank Adaptation) to improve its ability to understand and respond to emotional contexts. It can adapt its communication style based on different tones (cheerful, supportive, empathetic, calm, friendly, angry) to provide more personalized interactions.

## Features

- **Fine-tuned for empathy**: Model fine-tuned on Empathetic Dialogues dataset using LoRA
- **Tone-aware responses**: Adjust the bot's communication style with different tones
- **Emotion detection**: Automatically detects user emotions and responds appropriately
- **Conversation history**: Maintains context across multiple conversation turns
- **Multiple tones**: Support for cheerful, supportive, empathetic, calm, friendly, and angry tones
- **Model evaluation**: Comprehensive evaluation scripts for perplexity, BLEU, and human testing
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

### Basic Usage - Run the Bot

Run the bot with base model:
```bash
python companion_bot.py
```

### Using Fine-tuned LoRA Model

Test the fine-tuned model interactively:
```bash
python testing_scripts/test_lora_bot.py
```

Run batch testing with predefined test cases:
```bash
python testing_scripts/test_lora_bot.py batch
```

### Commands (Interactive Mode)

- `tone <name>` - Change the bot's tone (cheerful, supportive, empathetic, calm, friendly, angry)
- `debug on/off` - Toggle debug mode to see prompts and detected emotions
- `reset` - Reset conversation history
- `save` - Save conversation to `conversation.json`
- `load` - Load conversation from `conversation.json`
- `exit` or `quit` - Exit the bot

### Examples

**Basic Chat:**
```

You: I got promoted at work!
Bot: Congratulations! That's amazing news! You must be so excited and proud of your achievement.
```

**Changing Tone:**
```
tone empathetic
Tone updated to: empathetic

You: I'm feeling really anxious about my exam.
Bot: I understand that anxiety can be really overwhelming. It's completely normal to feel this way before an exam. What specifically is making you feel anxious?
```

**With Fine-tuned Model:**
```bash
python testing_scripts/test_lora_bot.py

You: I'm grateful for everything I have.
Bot: That's wonderful that you're feeling grateful! Gratitude is such a powerful emotion. What are you most grateful for today?
```

## Model Evaluation

### Evaluate Base Model vs Fine-tuned Model

**On Cluster (using SLURM):**

Evaluate base model:
```bash
cd testing_scripts
sbatch run_evaluate_base.sh
```

Evaluate fine-tuned LoRA model:
```bash
cd testing_scripts
sbatch run_evaluate.sh
```

**Locally:**
```bash
# Evaluate base model
python evaluate_model.py \
  --model_path "models/opt-1.3b" \
  --test_data "data/test.json" \
  --use_emotion \
  --output_file "eval_base.json"

# Evaluate fine-tuned model
python evaluate_model.py \
  --model_path "models/opt-1.3b" \
  --lora_path "checkpoints/opt-1.3b-empathetic-lora" \
  --test_data "data/test.json" \
  --use_emotion \
  --output_file "eval_lora.json"
```

### Evaluation Metrics

The evaluation script computes:
- **Perplexity**: Lower is better - measures language modeling quality
- **BLEU Score**: Higher is better - measures n-gram overlap with reference
- **Average Loss**: Lower is better - measures prediction accuracy


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
- peft >= 0.3.0 (for LoRA fine-tuning)
- sacrebleu >= 2.0.0 (for BLEU score evaluation)
- datasets >= 2.0.0 (for training)

Install all dependencies:
```bash
pip install -r requirements.txt
```

<!-- ## Project Structure

```
.
├── companion_bot.py          # Main bot implementation
├── evaluate_model.py          # Model evaluation script
├── train_lora.py              # LoRA fine-tuning script
├── emotion_detector.py        # Emotion detection module
├── testing_scripts/          # Testing and evaluation scripts
│   ├── run_evaluate.sh       # Batch script for LoRA model evaluation
│   ├── run_evaluate_base.sh  # Batch script for base model evaluation
│   └── test_lora_bot.py      # Interactive and batch testing script
├── models/                    # Base model directory
├── checkpoints/               # Fine-tuned model checkpoints
└── data/                      # Training and test data
``` -->

## Model

This project uses **OPT-1.3B** (Open Pre-trained Transformer) as the base model, fine-tuned on the Empathetic Dialogues dataset using LoRA (Low-Rank Adaptation).

### Model Details

- **Base Model**: OPT-1.3B (1.3 billion parameters)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Data**: Empathetic Dialogues dataset
- **Fine-tuned Features**: Emotion-aware response generation

### Model Files

- Base model: `models/opt-1.3b/`
- Fine-tuned LoRA adapter: `checkpoints/opt-1.3b-empathetic-lora/` or `code/lora-tuned/opt-1.3b-empathetic-lora-exp5-r32/`

The model will be automatically downloaded on first run if not already present locally.

## License

This project is open source and available under the MIT License.



