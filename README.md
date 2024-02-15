# Efficient fine-tuning of Llama 2

Streamlined tokenizing for effective text processing and fine-tuned the Transformer-based model on the Alpaca Dataset, to achieve a balance between processing efficiency and model accuracy.

**alpaca_data_dummy.json** - Alpaca dataset used to train the Llama model

**example_text_completion.py** - Python script for generating text using the Llama 2 model

**finetune.py** - Fine-tuned Llama 2 model

**inference.py** - Python script for running inference

**llama**
    - **generation.py** - Text generation module that generates text from prompts using the tokenizer
    - **lora.py** - Custom PyTorch linear layer class that incorporates LoRA (Low-Rank Adaptation)
    - **model.py** - Llama 2 model
    - **tokenizer.py** - Tokenizer class for encoding text to token IDs and decoding token IDs back to text