# MLX Fine-tuning Project

A comprehensive toolkit for fine-tuning large language models using MLX (for Apple Silicon) and integrating with Ollama.

## Project Structure

```
├── Modelfile             # Configuration file for Ollama model creation
├── src/                  # Source code
│   ├── material_generation.py  # Generate Q&A pairs from source materials
│   ├── clean_qa_data.py        # Clean and process generated Q&A data
│   ├── compare_models.py       # Compare MLX fine-tuned model with Ollama models
│   ├── grab_data.py            # Utility for data collection
│   └── ollama_integration.py   # Ollama integration utilities
├── data/                 # Data files
│   ├── raw/              # Raw data before processing
│   └── processed/        # Processed data ready for training (Train.jsonl, Valid.jsonl, Test.jsonl)
├── models/               # Model files
│   ├── adapters 1/       # First version of fine-tuned model adapters
│   ├── adapters 2/       # Second version of fine-tuned model adapters
│   └── adapters X/       # Additional versions as needed
├── backup/               # Backup of previous training runs
│   ├── cleaned-dataset 1/# First version of cleaned dataset
│   ├── cleaned-dataset 2/# Second version of cleaned dataset
│   └── cleaned-dataset X/# Additional versions as needed
├── outputs/              # Generated outputs from models
├── course_materials/     # Source course materials (PDFs, slides, docs)
└── scripts/              # Utility scripts
    └── manage_datasets.py # Script to manage dataset versions
```

## Prerequisites

1. MLX installed:
```bash
pip install mlx-lm
```

2. Ollama installed (for model comparison):
```bash
# MacOS
curl -fsSL https://ollama.com/install.sh | sh
```

3. Required Python libraries:
```bash
pip install jsonlines transformers python-pptx python-docx pymupdf
```

## Workflow

### 1. Generate Training Data

Process course materials and generate question-answer pairs:

```bash
python src/material_generation.py --slides_dir "course_materials" --use_ollama --ollama_model "deepseek-r1:latest" --num_samples 500
```

Options:
- `--slides_dir`: Directory containing educational materials (.pptx, .docx, .pdf, .txt)
- `--use_ollama`: Use Ollama for generating Q&A pairs (recommended)
- `--ollama_model`: Ollama model to use (default: "deepseek-r1:latest")
- `--num_samples`: Number of Q&A pairs to generate
- `--output`: Output file path (default: "../data/raw/course_qa_data.jsonl")
- `--split`: Split the generated data into train/validation/test sets

### 2. Clean and Process the Data

Clean the generated data and prepare it for fine-tuning:

```bash
python src/clean_qa_data.py
```

This will:
- Clean up thinking artifacts and formatting
- Split data into train/validation/test sets
- Save processed data to `data/processed/`

### 3. Backup Your Dataset (Optional)

After cleaning your data, you can backup the dataset for future reference:

```bash
python scripts/manage_datasets.py backup
```

This will:
- Create a versioned copy of your dataset in the `backup/` directory
- Auto-increment version numbers (e.g., "cleaned-dataset 3")
- Add metadata with timestamp and file information

Other dataset management options:
```bash
# List all available dataset versions
python scripts/manage_datasets.py list

# Restore a specific dataset version for training
python scripts/manage_datasets.py restore "cleaned-dataset 1"

# Create a backup with a custom name
python scripts/manage_datasets.py backup --name "my-custom-dataset"
```

### 4. Fine-tune the Model

Fine-tune the model using MLX:

```bash
mlx_lm.lora --model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" --train --data "./data/processed" --iters 100 --num-layer 32
```

Options:
- `--model`: Base model to fine-tune
- `--data`: Directory containing the processed data
- `--iters`: Number of iterations for training
- `--num-layer`: Number of layers to fine-tune
- `--batch-size`: Batch size for training
- `--grad-checkpoint`: Use gradient checkpointing

After training, rename the generated `adapters` folder to maintain versioning:
```bash
# Assuming this is your third adapter version
mv adapters/ models/adapters\ 3/
```

### 5. Test and Compare Models

Compare your fine-tuned MLX model with Ollama models:

```bash
# List available adapter versions
python src/compare_models.py --list-adapters

# Use a specific adapter version
python src/compare_models.py --prompt "What is tort law?" --adapter-version "adapters 1" --mlx-only

# Use the latest adapter version
python src/compare_models.py --prompt "What is tort law?" --mlx-only

# Just use the Ollama model
python src/compare_models.py --prompt "What is tort law?" --ollama-model "deepseek-r1:latest" --ollama-only

# Compare both models
python src/compare_models.py --prompt "What is tort law?" --ollama-model "deepseek-r1:latest"

# Control temperature and max tokens
python src/compare_models.py --prompt "What is tort law?" --temp 0.2 --max-tokens 1000

# Save the output to a file
python src/compare_models.py --prompt "What is tort law?" --save-output
```

### 6. Ollama Integration (Optional)

If you want to use your fine-tuned model with Ollama:

```bash
# Create a Modelfile (specifying which adapter version to use)
echo 'FROM deepseek-r1:latest
ADAPTER ./models/adapters 2' > Modelfile

# Create a custom Ollama model
ollama create tuned-deepseek-r1 -f Modelfile

# Run the model
ollama run tuned-deepseek-r1
```

## Notes

- The fine-tuned adapters are stored in `./models/adapters X/` directories
- Model outputs are saved to `./outputs/` when using the `--save-output` flag
- Advanced fine-tuning parameters can be adjusted in the MLX command
- For better results, increase the number of training iterations (e.g., --iters 500)

## Troubleshooting

If your fine-tuned model generates responses that include "thinking" patterns:
1. Use a better system prompt in `compare_models.py` 
2. Increase training iterations
3. Try a lower temperature setting (e.g., --temp 0.2)
