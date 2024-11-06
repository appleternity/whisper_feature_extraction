# Whisper Embedding Extractor

This tool extracts embeddings from audio files using OpenAI's Whisper model. It processes audio files in batches and saves the embeddings in HDF5 format for privacy-preserving downstream processing.


## Installation
We suggest initiating a virtual environment first.

### 1. Clone the repo
```bash
git clone https://github.com/appleternity/whisper_feature_extraction.git
```

### 2. Install PyTorch

See https://pytorch.org/get-started/locally/ for details.

For CPU-only:
```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

For GPU support (CUDA 12.1):
```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install other dependencies

```bash
python -m pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python extract_embeddings.py --audio_folder /path/to/audio/files --output_path embeddings.h5
```

Full options:
Use `cuda` for gpu and `cpu` for cpu.
```bash
python extract_embeddings.py \
    --audio_folder /path/to/audio/files \
    --output_path "embeddings.h5" \
    --model_name "openai/whisper-tiny" \
    --layer_idx 0 \
    --batch_size 8 \
    --device cuda
```

Actual executable command:
```bash
python feature_extracion.py \
    --audio_folder audios_files \
    --output_path output/embeddings.h5 \
    --model_name "openai/whisper-tiny" \
    --layer_idx 0 \
    --device "cpu" \
    --batch_size 1 
```

### Arguments

- `--audio_folder`: Path to folder containing audio files (required)
- `--output_path`: Path where embeddings will be saved (required)
- `--model_name`: Whisper model to use from Huggingface (default: `openai/whisper-tiny`)
- `--layer_idx`: Which layer to extract embeddings from (default: `0`)
  - `0`: embedding layer
  - `1+`: encoder layers
- `--batch_size`: Number of files to process simultaneously (default: `8`). A larger batch_size will require more memory. Set to `1` if the memory resource is limited.
- `--device`: Device to run the model on (default: `cuda` if available, else `cpu`). Use `mps` if on mac and M1/M2/M3/M4 is available.

## Output Format

The script saves the embeddings in HDF5 format with two datasets:
- `embeddings`: The extracted embeddings tensor
- `attention_masks`: The corresponding attention masks

## Memory Considerations

If you run into memory issues:
1. Reduce the batch size using `--batch_size`
2. Process on CPU if GPU memory is limited

## Sample Data
The sample audio files are taken from The LJ Speech Dataset.
https://keithito.com/LJ-Speech-Dataset/
