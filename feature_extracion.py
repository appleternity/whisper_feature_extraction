import argparse
from pathlib import Path
import logging
import librosa

import torch
from transformers import WhisperModel, WhisperFeatureExtractor
import h5py
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from Whisper model layers for privacy-preserving processing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--audio_folder",
        type=str,
        required=True,
        help="Path to a folder containing audio files. Supports .wav and .mp3 formats",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="File path where embeddings and metadata will be saved",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/whisper-tiny",
        help="Name of the Whisper model from Huggingface hub (e.g., openai/whisper-tiny)",
    )
    parser.add_argument(
        "--layer_idx",
        type=int,
        default=0,
        help="Which layer to extract embeddings from: 0 for embedding layer, 1 and up for encoder layers",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of audio files to process simultaneously. Reduce this if running out of memory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Device to run the model on: "cuda" for GPU, "cpu" for CPU',
    )
    return parser.parse_args()


class WhisperEmbeddingExtractor:
    """
    A class to extract embeddings from a Whisper model.

    This extractor loads a specified Whisper model and extracts embeddings
    from a chosen layer. These embeddings can be used for downstream tasks
    while keeping the original audio private.
    """

    def __init__(
        self, model_name="openai/whisper-large-v3", layer_idx=0, device="cuda"
    ):
        """
        Initialize the extractor with a specific Whisper model and configuration.

        Args:
            model_name: The name/path of the model on Huggingface hub
            layer_idx: Which layer to extract (0 = embedding layer)
            device: Which device to run on ('cuda' for GPU, 'cpu' for CPU)
        """
        self.device = device
        self.layer_idx = layer_idx
        self.model_name = model_name

        # Load the Whisper model from Huggingface
        logger.info(f"Loading Whisper model: {model_name}")
        self.model = WhisperModel.from_pretrained(model_name)
        self.model.to(device)  # Move model to specified device (GPU/CPU)

        # Load the Whisper feature extractor from Huggingface
        logger.info(f"Loading Whisper feature extractor: {model_name}")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            model_name,
            return_attention_mask=True,
        )

    @torch.inference_mode()  # More optimized version of no_grad for inference
    def extract_embeddings(self, audio_paths, batch_size=8):
        """
        Extract embeddings from audio files in batches.

        Args:
            audio_paths: List of paths to audio files
            batch_size: Number of files to process simultaneously

        Returns:
            tuple: (embeddings tensor, attention mask tensor)
        """
        all_embeddings = []
        all_attention_masks = []

        # Process audio files in batches to manage memory
        for i in tqdm(
            range(0, len(audio_paths), batch_size),
            desc="Extracting embeddings"
        ):
            batch_paths = audio_paths[i : i + batch_size]

            # Load audio files and convert to raw audio data
            raw_audios = [librosa.load(str(path), sr=16000)[0] for path in batch_paths]

            # Convert audio to features that the model can process
            # The feature extractor handles resampling to 16kHz and converting to log-mel spectrograms
            inputs = self.feature_extractor(
                raw_audios,
                return_tensors="pt",
                sampling_rate=16000
            )

            # Move inputs to the same device as the model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Run the encoder part of Whisper to get embeddings
            outputs = self.model.encoder(
                inputs["input_features"],
                output_hidden_states=True,
                return_dict=True,
            )

            # Get embeddings from the specified layer
            # Layer 0 is the embedding layer (raw features)
            # Layers 1+ are the outputs of each encoder layer
            layer_outputs = outputs.hidden_states[self.layer_idx]

            # Move results back to CPU and store
            all_embeddings.append(layer_outputs.cpu())
            all_attention_masks.append(inputs["attention_mask"].cpu())

        # Combine all batches into single tensors
        return torch.cat(all_embeddings), torch.cat(all_attention_masks)


def save_embeddings(embeddings, attention_masks, output_path):
    """
    Save the extracted embeddings and metadata to disk.

    Saves in HDF5 format, which is efficient for large numerical arrays
    and preserves the relationship between embeddings and their source files.

    Args:
        embeddings: Tensor of extracted embeddings
        attention_masks: Tensor of attention masks
        output_path: File path to save the results
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("embeddings", data=embeddings.numpy())
        f.create_dataset("attention_masks", data=attention_masks.numpy())

    logger.info(f"Saved embeddings to {output_path}")
    logger.info(f"Embeddings shape: {embeddings.shape}")


def main():
    """
    Main function to run the embedding extraction pipeline.
    """
    args = parse_args()

    # List all audio files from the provided folder
    audio_folder = Path(args.audio_folder)
    audio_paths = list(audio_folder.glob("**/*.wav")) + list(
        audio_folder.glob("**/*.mp3")
    )

    # Remove duplicates and sort for consistency
    audio_paths = sorted(list(set(audio_paths)))
    logger.info(f"Found {len(audio_paths)} audio files")

    # Create the embedding extractor
    extractor = WhisperEmbeddingExtractor(
        model_name=args.model_name,
        layer_idx=args.layer_idx,
        device=args.device,
    )

    # Extract embeddings from all audio files
    embeddings, attention_masks = extractor.extract_embeddings(
        audio_paths,
        batch_size=args.batch_size,
    )

    # Save the results
    save_embeddings(embeddings, attention_masks, args.output_path)


if __name__ == "__main__":
    main()
