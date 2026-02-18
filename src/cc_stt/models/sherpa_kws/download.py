"""Download utility for Sherpa-ONNX KWS models."""

import os
import tarfile
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError


MODELS = {
    "sherpa-onnx-kws-zipformer-wenetspeech-3.3M": {
        "url": (
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/"
            "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2"
        ),
        "filename": "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2",
        "extract_dir": "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01",
    },
    "sherpa-onnx-kws-zipformer-zh-en-3M": {
        "url": (
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/"
            "sherpa-onnx-kws-zipformer-zh-en-3M-2024-12-20.tar.bz2"
        ),
        "filename": "sherpa-onnx-kws-zipformer-zh-en-3M-2024-12-20.tar.bz2",
        "extract_dir": "sherpa-onnx-kws-zipformer-zh-en-3M-2024-12-20",
    },
}


def download_model(model_name: str, output_dir: str | Path) -> Path:
    """Download and extract a Sherpa-ONNX KWS model.

    Args:
        model_name: Name of the model to download
        output_dir: Directory to download and extract the model to

    Returns:
        Path to the extracted model directory

    Raises:
        ValueError: If model_name is not recognized
        URLError: If download fails
    """
    if model_name not in MODELS:
        available = ", ".join(MODELS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    model_info = MODELS[model_name]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tar_path = output_path / model_info["filename"]

    # Download if not exists
    if not tar_path.exists():
        print(f"Downloading {model_name}...")
        try:
            urlretrieve(model_info["url"], tar_path)
        except URLError as e:
            if tar_path.exists():
                tar_path.unlink()
            raise URLError(f"Failed to download {model_name}: {e}")

    # Extract if not already extracted
    extract_dir = output_path / model_info["extract_dir"]
    if not extract_dir.exists():
        print(f"Extracting {model_name}...")
        with tarfile.open(tar_path, "r:bz2") as tar:
            tar.extractall(output_path)

    return extract_dir


def ensure_model_exists(model_dir: str | Path) -> Path:
    """Ensure a model exists, downloading if necessary.

    Auto-detects the model name from the directory path and downloads
    if the model files are not present.

    Args:
        model_dir: Path to the model directory (may contain model name)

    Returns:
        Path to the model directory

    Raises:
        ValueError: If model cannot be auto-detected from path
    """
    model_path = Path(model_dir)

    # If directory exists and has files, assume it's valid
    if model_path.exists() and any(model_path.iterdir()):
        return model_path

    # Try to detect model name from path
    path_str = str(model_path)
    for model_name in MODELS:
        if model_name in path_str:
            parent_dir = model_path.parent
            return download_model(model_name, parent_dir)

    # If model directory name matches extracted dir name, try to find parent
    for model_name, info in MODELS.items():
        if model_path.name == info["extract_dir"]:
            return download_model(model_name, model_path.parent)

    raise ValueError(
        f"Cannot auto-detect model for path: {model_dir}. "
        f"Available models: {', '.join(MODELS.keys())}"
    )
