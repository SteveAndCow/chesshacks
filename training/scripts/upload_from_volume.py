"""
Upload trained model from Modal Volume to HuggingFace.
"""
import modal

app = modal.App("upload-model-to-hf")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface-hub",
    "torch",
)

# Modal Volume for persistent data storage
data_volume = modal.Volume.from_name("chess-training-data")


@app.function(
    image=image,
    volumes={"/data": data_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def upload_model(repo_id: str, model_name: str):
    """
    Upload model from Modal Volume to HuggingFace.

    Args:
        repo_id: HuggingFace repo (e.g., "steveandcow/chesshacks-bot")
        model_name: Model file name in volume (e.g., "cnn_baseline")
    """
    import os
    from pathlib import Path
    from huggingface_hub import HfApi

    print("="*60)
    print("UPLOADING MODEL FROM MODAL VOLUME TO HUGGINGFACE")
    print("="*60)

    # Check token
    token = os.getenv("HF_TOKEN")
    if not token:
        print("❌ HF_TOKEN not found!")
        return {"error": "No HF token"}

    print(f"✅ Token found: {token[:10]}...{token[-4:]}")

    # Find model in volume
    model_path = f"/data/models/{model_name}.pt"

    if not Path(model_path).exists():
        print(f"❌ Model not found at: {model_path}")
        print("\nAvailable models in /data/models/:")
        models_dir = Path("/data/models")
        if models_dir.exists():
            for file in models_dir.glob("*.pt"):
                print(f"  - {file.name}")
        else:
            print("  (no models directory found)")
        return {"error": "Model not found", "path": model_path}

    print(f"✅ Found model at: {model_path}")

    # Get file size
    file_size = Path(model_path).stat().st_size / (1024 * 1024)  # MB
    print(f"Model size: {file_size:.2f} MB")

    # Upload to HuggingFace
    try:
        print(f"\n📤 Uploading to: {repo_id}/{model_name}.pt")
        api = HfApi(token=token)

        result = api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=f"{model_name}.pt",
            repo_id=repo_id,
        )

        print(f"✅ Upload successful!")
        print(f"Model URL: https://huggingface.co/{repo_id}/blob/main/{model_name}.pt")

        return {
            "status": "success",
            "repo_id": repo_id,
            "model_name": model_name,
            "url": f"https://huggingface.co/{repo_id}",
        }
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return {"error": str(e)}


@app.local_entrypoint()
def main(
    repo_id: str = "steveandcow/chesshacks-bot",
    model_name: str = "cnn_baseline"
):
    """
    Upload model to HuggingFace.

    Usage:
        modal run scripts/upload_from_volume.py
        modal run scripts/upload_from_volume.py --model-name cnn_baseline
    """
    print(f"🚀 Uploading model from Modal Volume...")
    print(f"Repo: {repo_id}")
    print(f"Model: {model_name}")
    print()

    result = upload_model.remote(repo_id, model_name)

    print("\n" + "="*60)
    print("RESULT")
    print("="*60)
    print(result)
