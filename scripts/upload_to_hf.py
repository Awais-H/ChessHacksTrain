"""
Upload existing model from Modal volume to HuggingFace.
Usage: modal run scripts/upload_to_hf.py
"""

import modal
import os
import sys
from pathlib import Path

app = modal.App("chess-upload-to-hf")

# Get the project root directory
project_root = Path(__file__).parent.parent

# Create Modal image with dependencies
# Force rebuild to pick up new config changes
image = (
    modal.Image.debian_slim()
    .pip_install("huggingface-hub", "torch")
    .add_local_dir(
        local_path=str(project_root / "lib"),
        remote_path="/root/lib",
        copy=True
    )
)


@app.function(
    image=image,
    volumes={"/data": modal.Volume.from_name("chess-data", create_if_missing=True)},
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def upload_model():
    """Upload the trained model from Modal volume to HuggingFace."""
    import sys
    sys.path.insert(0, '/root/lib')
    
    from config import HF_REPO_ID, HF_MODEL_FILENAME
    from hf_utils import upload_model
    
    model_path = "/data/chess_model.pth"
    
    print("="*60)
    print("UPLOADING MODEL TO HUGGINGFACE")
    print("="*60)
    print(f"\nRepository: {HF_REPO_ID}")
    print(f"Model file: {HF_MODEL_FILENAME}")
    print(f"Local path: {model_path}\n")
    
    # Check if model exists
    import os
    if not os.path.exists(model_path):
        print(f"✗ Model not found at {model_path}")
        return {"success": False, "error": "Model file not found"}
    
    # Get file size
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model size: {file_size:.2f} MB\n")
    
    # Upload
    success = upload_model(
        model_path=model_path,
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILENAME,
        commit_message="Upload trained model from Modal"
    )
    
    if success:
        print("\n" + "="*60)
        print("✓ UPLOAD SUCCESSFUL!")
        print("="*60)
        print(f"\nView your model at:")
        print(f"https://huggingface.co/{HF_REPO_ID}\n")
    else:
        print("\n✗ Upload failed. Check the error above.")
    
    return {"success": success}


@app.local_entrypoint()
def main():
    """Local entrypoint."""
    result = upload_model.remote()
    
    if result["success"]:
        print("\n✓ Model uploaded successfully!")
    else:
        print(f"\n✗ Upload failed: {result.get('error', 'Unknown error')}")
