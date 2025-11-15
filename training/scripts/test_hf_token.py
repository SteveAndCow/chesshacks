"""
Quick test to verify HuggingFace token works in Modal.
"""
import modal

app = modal.App("test-hf-token")

image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface-hub")

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def test_token():
    import os
    from huggingface_hub import HfApi

    print("Testing HuggingFace authentication...")

    # Check if token exists
    token = os.getenv("HF_TOKEN")
    if not token:
        print("❌ HF_TOKEN not found in environment!")
        return {"status": "failed", "error": "No token"}

    print(f"✅ Token found: {token[:10]}...{token[-4:]}")

    # Test authentication
    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        print(f"✅ Authenticated as: {user_info['name']}")
        print(f"✅ Username: {user_info.get('fullname', 'N/A')}")
        return {"status": "success", "user": user_info['name']}
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return {"status": "failed", "error": str(e)}

@app.local_entrypoint()
def main():
    result = test_token.remote()
    print(f"\nResult: {result}")
