from pathlib import Path

from transformers import ViTForImageClassification, ViTImageProcessor


# ============================================================
# CONFIG
# ============================================================

MODEL_DIR = Path("models/cat-dog-vit")

# CHANGE THIS
# Example:
# "dkout/cat-dog-vit"
HF_REPO_ID = "dkoutzia97/cat-dog-vit"


# ============================================================
# LOAD LOCAL MODEL
# ============================================================

print("Loading local model...")

model = ViTForImageClassification.from_pretrained(
    MODEL_DIR
)

processor = ViTImageProcessor.from_pretrained(
    MODEL_DIR
)

print("Local model loaded.")

print("Model labels:")
print(model.config.id2label)


# ============================================================
# PUSH MODEL
# ============================================================

print()
print(f"Uploading model to: {HF_REPO_ID}")
print()

model.push_to_hub(
    HF_REPO_ID,
    commit_message="Upload trained cat-dog ViT model"
)

print("Model uploaded.")


# ============================================================
# PUSH IMAGE PROCESSOR
# ============================================================

processor.push_to_hub(
    HF_REPO_ID,
    commit_message="Upload ViT image processor"
)

print("Image processor uploaded.")

print()
print("DONE")
print(f"Model: https://huggingface.co/{HF_REPO_ID}")