import os

import torch
from PIL import Image
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
)

MODEL_ID = os.getenv(
    "MODEL_ID",
    "dkoutzia97/cat-dog-vit",
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Loading model from Hugging Face: {MODEL_ID}")
print(f"Using device: {DEVICE}")

processor = ViTImageProcessor.from_pretrained(MODEL_ID)
model = ViTForImageClassification.from_pretrained(MODEL_ID)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")
print("Labels:", model.config.id2label)


@torch.no_grad()
def predict_image(image: Image.Image):

    image = image.convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    predicted_id = probabilities.argmax().item()

    predicted_label = model.config.id2label[
        predicted_id
    ]

    confidence = probabilities[
        predicted_id
    ].item()

    cat_id = model.config.label2id["cat"]
    dog_id = model.config.label2id["dog"]

    cat_probability = probabilities[cat_id].item()
    dog_probability = probabilities[dog_id].item()

    return {
        "label": predicted_label,
        "confidence": confidence,
        "probabilities": {
            "cat": cat_probability,
            "dog": dog_probability,
        },
    }