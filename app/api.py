from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.inference import predict_image


app = FastAPI(
    title="Cat vs Dog Classifier API",
    description=(
        "Cat vs dog image classification "
        "using a fine-tuned Vision Transformer."
    ),
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Cat vs Dog Classifier API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    if (
        file.content_type is None
        or not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image.",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    try:
        image = Image.open(
            BytesIO(contents)
        ).convert("RGB")

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded image.",
        )

    result = predict_image(image)

    return {
        "filename": file.filename,
        "prediction": result["label"],
        "confidence": round(
            result["confidence"],
            4,
        ),
        "probabilities": {
            "cat": round(
                result["probabilities"]["cat"],
                4,
            ),
            "dog": round(
                result["probabilities"]["dog"],
                4,
            ),
        },
    }