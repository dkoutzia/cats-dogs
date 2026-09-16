import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score

from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    Trainer,
    TrainingArguments,
)

from training.dataset import (
    CatsDogsDataset,
    LABEL2ID,
    ID2LABEL,
)


MODEL_NAME = "google/vit-base-patch16-224-in21k"

TRAIN_CSV = "data/train.csv"
VALIDATION_CSV = "data/validation.csv"

OUTPUT_DIR = "models/cat-dog-vit"


def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = np.argmax(
        predictions,
        axis=1
    )

    return {
        "accuracy": accuracy_score(
            labels,
            predictions
        )
    }


def main():

    print("PyTorch:", torch.__version__)

    print(
        "CUDA available:",
        torch.cuda.is_available()
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    train_df = pd.read_csv(
        TRAIN_CSV
    )

    validation_df = pd.read_csv(
        VALIDATION_CSV
    )

    print(
        f"Training images: {len(train_df)}"
    )

    print(
        f"Validation images: {len(validation_df)}"
    )

    processor = (
        ViTImageProcessor
        .from_pretrained(MODEL_NAME)
    )

    train_dataset = CatsDogsDataset(
        train_df,
        processor,
        train=True,
    )

    validation_dataset = CatsDogsDataset(
        validation_df,
        processor,
        train=False,
    )

    model = (
        ViTForImageClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=2,
            label2id=LABEL2ID,
            id2label=ID2LABEL,
            ignore_mismatched_sizes=True,
        )
    )
    # Freeze the entire ViT backbone
    for param in model.vit.parameters():
        param.requires_grad = False

    # Classification head remains trainable
    for param in model.classifier.parameters():
        param.requires_grad = True
    training_args = TrainingArguments(

        output_dir=OUTPUT_DIR,

        num_train_epochs=3,

        per_device_train_batch_size=6,

        per_device_eval_batch_size=6,

        gradient_accumulation_steps=3,

        learning_rate=2e-5,

        weight_decay=0.01,

        evaluation_strategy="epoch",

        save_strategy="epoch",

        load_best_model_at_end=True,

        metric_for_best_model="accuracy",

        greater_is_better=True,

        save_total_limit=2,

        logging_steps=25,

        fp16=torch.cuda.is_available(),

        dataloader_num_workers=0,

        report_to="none",
    )

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=validation_dataset,

        compute_metrics=compute_metrics,
    )

    print("\nStarting training...\n")

    trainer.train()

    trainer.save_model(
        OUTPUT_DIR
    )

    processor.save_pretrained(
        OUTPUT_DIR
    )

    print(
        f"\nModel saved to {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()