import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    Trainer,
    TrainingArguments,
)

from training.dataset import CatsDogsDataset


MODEL_PATH = "models/cat-dog-vit"
TEST_CSV = "data/test.csv"


def main():

    print("Loading trained model...")

    processor = (
        ViTImageProcessor
        .from_pretrained(MODEL_PATH)
    )

    model = (
        ViTForImageClassification
        .from_pretrained(MODEL_PATH)
    )

    test_df = pd.read_csv(
        TEST_CSV
    )

    test_dataset = CatsDogsDataset(
        test_df,
        processor,
        train=False,
    )

    evaluation_args = TrainingArguments(
        output_dir="outputs/evaluation",
        per_device_eval_batch_size=2,
        dataloader_num_workers=0,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=evaluation_args,
    )

    print(
        f"Test images: {len(test_dataset)}"
    )

    predictions = trainer.predict(
        test_dataset
    )

    y_true = predictions.label_ids

    y_pred = np.argmax(
        predictions.predictions,
        axis=1
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true,
        y_pred
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=["cat", "dog"],
        zero_division=0,
    )

    print("\n==============================")
    print("TEST RESULTS")
    print("==============================")

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1       : {f1:.4f}"
    )

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(report)

    pd.DataFrame([
        {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    ]).to_csv(
        "outputs/test_metrics.csv",
        index=False
    )

    np.savetxt(
        "outputs/confusion_matrix.csv",
        matrix,
        delimiter=",",
        fmt="%d"
    )

    print(
        "\nMetrics saved to outputs/"
    )


if __name__ == "__main__":
    main()