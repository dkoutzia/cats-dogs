from pathlib import Path
import argparse
import shutil

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from transformers import CLIPProcessor, CLIPModel


# ============================================================
# CONFIGURATION
# ============================================================

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

LABEL2ID = {
    "cat": 0,
    "dog": 1,
}

ID2LABEL = {
    0: "cat",
    1: "dog",
}


# ============================================================
# DATASET LABELING
# ============================================================

def get_label_from_filename(image_number):
    """
    The filename determines the expected dataset label.

    0-12499       -> cat
    12500-24999   -> dog
    """

    if 0 <= image_number <= 12499:
        return "cat", 0

    if 12500 <= image_number <= 24999:
        return "dog", 1

    return None, None


# ============================================================
# COLLECT DATASET
# ============================================================

def collect_images(source_dir):
    source_dir = Path(source_dir)

    rows = []

    for image_path in sorted(source_dir.iterdir()):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        try:
            image_number = int(image_path.stem)
        except ValueError:
            continue

        label, label_id = get_label_from_filename(image_number)

        if label is None:
            continue

        rows.append(
            {
                "image_path": str(image_path.resolve()),
                "filename": image_path.name,
                "image_number": image_number,
                "label": label,
                "label_id": label_id,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# IMAGE INTEGRITY CHECK
# ============================================================

def validate_image_integrity(df):

    valid_rows = []
    corrupted_rows = []

    print()
    print("Checking image integrity...")

    for _, row in df.iterrows():

        image_path = Path(row["image_path"])

        try:

            # First verify file structure
            with Image.open(image_path) as image:
                image.verify()

            # Then actually decode the image
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image.load()

            valid_rows.append(row)

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as exc:

            corrupted_row = row.copy()
            corrupted_row["reason"] = str(exc)

            corrupted_rows.append(corrupted_row)

    return (
        pd.DataFrame(valid_rows),
        pd.DataFrame(corrupted_rows),
    )


# ============================================================
# CLIP MODEL
# ============================================================

class CLIPScreeningModel:

    def __init__(self, device):

        self.device = device

        print()
        print("Loading pretrained CLIP model...")
        print(CLIP_MODEL_NAME)

        self.processor = CLIPProcessor.from_pretrained(
            CLIP_MODEL_NAME
        )

        self.model = CLIPModel.from_pretrained(
            CLIP_MODEL_NAME
        )

        self.model.to(self.device)
        self.model.eval()

        self.prompts = [
            "a photo of a cat",
            "a photo of a dog",
            "a photo of something that is neither a cat nor a dog",
        ]

    @torch.no_grad()
    def predict(self, image):

        inputs = self.processor(
            text=self.prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        outputs = self.model(**inputs)

        logits = outputs.logits_per_image[0]

        probabilities = torch.softmax(
            logits,
            dim=-1,
        ).cpu().numpy()

        predicted_index = int(
            np.argmax(probabilities)
        )

        return {
            "cat_probability": float(probabilities[0]),
            "dog_probability": float(probabilities[1]),
            "neither_probability": float(probabilities[2]),
            "predicted_index": predicted_index,
        }


# ============================================================
# CLIP DATASET FILTERING
# ============================================================

def screen_dataset(
    df,
    device,
    threshold=0.65,
):

    clip_model = CLIPScreeningModel(device)

    accepted_rows = []
    rejected_rows = []

    total = len(df)

    print()
    print(f"Running CLIP semantic filtering on {total} images...")
    print()

    for index, (_, row) in enumerate(
        df.iterrows(),
        start=1,
    ):

        image_path = Path(
            row["image_path"]
        )

        try:

            with Image.open(image_path) as image:

                image = image.convert("RGB")

                result = clip_model.predict(
                    image
                )

            expected_label = row["label"]

            if expected_label == "cat":

                expected_probability = (
                    result["cat_probability"]
                )

                expected_index = 0

            else:

                expected_probability = (
                    result["dog_probability"]
                )

                expected_index = 1

            predicted_index = result[
                "predicted_index"
            ]

            accepted = True
            reasons = []

            # Reject if CLIP believes the image is neither.
            if predicted_index == 2:

                accepted = False

                reasons.append(
                    "clip_predicted_neither"
                )

            # Reject low confidence for expected class.
            if expected_probability < threshold:

                accepted = False

                reasons.append(
                    f"expected_probability_below_{threshold}"
                )

            # Reject disagreement between filename
            # and CLIP semantic prediction.
            if predicted_index != expected_index:

                accepted = False

                reasons.append(
                    "clip_prediction_disagrees_with_filename"
                )

            output_row = row.copy()

            output_row[
                "clip_cat_probability"
            ] = result["cat_probability"]

            output_row[
                "clip_dog_probability"
            ] = result["dog_probability"]

            output_row[
                "clip_neither_probability"
            ] = result["neither_probability"]

            if predicted_index == 0:

                clip_label = "cat"

            elif predicted_index == 1:

                clip_label = "dog"

            else:

                clip_label = "neither"

            output_row[
                "clip_predicted_label"
            ] = clip_label

            output_row[
                "clip_expected_probability"
            ] = expected_probability

            if accepted:

                output_row["screening_status"] = "accepted"

                accepted_rows.append(
                    output_row
                )

            else:

                output_row[
                    "screening_status"
                ] = "rejected"

                output_row["reason"] = ";".join(
                    reasons
                )

                rejected_rows.append(
                    output_row
                )

        except Exception as exc:

            output_row = row.copy()

            output_row[
                "screening_status"
            ] = "rejected"

            output_row[
                "reason"
            ] = f"clip_error: {exc}"

            rejected_rows.append(
                output_row
            )

        if index % 100 == 0 or index == total:

            print(
                f"Processed {index}/{total}"
            )

    return (
        pd.DataFrame(accepted_rows),
        pd.DataFrame(rejected_rows),
    )


# ============================================================
# BALANCED SPLIT
# ============================================================

def create_splits(
    df,
    seed=42,
):

    train_df, temporary_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label_id"],
        random_state=seed,
    )

    validation_df, test_df = train_test_split(
        temporary_df,
        test_size=0.50,
        stratify=temporary_df["label_id"],
        random_state=seed,
    )

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ============================================================
# COPY SPLIT IMAGES
# ============================================================

def copy_split_images(
    df,
    split_name,
    output_dir,
):

    output_dir = Path(output_dir)

    for label in ["cat", "dog"]:

        destination = (
            output_dir
            / split_name
            / f"{label}s"
        )

        destination.mkdir(
            parents=True,
            exist_ok=True,
        )

    new_paths = []

    for _, row in df.iterrows():

        source = Path(
            row["image_path"]
        )

        destination_dir = (
            output_dir
            / split_name
            / f"{row['label']}s"
        )

        destination = (
            destination_dir
            / source.name
        )

        shutil.copy2(
            source,
            destination,
        )

        new_paths.append(
            str(destination.resolve())
        )

    result = df.copy()

    result["image_path"] = new_paths

    return result


# ============================================================
# SAVE REJECTED IMAGES
# ============================================================

def save_rejected_images(
    df,
    output_dir,
):

    output_dir = Path(output_dir)

    rejected_dir = (
        output_dir / "rejected"
    )

    rejected_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    new_paths = []

    for _, row in df.iterrows():

        source = Path(
            row["image_path"]
        )

        destination = (
            rejected_dir / source.name
        )

        shutil.copy2(
            source,
            destination,
        )

        new_paths.append(
            str(destination.resolve())
        )

    result = df.copy()

    if len(result) > 0:

        result[
            "rejected_image_path"
        ] = new_paths

    return result


# ============================================================
# SUMMARY
# ============================================================

def create_summary(
    train_df,
    validation_df,
    test_df,
    rejected_df,
    corrupted_df,
):

    datasets = [
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df),
        ("rejected", rejected_df),
        ("corrupted", corrupted_df),
    ]

    rows = []

    for dataset_name, dataframe in datasets:

        if len(dataframe) == 0:

            rows.append(
                {
                    "dataset": dataset_name,
                    "total": 0,
                    "cats": 0,
                    "dogs": 0,
                }
            )

            continue

        rows.append(
            {
                "dataset": dataset_name,
                "total": len(dataframe),
                "cats": int(
                    (
                        dataframe["label"]
                        == "cat"
                    ).sum()
                ),
                "dogs": int(
                    (
                        dataframe["label"]
                        == "dog"
                    ).sum()
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default="data/raw",
    )

    parser.add_argument(
        "--output",
        default="data",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
    )

    args = parser.parse_args()

    source_dir = Path(
        args.source
    )

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Collect images
    # --------------------------------------------------------

    df = collect_images(
        source_dir
    )

    print()
    print("==============================")
    print("INITIAL DATASET")
    print("==============================")

    print(
        f"Images found: {len(df)}"
    )

    if len(df) == 0:

        raise RuntimeError(
            "No valid numbered images were found."
        )

    print()
    print("Initial label distribution:")
    print(
        df["label"].value_counts()
    )

    # --------------------------------------------------------
    # 2. Image integrity
    # --------------------------------------------------------

    (
        valid_df,
        corrupted_df,
    ) = validate_image_integrity(df)

    print()
    print("==============================")
    print("IMAGE INTEGRITY")
    print("==============================")

    print(
        f"Valid: {len(valid_df)}"
    )

    print(
        f"Corrupted: {len(corrupted_df)}"
    )

    corrupted_df.to_csv(
        output_dir
        / "corrupted_images.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 3. CLIP
    # --------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        f"CLIP device: {device}"
    )

    if device == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    (
        accepted_df,
        rejected_df,
    ) = screen_dataset(
        valid_df,
        device=device,
        threshold=args.threshold,
    )

    # --------------------------------------------------------
    # 4. Save rejected
    # --------------------------------------------------------

    rejected_df = save_rejected_images(
        rejected_df,
        output_dir,
    )

    rejected_df.to_csv(
        output_dir
        / "rejected_images.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 5. Create balanced splits
    # --------------------------------------------------------

    (
        train_df,
        validation_df,
        test_df,
    ) = create_splits(
        accepted_df
    )

    # --------------------------------------------------------
    # 6. Copy images
    # --------------------------------------------------------

    train_df = copy_split_images(
        train_df,
        "train",
        output_dir,
    )

    validation_df = copy_split_images(
        validation_df,
        "validation",
        output_dir,
    )

    test_df = copy_split_images(
        test_df,
        "test",
        output_dir,
    )

    # --------------------------------------------------------
    # 7. Save CSVs
    # --------------------------------------------------------

    train_df.to_csv(
        output_dir / "train.csv",
        index=False,
    )

    validation_df.to_csv(
        output_dir / "validation.csv",
        index=False,
    )

    test_df.to_csv(
        output_dir / "test.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 8. Summary
    # --------------------------------------------------------

    summary = create_summary(
        train_df,
        validation_df,
        test_df,
        rejected_df,
        corrupted_df,
    )

    summary.to_csv(
        output_dir
        / "split_summary.csv",
        index=False,
    )

    print()
    print("==============================")
    print("FINAL DATASET SUMMARY")
    print("==============================")

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Dataset preparation completed."
    )


if __name__ == "__main__":
    main()