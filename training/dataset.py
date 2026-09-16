import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


LABEL2ID = {
    "cat": 0,
    "dog": 1,
}

ID2LABEL = {
    0: "cat",
    1: "dog",
}


class CatsDogsDataset(Dataset):

    def __init__(
        self,
        dataframe,
        processor,
        train=False,
    ):

        self.dataframe = dataframe.reset_index(
            drop=True
        )

        self.processor = processor

        if train:

            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    224,
                    scale=(0.8, 1.0)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
            ])

        else:

            self.transform = transforms.Compose([
                transforms.Resize(
                    (224, 224)
                ),
            ])

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image = Image.open(
            row["image_path"]
        ).convert("RGB")

        image = self.transform(image)

        encoded = self.processor(
            images=image,
            return_tensors="pt"
        )

        return {
            "pixel_values": encoded[
                "pixel_values"
            ].squeeze(0),

            "labels": torch.tensor(
                int(row["label_id"]),
                dtype=torch.long
            ),
        }