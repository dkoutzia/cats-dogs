# Cats & Dogs Image Classifier

A simple image classification application that distinguishes between cats and dogs using a fine-tuned Vision Transformer (ViT). The project includes dataset preparation and filtering, model training and evaluation, a FastAPI inference API, a Streamlit web interface, and Docker deployment.

## 1. Dataset Processing
a little bit more than **24,000 images**. The intended class is determined from the filename/index rather than from visual inspection:

* Images `0–12499` → **cat**
* Images `12500–24999` → **dog**

The original labels are therefore preserved and are not changed based on the filtering model.

Before training, the images are checked for integrity and processed using a dedicated pretrained model as a filter (`openai/clip-vit-base-patch32`) and a condition where we only accept image files. An image is retained only when the pretrained model agrees with the expected filename-based label and its prediction has sufficient confidence. Images that fail this semantic consistency check are rejected and recorded separately, while corrupted/unreadable images are also recorded.

The resulting clean dataset is split using a **stratified 70/15/15 train/validation/test split** with a fixed random seed of **42**, preserving the cat/dog class distribution across the three subsets.

The raw images are kept separate from the processed dataset, while rejected and corrupted images are logged for traceability.

## 2. Model Training and Evaluation

The final classifier is based on the pretrained:

`google/vit-base-patch16-224-in21k`

The Vision Transformer was fine-tuned as a binary image classifier for the two classes, **cat** and **dog**. Images are processed using the corresponding ViT image processor before being passed to the model.

The model was evaluated on the held-out test set using a confusion matrix and standard classification metrics.

### Test-set results

| Metric    |        Value |
| --------- | -----------: |
| Accuracy  | 0.9982793232004589
 |
| Precision | 0.9988597491448119 |
| Recall    | 0.997722095671981 |
| F1-score  | 0.9982905982905983 |

### Confusion Matrix

```text
                 Predicted
                 Cat     Dog

Actual Cat       1729     2
Actual Dog       4     1752
```

The evaluation is performed on images that were not used during model training, providing an estimate of the classifier's generalization to unseen images.

## 3. Image Filtering Before Classification

For a production system, an additional input-validation stage can be placed before the final cat/dog classifier.

The purpose is to prevent unrelated images, such as cars, people, landscapes, or other objects, from being forced into one of the two classes.

### Input Validation for Uploaded Images

There are two possible approaches for preventing unrelated images from being classified as cats or dogs.

**Approach 1 — Separate pretrained model as a semantic filter**

A separate pretrained cat-vs-dog model can be used as an initial semantic filter. The uploaded image is first passed through this model, and the image is rejected if neither class has sufficient confidence or if the prediction confidence/margin is below a predefined threshold.

For example:

```text
if max(P(cat), P(dog)) < threshold:
    reject image
else:
    classify image
```

This separates **input validation** from the final fine-tuned classifier and can reduce the risk of unrelated images being classified as cats or dogs.

**Approach 2 — Use the final classifier with a high confidence threshold**

Alternatively, the fine-tuned cat-vs-dog classifier itself can be used for input validation. If the model's highest predicted probability is below a sufficiently high threshold, the image is rejected rather than returning a cat/dog prediction.

For example:

```text
if max(P(cat), P(dog)) < high_threshold:
    reject image
else:
    accept prediction
```

This approach does not require an additional model and keeps the validation process within the same classification pipeline. The threshold should be selected using a validation set containing both valid cat/dog images and representative unrelated or out-of-distribution images.


## 4. FastAPI and Streamlit

The project exposes the trained model through a **FastAPI REST API**. The `/predict` endpoint accepts an uploaded image and returns the predicted class, confidence, and class probabilities.

FastAPI provides a model-serving interface that can be consumed by other applications or services, making the classifier independent from the user interface.

A **Streamlit web application** provides a simple interface where users can upload an image and see the prediction and class probabilities.

The Docker deployment runs both components:

```text
                    Docker Container
                         │
             ┌───────────┴───────────┐
             │                       │
        FastAPI :8000          Streamlit :7860
             │                       │
             └──────────┬────────────┘
                        │
                  ViT classifier
```

The Streamlit application can also perform inference directly using the same inference module. This allows the Streamlit Cloud deployment to operate without requiring a separate externally hosted FastAPI server.

## 5. Docker

The complete local application is containerized using Docker. The Docker image contains the Python environment, PyTorch, Transformers, FastAPI, Streamlit, and the required application code.

The trained model is **not stored in the Git repository**. Instead, the inference code downloads the final model from Hugging Face:

`dkoutzia97/cat-dog-vit`

This keeps the repository lightweight and allows the same model to be used by both the API and the web application.

### Run locally with Docker

Build the image:

```bash
docker build -t catdog:latest .
```

Run the application:

```bash
docker run --rm --gpus all -p 8000:8000 -p 7860:7860 catdog:latest
```

The FastAPI Swagger interface is then available at:

```text
http://localhost:8000/docs
```

The Streamlit application is available at:

```text
http://localhost:7860
```

The Docker container starts both services automatically using Supervisor.

### Run FastAPI without Docker

After installing the dependencies:

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

The API can then be accessed through:

```text
http://localhost:8000/docs
```

## 6. Online Demo

The Streamlit application is also deployed through Streamlit Community Cloud, allowing the application to be tested remotely without requiring the developer's computer or local Docker container to be running.

**Live application:**
https://cats-dogs-gu239nm63zdnthm8csgwyy.streamlit.app/

**Source code:**
https://github.com/dkoutzia/cats-dogs
