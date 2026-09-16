import streamlit as st
from PIL import Image

from inference import predict_image

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱",
    layout="centered",
)

st.title("🐱 Cat vs Dog Classifier")

st.write(
    "Upload an image and the trained Vision Transformer "
    "will classify it as a cat or dog."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True,
    )

    if st.button("Classify image"):
        with st.spinner("Classifying..."):
            result = predict_image(image)

        prediction = result["label"]
        confidence = result["confidence"]

        if prediction.lower() == "cat":
            st.success(
                f"Prediction: CAT "
                f"({confidence * 100:.2f}% confidence)"
            )
        else:
            st.info(
                f"Prediction: DOG "
                f"({confidence * 100:.2f}% confidence)"
            )

        st.subheader("Class probabilities")

        probabilities = result["probabilities"]

        st.write(
            f"Cat: {probabilities['cat'] * 100:.2f}%"
        )
        st.progress(probabilities["cat"])

        st.write(
            f"Dog: {probabilities['dog'] * 100:.2f}%"
        )
        st.progress(probabilities["dog"])