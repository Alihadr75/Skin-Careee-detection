import streamlit as st
import joblib
import pandas as pd
# Skin lesion photo upload
st.subheader("Upload Skin Lesion Photo")

uploaded_image = st.file_uploader(
    "Choose a skin lesion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:
    st.image(
        uploaded_image,
        caption="Uploaded Skin Lesion Image",
        use_container_width=True
    )

    st.success("Image uploaded successfully!")
from pathlib import Path

st.set_page_config(
    page_title="Skin Lesion Classification Demo",
    page_icon="🔬",
    layout="centered",
)

MODEL_PATH = Path(__file__).parent / "best_skin_cancer_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("🔬 Skin Lesion Classification")
st.caption("Educational machine-learning demonstration")

st.warning(
    "Research/educational use only. This model was trained on synthetic data, "
    "not clinically verified patient data. It is not a medical device, cannot "
    "diagnose skin cancer, and must not be used to make healthcare decisions. "
    "A qualified dermatologist should assess any concerning lesion."
)

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

st.subheader("Enter the lesion characteristics")
st.write("The model accepts structured features, not an image upload.")

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=35, step=1)
    sex = st.selectbox("Sex", ["female", "male"])
    site = st.selectbox(
        "Anatomical site",
        ["head/neck", "lower extremity", "palms/soles", "torso", "upper extremity"],
    )
    asymmetry = st.selectbox("Asymmetry", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    border_irregularity = st.selectbox(
        "Border irregularity", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
    )
    color_variation = st.selectbox(
        "Color variation", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
    )
    diameter_mm = st.number_input(
        "Diameter (mm)", min_value=0.0, max_value=100.0, value=5.0, step=0.1
    )
    evolution = st.selectbox(
        "Has the lesion changed over time?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
    )

    submitted = st.form_submit_button("Run model demonstration", type="primary")

if submitted:
    # This mirrors pandas.get_dummies(..., drop_first=True) from the training script.
    row = {
        "age": int(age),
        "asymmetry": int(asymmetry),
        "border_irregularity": int(border_irregularity),
        "color_variation": int(color_variation),
        "diameter_mm": float(diameter_mm),
        "evolution": int(evolution),
        "anatom_site_lower extremity": int(site == "lower extremity"),
        "anatom_site_palms/soles": int(site == "palms/soles"),
        "anatom_site_torso": int(site == "torso"),
        "anatom_site_upper extremity": int(site == "upper extremity"),
        "sex_male": int(sex == "male"),
    }

    expected_features = list(getattr(model, "feature_names_in_", []))
    if not expected_features:
        st.error("The saved model does not expose its expected feature names.")
        st.stop()

    input_df = pd.DataFrame([row]).reindex(columns=expected_features, fill_value=0)

    try:
        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        class_index = list(model.classes_).index(prediction)
        confidence = float(probabilities[class_index]) * 100

        st.subheader("Model output")
        if prediction == 1:
            st.error("Class 1: model output = Malignant")
        else:
            st.success("Class 0: model output = Benign")

        st.metric("Model probability for the predicted class", f"{confidence:.1f}%")
        st.caption(
            "This is the model's probability estimate on synthetic training data, "
            "not a clinical probability or a measure of diagnostic certainty."
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.divider()
st.caption(
    "Do not use this demonstration to diagnose, rule out, or treat any condition. "
    "Seek professional medical advice for a changing, bleeding, painful, or otherwise "
    "concerning skin lesion."
)
