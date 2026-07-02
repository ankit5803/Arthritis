import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from src.model import KneeOAModel
import pandas as pd

# ---------------- PAGE ----------------
st.set_page_config(
    page_title="Knee Osteoarthritis AI",
    page_icon="🦴",
    layout="centered"
)

st.title("🦴 Knee Osteoarthritis Detection System")
st.markdown("Upload a knee X-ray image and the AI model will predict osteoarthritis severity.")

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = KneeOAModel()
    model.load_state_dict(
        torch.load("best_model.pth", map_location=device)
    )
    model.to(device)
    model.eval()
    return model

model = load_model()

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- LABELS ----------------
grade_labels = {
    0: "Grade 0",
    1: "Grade 1",
    2: "Grade 2",
    3: "Grade 3",
    4: "Grade 4"
}

grade_meanings = {
    0: "Healthy knee joint. No visible osteoarthritis.",
    1: "Possible early joint-space narrowing. Very mild changes.",
    2: "Mild osteoarthritis detected. Early cartilage degeneration.",
    3: "Moderate osteoarthritis. Clear structural degeneration visible.",
    4: "Severe osteoarthritis. Significant joint damage detected."
}

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Knee X-ray Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded X-ray",
        use_container_width=True
    )

    input_tensor = transform(image).unsqueeze(0).to(device)

    with st.spinner("Analyzing image..."):

        with torch.no_grad():

            output = model(input_tensor)

            probs = torch.softmax(output, dim=1)[0]

            confidence, pred = torch.max(probs, 0)

            pred_class = pred.item()
            confidence = confidence.item() * 100

    st.success("Analysis complete")

    # ---------------- MAIN RESULT ----------------
    st.subheader("Prediction")

    st.metric(
        label="Predicted Severity",
        value=grade_labels[pred_class]
    )

    st.metric(
        label="Confidence",
        value=f"{confidence:.2f}%"
    )

    # ---------------- INTERPRETATION ----------------
    st.subheader("Clinical Interpretation")

    st.info(
        grade_meanings[pred_class]
    )

    # ---------------- PROBABILITIES ----------------
    st.subheader("Prediction Confidence Across All Grades")

    df = pd.DataFrame({
        "Grade": ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"],
        "Probability": probs.cpu().numpy()
    })

    st.bar_chart(
        df.set_index("Grade")
    )

    # ---------------- DISCLAIMER ----------------
    st.markdown("---")

    st.warning(
        """
        **Medical Disclaimer**

        This system is an AI-assisted screening tool.

        Predictions should not be considered a medical diagnosis.

        Please consult a qualified orthopedic specialist for clinical evaluation.
        """
    )