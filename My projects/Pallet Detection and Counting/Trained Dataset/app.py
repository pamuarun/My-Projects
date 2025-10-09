import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from roboflow import Roboflow
import tempfile
import os

# Load Roboflow model
@st.cache_resource
def load_model():
    rf = Roboflow(api_key="1yES149xy1ye6aInxjo2")
    project = rf.workspace().project("my-first-project-a7fqy")
    model = project.version(9).model
    return model

model = load_model()

# Streamlit UI
st.title("📦 Pallet Detection App (Roboflow SDK)")
st.markdown("Upload an image to detect pallets using your trained model.")

# Upload image
uploaded_file = st.file_uploader("📤 Upload a pallet image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Create temp dir and file
    temp_dir = tempfile.mkdtemp()
    image_path = os.path.join(temp_dir, "uploaded_image.png")

    # Open and convert image
    image = Image.open(uploaded_file).convert("RGB")
    image.save(image_path, format="PNG")  # Save as PNG to avoid JPEG+alpha error

    # Run inference
    results = model.predict(image_path)
    predictions = results.json()['predictions']

    # Draw on image
    draw = ImageDraw.Draw(image)

    # Load font
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=22)
    except:
        font = ImageFont.load_default()

    # Draw bounding boxes and labels
    for pred in predictions:
        x = pred['x']
        y = pred['y']
        w = pred['width']
        h = pred['height']
        conf = pred['confidence']
        class_name = pred['class']

        x0, y0 = x - w / 2, y - h / 2
        x1, y1 = x + w / 2, y + h / 2

        # Draw box
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)

        label = f"{class_name} {conf:.2f}"
        text_size = draw.textbbox((0, 0), label, font=font)
        text_width = text_size[2] - text_size[0]
        text_height = text_size[3] - text_size[1]

        # Draw background for label
        draw.rectangle(
            [(x0, y0), (x0 + text_width + 6, y0 + text_height + 4)],
            fill="black"
        )
        draw.text((x0 + 3, y0 + 2), label, fill="white", font=font)

    # Display total count and image
    st.success(f"🔢 Total pallets detected: {len(predictions)}")
    st.image(image, caption="🖼️ Detected Pallets with Bounding Boxes", use_column_width=True)
