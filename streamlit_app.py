import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

st.set_page_config(page_title="Skin Lesion Classifier", layout="wide")

classes = [
    'Actinic keratoses',
    'Basal cell carcinoma',
    'Benign keratosis-like lesions',
    'Dermatofibroma',
    'Melanocytic nevi',
    'Melanoma',
    'Vascular lesions'
]
num_classes = len(classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/skin_lesion_resnet18.pth"

@st.cache_resource
def load_model():
    try:
        model = models.resnet18(weights=None) 
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
        map_location = torch.device('cpu') if not torch.cuda.is_available() else None
        
        state_dict = torch.load(model_path, map_location=map_location)
        model.load_state_dict(state_dict)
        
        model = model.to(device)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at: {model_path}. Please run the training notebook first.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_transform():
    """
    Returns the validation transform pipeline used in training.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def preprocess_image(image):
    transform = get_transform()
    return transform(image).unsqueeze(0)

def batch_predict(images, model):
    model.eval()
    transform = get_transform()
    
    batch_tensors = []
    for img_np in images:
        img_pil = Image.fromarray(img_np.astype('uint8'))
        batch_tensors.append(transform(img_pil))
    
    batch = torch.stack(batch_tensors, dim=0).to(device)
    
    with torch.no_grad():
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        
    return probs.detach().cpu().numpy()

def main():
    st.title("Skin Lesion Classification")
    st.markdown("This app demonstrates the **ResNet18** model trained to classify skin lesions.")

    with st.expander("Classes"):
        st.markdown("""
        * Actinic keratoses
        * Basal cell carcinoma
        * Benign keratosis-like lesions
        * Dermatofibroma
        * Melanocytic nevi
        * Melanoma
        * Vascular lesions
        """)

    model = load_model()
    if model is None:
        return

    uploaded_file = st.file_uploader("Choose a dermatoscopic image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        image = Image.open(uploaded_file).convert('RGB')
        with col1:
            st.subheader("Input Image")
            st.image(image, caption='Uploaded Image', use_container_width=True)

        img_tensor = preprocess_image(image).to(device)
        
        with st.spinner("Analyzing..."):
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = F.softmax(outputs, dim=1)
            
            probs_np = probs.cpu().numpy()[0]
            prediction_idx = np.argmax(probs_np)
            prediction_class = classes[prediction_idx]
            confidence = probs_np[prediction_idx]

        with col2:
            st.subheader("Prediction Results")
            st.success(f"**Prediction: {prediction_class}**")
            st.metric(label="Confidence", value=f"{confidence:.2%}")
            
            st.write("**Probability Distribution:**")
            prob_dict = {cls: float(p) for cls, p in zip(classes, probs_np)}
            st.bar_chart(prob_dict)

        st.write("---")
        st.subheader("Explainability (LIME)")
        st.write("Click the button below to generate a LIME explanation. This highlights the regions of the image that contributed most to the prediction.")
        
        if st.button("Generate LIME Explanation"):
            with st.spinner("Generating LIME explanation... (This might take a few seconds)"):
                try:
                    explainer = lime_image.LimeImageExplainer()
                    
                    img_resized = image.resize((224, 224))
                    img_numpy = np.array(img_resized)
                    
                    def predict_fn(images):
                        return batch_predict(images, model)

                    explanation = explainer.explain_instance(
                        img_numpy, 
                        predict_fn, 
                        top_labels=5, 
                        hide_color=0, 
                        num_samples=2000
                    )

                    temp, mask = explanation.get_image_and_mask(
                        explanation.top_labels[0], 
                        positive_only=True, 
                        num_features=5, 
                        hide_rest=False
                    )

                    img_to_show = temp.copy()
                    if img_to_show.max() > 1:
                        img_to_show = img_to_show / 255.0
                        
                    img_boundary = mark_boundaries(img_to_show, mask, color=(1, 1, 0)) # Yellow boundaries

                    c1, c2, c3 = st.columns([1, 1, 1])
                    with c2:
                        st.image(img_boundary, caption=f"LIME Explanation for '{classes[explanation.top_labels[0]]}'", use_container_width=True)
                    st.info("The yellow boundaries highlight the superpixels that positively influenced the classifier's decision.")
                    
                except Exception as e:
                    st.error(f"An error occurred during LIME explanation: {e}")

if __name__ == "__main__":
    main()
