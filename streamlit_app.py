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

# --- Configuration ---
st.set_page_config(page_title="Skin Lesion Classifier", layout="wide")

# --- Constants & Classes ---
# Classes are sorted alphabetically as LabelEncoder was used during training
CLASSES = [
    'Actinic keratoses',
    'Basal cell carcinoma',
    'Benign keratosis-like lesions',
    'Dermatofibroma',
    'Melanocytic nevi',
    'Melanoma',
    'Vascular lesions'
]
NUM_CLASSES = len(CLASSES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/skin_lesion_resnet18.pth"

# --- Model Loading ---
@st.cache_resource
def load_model():
    """
    Loads the ResNet18 model with the custom head and trained weights.
    """
    try:
        # Initialize architecture
        model = models.resnet18(weights=None) 
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
        
        # Load weights
        # Map to CPU if CUDA is not available
        map_location = torch.device('cpu') if not torch.cuda.is_available() else None
        
        # Load the state dictionary
        state_dict = torch.load(MODEL_PATH, map_location=map_location)
        model.load_state_dict(state_dict)
        
        model = model.to(DEVICE)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at: {MODEL_PATH}. Please run the training notebook first.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Preprocessing ---
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
    """
    Preprocesses a PIL image for the model.
    """
    transform = get_transform()
    return transform(image).unsqueeze(0) # Add batch dimension

# --- LIME Helper ---
def batch_predict(images, model):
    """
    Prediction function for LIME.
    images: numpy array of images (N, H, W, C)
    """
    model.eval()
    transform = get_transform()
    
    # Convert numpy images back to PIL for transformation
    batch_tensors = []
    for img_np in images:
        img_pil = Image.fromarray(img_np.astype('uint8'))
        batch_tensors.append(transform(img_pil))
    
    batch = torch.stack(batch_tensors, dim=0).to(DEVICE)
    
    with torch.no_grad():
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        
    return probs.detach().cpu().numpy()

# --- Main App ---
def main():
    st.title("🔎 Skin Lesion Classification")
    st.markdown("""
    This app demonstrates the **ResNet18** model trained to classify skin lesions.
    
    **Classes:**
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

    # File Uploader
    uploaded_file = st.file_uploader("Choose a dermatoscopic image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        # Display Image
        image = Image.open(uploaded_file).convert('RGB')
        with col1:
            st.subheader("Input Image")
            st.image(image, caption='Uploaded Image', use_container_width=True)

        # Inference
        img_tensor = preprocess_image(image).to(DEVICE)
        
        with st.spinner("Analyzing..."):
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = F.softmax(outputs, dim=1)
            
            probs_np = probs.cpu().numpy()[0]
            prediction_idx = np.argmax(probs_np)
            prediction_class = CLASSES[prediction_idx]
            confidence = probs_np[prediction_idx]

        # Results
        with col2:
            st.subheader("Prediction Results")
            st.success(f"**Prediction: {prediction_class}**")
            st.metric(label="Confidence", value=f"{confidence:.2%}")
            
            st.write("**Probability Distribution:**")
            # Create a dictionary for the bar chart
            prob_dict = {cls: float(p) for cls, p in zip(CLASSES, probs_np)}
            st.bar_chart(prob_dict)

        # LIME Explanation Section
        st.write("---")
        st.subheader("🔍 Explainability (LIME)")
        st.write("Click the button below to generate a LIME explanation. This highlights the regions of the image that contributed most to the prediction.")
        
        if st.button("Generate LIME Explanation"):
            with st.spinner("Generating LIME explanation... (This might take a few seconds)"):
                try:
                    explainer = lime_image.LimeImageExplainer()
                    
                    # LIME works with numpy arrays (H, W, C)
                    # Resize to model input size for consistency with training
                    img_resized = image.resize((224, 224))
                    img_numpy = np.array(img_resized)
                    
                    # Wrapper for prediction
                    def predict_fn(images):
                        return batch_predict(images, model)

                    # Run LIME
                    # num_samples=1000 is a tradeoff for speed (default is often 1000)
                    explanation = explainer.explain_instance(
                        img_numpy, 
                        predict_fn, 
                        top_labels=5, 
                        hide_color=0, 
                        num_samples=1000 
                    )

                    # Get image and mask for the top prediction
                    temp, mask = explanation.get_image_and_mask(
                        explanation.top_labels[0], 
                        positive_only=True, 
                        num_features=5, 
                        hide_rest=False
                    )

                    # Mark boundaries
                    # temp is the image, mask is the explanation region
                    # mark_boundaries expects float image [0, 1] if max > 1
                    img_to_show = temp.copy()
                    if img_to_show.max() > 1:
                        img_to_show = img_to_show / 255.0
                        
                    img_boundary = mark_boundaries(img_to_show, mask, color=(1, 1, 0)) # Yellow boundaries

                    st.image(img_boundary, caption=f"LIME Explanation for '{CLASSES[explanation.top_labels[0]]}'", use_container_width=True)
                    st.info("The yellow boundaries highlight the superpixels that positively influenced the classifier's decision.")
                    
                except Exception as e:
                    st.error(f"An error occurred during LIME explanation: {e}")

if __name__ == "__main__":
    main()
