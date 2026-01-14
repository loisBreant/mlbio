import kagglehub
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn.functional as F

# Pour LIME
from lime import lime_image
from skimage.segmentation import mark_boundaries

# ==========================================
# 1. CONFIGURATION & CONSTANTES
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation du device : {DEVICE}")

SUBSET_SIZE = 2000  # <--- CONSTANTE: Nombre d'images à utiliser (sur 10000)
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

# ==========================================
# 2. CHARGEMENT DES DONNÉES (HAM10000)
# ==========================================
print("\n--- Téléchargement du Dataset ---")
# Téléchargement via KaggleHub
path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
print("Chemin du dataset:", path)

# Organisation des chemins de fichiers
# Le dataset est souvent divisé en plusieurs dossiers, on rassemble tout.
image_paths_dict = {}
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.jpg'):
            # On mappe l'ID de l'image (sans extension) vers son chemin complet
            image_id = os.path.splitext(file)[0]
            image_paths_dict[image_id] = os.path.join(root, file)

# Lecture du CSV de métadonnées
metadata_path = os.path.join(path, 'HAM10000_metadata.csv')
df = pd.read_csv(metadata_path)

# Ajout de la colonne 'path' au dataframe
df['path'] = df['image_id'].map(image_paths_dict.get)

# Nettoyage (au cas où certaines images ne sont pas trouvées)
df = df.dropna(subset=['path'])

# Dictionnaire pour rendre les labels lisibles (Optionnel mais pratique)
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
df['cell_type'] = df['dx'].map(lesion_type_dict.get)

# Encodage des labels (texte -> 0, 1, 2...)
le = LabelEncoder()
df['label'] = le.fit_transform(df['cell_type'])
NUM_CLASSES = len(le.classes_)

# ==========================================
# 3. CRÉATION DU SUBSET & DATASETS
# ==========================================
print(f"\n--- Création du Subset ({SUBSET_SIZE} images) ---")
# On mélange et on prend juste le nombre défini par la constante
if SUBSET_SIZE < len(df):
    df_subset = df.sample(n=SUBSET_SIZE, random_state=42).reset_index(drop=True)
else:
    df_subset = df

# Split Train / Test
train_df, test_df = train_test_split(df_subset, test_size=0.2, stratify=df_subset['label'], random_state=42)

# Définition des transformations (Prétraitement pour ResNet)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), # Augmentation de données simple
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Classe Dataset PyTorch personnalisée
class SkinDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['path']
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(int(self.df.iloc[idx]['label']), dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

train_dataset = SkinDataset(train_df, transform=data_transforms['train'])
test_dataset = SkinDataset(test_df, transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 4. MODÈLE (TRANSFER LEARNING)
# ==========================================
print("\n--- Chargement du modèle ResNet18 pré-entraîné ---")
model = models.resnet18(pretrained=True)

# On gèle les poids (optionnel, pour aller plus vite on ne ré-entraîne que la fin)
for param in model.parameters():
    param.requires_grad = False

# Remplacement de la dernière couche (Fully Connected)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(DEVICE)

# Configuration de l'entraînement
criterion = nn.CrossEntropyLoss()
# On optimise seulement la dernière couche
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# ==========================================
# 5. ENTRAÎNEMENT
# ==========================================
print("\n--- Début de l'entraînement ---")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Acc: {epoch_acc:.2f}%")

print("Entraînement terminé.")

# ==========================================
# 6. INTERPRÉTABILITÉ AVEC LIME
# ==========================================
print("\n--- Démarrage de LIME ---")

# Fonction de prédiction spéciale pour LIME
# LIME passe une image numpy (H, W, 3), il faut la convertir en Tensor PyTorch
def batch_predict(images):
    model.eval()
    batch = torch.stack([data_transforms['val'](Image.fromarray(i)) for i in images], dim=0)
    batch = batch.to(DEVICE)
    
    with torch.no_grad():
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        
    return probs.detach().cpu().numpy()

# Initialisation de l'explainer
explainer = lime_image.LimeImageExplainer()

# Prendre une image au hasard du set de test pour l'expliquer
idx = np.random.randint(0, len(test_dataset))
img_tensor, label_idx = test_dataset[idx]

# Convertir le tenseur en image numpy pour LIME (denormalization)
# Note: L'image passée à LIME doit ressembler à une photo normale
img_numpy = np.array(Image.open(test_df.iloc[idx]['path']).convert('RGB').resize((224, 224)))

print(f"Explication de l'image {idx}. Vraie classe : {le.classes_[label_idx]}")

# Génération de l'explication
explanation = explainer.explain_instance(
    img_numpy, 
    batch_predict, # Notre fonction de prédiction
    top_labels=5, 
    hide_color=0, 
    num_samples=1000 # Nombre de perturbations
)

# Visualisation
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0], 
    positive_only=True, 
    num_features=5, 
    hide_rest=False
)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_numpy)
plt.title("Image Originale")
plt.axis('off')

plt.subplot(1, 2, 2)
# mark_boundaries dessine les contours des superpixels importants
plt.imshow(mark_boundaries(temp / 255 + 0.5, mask)) # Petit ajustement de luminosité pour l'affichage
plt.title(f"Explication LIME\n(Prédiction: {le.classes_[explanation.top_labels[0]]})")
plt.axis('off')
plt.tight_layout()
plt.show()