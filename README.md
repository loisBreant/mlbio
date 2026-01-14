# Classificateur de Lésions Cutanées (Skin Lesion Classifier)

Ce projet est une application de Deep Learning destinée à la classification automatique de lésions cutanées à partir d'images dermatoscopiques. Elle utilise un modèle **ResNet18** (réseau de neurones convolutif) pré-entraîné et fine-tuné, et intègre une interface utilisateur interactive basée sur **Streamlit**.

De plus, l'application fournit des explications visuelles pour ses prédictions grâce à **LIME** (Local Interpretable Model-agnostic Explanations), permettant de comprendre quelles zones de l'image ont influencé la décision du modèle.

## Fonctionnalités

*   **Classification Multi-classes** : Capable de détecter 7 types de lésions cutanées :
    *   Actinic keratoses (akiec)
    *   Basal cell carcinoma (bcc)
    *   Benign keratosis-like lesions (bkl)
    *   Dermatofibroma (df)
    *   Melanocytic nevi (nv)
    *   Melanoma (mel)
    *   Vascular lesions (vasc)
*   **Interface Web Interactive** : Upload d'images facile et visualisation des résultats en temps réel.
*   **Explicabilité (XAI)** : Intégration de LIME pour visualiser les superpixels qui contribuent positivement à la prédiction (contours jaunes sur l'image).
*   **Métriques de Confiance** : Affichage de la probabilité de la classe prédite et de la distribution des probabilités pour toutes les classes.

## Prérequis

*   Python 3.13 ou supérieur
*   Un gestionnaire de paquets comme `uv` (recommandé) ou `pip`.
*   Un GPU est recommandé pour l'entraînement et l'inférence rapide, mais le code fonctionne également sur CPU.

## Installation

Ce projet utilise `uv` pour la gestion des dépendances, assurant des installations rapides et fiables.

1.  **Cloner le dépôt :**
    ```bash
    git clone <votre-url-de-repo>
    cd mlbio
    ```

2.  **Installer les dépendances :**

    Si vous utilisez `uv` :
    ```bash
    uv sync
    ```

    Ou avec `pip` standard (en utilisant le fichier `pyproject.toml`) :
    ```bash
    pip install .
    ```

## Utilisation

### Lancer l'application Web

Pour démarrer l'interface Streamlit :

```bash
streamlit run streamlit_app.py
# ou avec uv
uv run streamlit run streamlit_app.py
```

L'application s'ouvrira dans votre navigateur par défaut (généralement à l'adresse `http://localhost:8501`).

### Entraînement du Modèle

Le modèle est entraîné via le notebook Jupyter `train.ipynb`. Ce notebook contient les étapes de :
1.  Préparation des données (Data Loading & Augmentation).
2.  Chargement du modèle ResNet18 pré-entraîné.
3.  Modification de la dernière couche pour correspondre aux 7 classes.
4.  Boucle d'entraînement et de validation.
5.  Sauvegarde du modèle dans `models/skin_lesion_resnet18.pth`.

Si vous souhaitez réentraîner le modèle, assurez-vous d'avoir le dataset complet ou utilisez le sous-ensemble fourni dans `dataset_subset/`.

## Structure du Projet

*   `streamlit_app.py` : Le point d'entrée de l'application web. Contient la logique d'inférence et d'interface.
*   `train.ipynb` : Notebook Jupyter utilisé pour l'entraînement du modèle.
*   `models/` : Contient les poids du modèle entraîné (`skin_lesion_resnet18.pth`).
*   `dataset_subset/` : Un sous-ensemble d'images pour tester ou lancer l'entraînement rapidement (organisé par classe).
*   `pyproject.toml` / `uv.lock` : Fichiers de configuration et de verrouillage des dépendances Python.
*   `runs/` : Logs et résultats d'expériences (probablement TensorBoard).

## Note sur les Données

Les données semblent provenir du dataset **HAM10000** (Human Against Machine with 10000 training images), une grande collection d'images dermatoscopiques multi-sources de lésions pigmentées courantes.
