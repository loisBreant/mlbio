---
theme: seriph
class: text-center
highlighter: shiki
lineNumbers: false
info: |
  ## Skin Disease Classification
  Automated diagnosis for dermatology using Deep Learning.
drawings:
  persist: false
transition: slide-left
title: Skin Disease Classification
---
<!-- 
<video autoplay loop muted playsinline class="absolute top-0 left-0 w-full h-full object-cover -z-10"><source src="/assets/background.mp4" type="video/mp4" /></video> -->

# Skin Disease Classification
## Automated Diagnosis for Dermatology

<div class="abs-br m-6 flex gap-2">
  <button @click="$slidev.nav.openInEditor()" title="Open in Editor" class="text-xl slidev-icon-btn opacity-50 !border-none !hover:text-white">
    <carbon:edit />
  </button>
  <a href="https://github.com/slidevjs/slidev" target="_blank" alt="GitHub"
    class="text-xl slidev-icon-btn opacity-50 !border-none !hover:text-white">
    <carbon-logo-github />
  </a>
</div>

<!-- 
  Biomedical Theme Note:
  Using a clean, serif-based theme conveys professionalism suitable for medical contexts.
  Background image is a placeholder for a medical/lab setting.
-->

---
layout: default
---

# Project Overview

<div class="grid grid-cols-2 gap-8">

<div>

### Context
Working for a company providing digital tools for dermatologists.

### Objective
Develop a machine learning model to automatically classify skin diseases from photographs.

### Goal
Assist medical professionals in early diagnosis and screening of skin conditions.

</div>



</div>

---

# Dataset: HAM10000

We utilize the **HAM10000** ("Human Against Machine with 10000 training images") dataset.

<div class="grid grid-cols-2 gap-8 mt-6">

<div>

- **Source**: Publicly available (Kaggle/Harvard).
- **Content**: Dermatoscopic images of pigmented skin lesions.
- **Classes**: 7 diagnostic categories (Melanoma, Nevi, etc.).
- **Subset**: We utilized a balanced subset of **2,000 images** for this development phase.

</div>

<div class="grid grid-cols-2 gap-2">
  <!-- PLACEHOLDERS FOR DATASET SAMPLES -->
  <div class="aspect-square bg-gray-200 flex items-center justify-center rounded text-xs text-gray-500">
    [Sample: Nevi]
  </div>
  <div class="aspect-square bg-gray-200 flex items-center justify-center rounded text-xs text-gray-500">
    [Sample: Melanoma]
  </div>
  <div class="aspect-square bg-gray-200 flex items-center justify-center rounded text-xs text-gray-500">
    [Sample: BKL]
  </div>
  <div class="aspect-square bg-gray-200 flex items-center justify-center rounded text-xs text-gray-500">
    [Sample: BCC]
  </div>
</div>

</div>

---
layout: default
---

# Methodology: Deep Learning Approach

We employed **Transfer Learning** to leverage pre-existing visual patterns.

### Model Architecture: ResNet18
- **Pre-trained**: On ImageNet (1M+ images).
- **Strategy**:
  1.  **Feature Extractor**: Frozen layers from ResNet18.
  2.  **Classifier**: Replaced the final Fully Connected layer to match our 7 classes.
  3.  **Optimization**: Fine-tuned using `CrossEntropyLoss` and `Adam` optimizer.

### Data Preprocessing
Standardization pipeline for robust model input:

```python {all|2-3|4-5|all}
transforms.Compose([
    transforms.Resize((224, 224)),      # Standardize input size
    transforms.RandomHorizontalFlip(),  # Augmentation
    transforms.RandomRotation(10),      # Augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.485,...], [0.229,...])
])
```

---

# Results & Performance

After training for **5 Epochs**:

<div class="grid grid-cols-2 gap-10 mt-8">

<div class="flex flex-col justify-center">

- **Accuracy**: **75.125%** on the validation set.
- **Loss**: **0.6** (MSE/CrossEntropy).
- **Inference Speed**: Real-time capable on standard hardware.

<div class="mt-4 p-4 bg-green-50 border-l-4 border-green-500 text-green-700">
  <p class="font-bold">Key Insight</p>
  <p>Transfer learning allowed for high accuracy despite a reduced dataset size of 2,000 images.</p>
</div>

</div>

<div class="bg-white p-4 rounded shadow-lg border border-gray-100 flex flex-col items-center justify-center">
  <!-- PLACEHOLDER FOR TRAINING CURVE -->
  <img src="/assets/training_curve.png" alt="Training Curves" class="w-full h-40 object-contain rounded mb-2"/>
  <p class="text-xs text-gray-500 mt-2">Training vs Validation Accuracy</p>
</div>

</div>

---

# Interpretability: LIME

To ensure trust in clinical settings, we use **LIME** (Local Interpretable Model-agnostic Explanations).

<div class="grid grid-cols-3 gap-4 mt-6">

<div class="col-span-1">
  <h3 class="text-lg font-bold mb-2">How it works</h3>
  <p class="text-sm opacity-80">LIME perturbs the input image to identify which superpixels most positively influence the model's prediction.</p>
</div>

<div class="col-span-2 flex gap-4 justify-center items-center">
    <div class="flex flex-col items-center">
        <!-- PLACEHOLDER FOR ORIGINAL IMAGE -->
        <div class="w-32 h-32 bg-gray-200 flex items-center justify-center rounded mb-2 text-xs">[Original]</div>
        <span class="text-xs font-bold">Input Lesion</span>
    </div>
    <carbon:arrow-right class="text-2xl opacity-50" />
    <div class="flex flex-col items-center">
        <!-- PLACEHOLDER FOR LIME EXPLANATION -->
        <div class="w-32 h-32 bg-yellow-100 border-2 border-yellow-400 flex items-center justify-center rounded mb-2 text-xs">[LIME Mask]</div>
        <span class="text-xs font-bold">Explanation</span>
    </div>
</div>

</div>

<div class="mt-4 text-center text-sm text-gray-600">
  <carbon:idea class="inline mr-1 text-yellow-500"/>
  Green regions indicate areas that the model used to identify the disease.
</div>

---
layout: center
class: text-center
---

# Conclusion

We developed a robust proof-of-concept for automated skin disease classification.

<div class="text-left max-w-lg mx-auto mt-8 space-y-2">
  <div class="flex items-center gap-2">
    <carbon:checkmark-filled class="text-green-500" />
    <span>Functional end-to-end ML pipeline</span>
  </div>
  <div class="flex items-center gap-2">
    <carbon:checkmark-filled class="text-green-500" />
    <span>75.125% Accuracy with Transfer Learning</span>
  </div>
  <div class="flex items-center gap-2">
    <carbon:checkmark-filled class="text-green-500" />
    <span>Explainable AI integration (LIME)</span>
  </div>
</div>

<div class="mt-12 text-sm opacity-50">
  Created with Slidev & PyTorch
</div>
