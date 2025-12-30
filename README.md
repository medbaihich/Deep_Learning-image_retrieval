
# Deep Learning Image Retrieval System

## 📌 Project Overview

This project implements a content-based **Image Retrieval Engine** using **Deep Learning**. Unlike traditional keyword search, this system allows users to provide an input image (query) and retrieves the most visually and semantically similar images from the database.

The system leverages **Transfer Learning**, using a pre-trained Convolutional Neural Network (CNN) to extract "Deep Features" from images. These features represent high-level concepts (shapes, textures, objects) rather than simple pixel colors, enabling highly accurate similarity detection.

## 📂 Dataset

The project uses a subset of the **CIFAR-10** dataset.

* **Format:** Turi Create SFrame.
* **Content:** Small (32x32) color images categorized into classes such as `bird`, `cat`, `dog`, and `automobile`.
* **Key Columns:**
* `image`: The raw image data.
* `label`: The category of the image.
* `deep_features`: A dense vector representation of the image extracted from a CNN.



## 🛠 Technologies & Concepts

* **Library:** Turi Create (`turicreate`).
* **Algorithm:** **k-Nearest Neighbors (k-NN)**.
* **Feature Extraction:**
* **Deep Features:** Instead of comparing raw pixels (which fail if an image is rotated or lighted differently), we calculate the distance between feature vectors. Images with similar "Deep Feature" vectors contain similar semantic content.



## 🚀 Methodology

1. **Data Loading:** Loaded training and testing image sets containing pre-computed deep features.
2. **Model Training:**
* Created a **Nearest Neighbor Model** using the `deep_features` column.
* The model indexes the high-dimensional feature vectors for fast retrieval.


3. **Querying (Retrieval):**
* **Cat Example:** Input an image of a cat. The model returns other cats, even if they have different colors or poses.
* **Car Example:** Input an image of an automobile. The model retrieves other cars, distinguishing them from animals.



## 📊 Sample Results

The model demonstrates semantic understanding of the images:

| Query Image | Retrieved Neighbors (Top Matches) |
| --- | --- |
| **Cat** (ID: 18) | Successfully retrieves other distinct cat images. |
| **Automobile** (ID: 8) | Successfully retrieves other cars, ignoring background noise. |

## ⚙️ Usage Instructions

1. **Environment Setup:**
Ensure you have Turi Create installed in your Python environment.
```bash
pip install turicreate

```


2. **Run the Notebook:**
```bash
jupyter notebook "Deep Learning - image retrieval.ipynb"

```


3. **Code Snippet (Retrieval):**
```python
import turicreate as tc

# Load data
image_train = tc.SFrame('image_train_data/')

# Create the Nearest Neighbor Model
knn_model = tc.nearest_neighbor_classifier.create(
    image_train,
    features=['deep_features'],
    label='label'
)

# Query with a specific image (e.g., a cat)
cat_image = image_train[18:19]
similar_images = knn_model.query(cat_image)
similar_images.explore()

```


---

**Author:** BAIHICH Mohamed
