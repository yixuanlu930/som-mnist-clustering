# SOM MNIST Clustering

An unsupervised learning project that applies a **Self-Organizing Map (SOM)** to the MNIST handwritten-digit dataset in order to study clustering, neuron specialization, class similarity, and prototype formation.

The project trains a **10 × 10 Kohonen map** on 12,000 MNIST images and analyzes how the two-dimensional neural grid organizes the original 784-dimensional image space.

---

## Overview

A Self-Organizing Map is an unsupervised neural-network model that projects high-dimensional data onto a lower-dimensional grid while attempting to preserve topological relationships.

In this project, a SOM is trained on handwritten digits from the **MNIST dataset**.

The main questions explored are:

1. Can a SOM learn meaningful structure from MNIST without using labels during training?
2. Does each neuron specialize in a single digit?
3. Which digits are most similar according to the learned representation?
4. Is a handwritten `5` more similar to a `4` or an `8`?
5. What visual patterns are represented by the neurons of the map?

---

## Dataset

The project uses the MNIST dataset obtained through:

```python
from sklearn.datasets import fetch_openml
```

MNIST contains:

```text
70,000 grayscale images
28 × 28 pixels
784 input features
10 digit classes: 0–9
```

For computational efficiency, the experiment uses:

```text
12,000 samples
```

The pixel values are normalized from:

```text
[0, 255]
```

to:

```text
[0, 1]
```

before training.

---

## Self-Organizing Map

The SOM consists of a two-dimensional:

```text
10 × 10
```

competitive neural grid.

This corresponds to:

```text
100 neurons
```

Each neuron contains a weight vector of:

```text
784 dimensions
```

which can later be reshaped into a:

```text
28 × 28
```

image.

These weight vectors therefore act as visual prototypes representing different regions of the MNIST input space.

---

## Training Configuration

The main training configuration is:

| Parameter                 |     Value |
| ------------------------- | --------: |
| Map size                  |   10 × 10 |
| Neurons                   |       100 |
| Training samples          |    12,000 |
| Input dimensions          |       784 |
| Epochs                    |       100 |
| Initial learning rate     |       0.8 |
| Final learning rate       |      0.05 |
| Final neighborhood radius |       0.4 |
| Distance                  | Euclidean |

Example:

```python
som = SOM(10, 10)

som.train(
    X_scaled,
    epochs=100,
    alpha0=0.8,
    alphaf=0.05,
    sigmaf=0.4,
    trace=10,
    verbose=1
)
```

The labels are **not used during SOM training**.

They are introduced only afterwards to evaluate and interpret the learned map.

---

# How a SOM Works

For every training sample:

```text
Input image
    │
    ▼
Compare with all neurons
    │
    ▼
Best Matching Unit (BMU)
    │
    ▼
Update BMU weights
    │
    ▼
Update neighboring neurons
```

The SOM gradually organizes itself so that similar input patterns activate nearby regions of the map.

This produces a topology in which visually similar digits tend to appear close to one another.

---

# Best Matching Unit

For each MNIST image, the SOM identifies the neuron whose weight vector is closest to the input.

This neuron is known as the:

```text
Best Matching Unit (BMU)
```

The BMU coordinates are used to determine where each digit is represented on the two-dimensional map.

---

# Neuron Labeling

After training, each neuron is assigned a dominant digit based on the MNIST samples for which that neuron becomes the BMU.

For example:

```text
Neuron (3, 7)

Digit 3 → 84 samples
Digit 8 → 10 samples
Digit 5 → 6 samples
```

The dominant label would therefore be:

```text
3
```

with purity:

```text
84 / 100 = 84%
```

---

# Neuron Purity

The project calculates the percentage of samples assigned to each neuron that belong to its dominant class.

Conceptually:

```text
purity =
samples from dominant class
────────────────────────────
total samples assigned
```

High purity indicates that a neuron has specialized strongly in a particular digit.

The trained SOM achieved approximately:

```text
Average neuron purity: 83.84%
```

and:

```text
71 neurons > 80% purity
```

This shows that most regions of the map become strongly specialized despite the absence of class labels during training.

---

# Classification Accuracy

Although SOM is fundamentally an unsupervised algorithm, a pseudo-classifier can be constructed by assigning each neuron its dominant MNIST label.

For every sample:

```text
Image
  │
  ▼
BMU
  │
  ▼
Dominant neuron label
  │
  ▼
Predicted digit
```

Using this strategy, the experiment achieves approximately:

```text
Global accuracy: 85.2%
```

This is a strong result considering that the labels are never used to optimize the SOM weights.

---

# Map Visualization

The project includes several complementary visualizations.

## SOM Activation Map

The provided:

```python
draw_map()
```

function visualizes:

* Neuron activity
* Dominant labels
* Cluster boundaries
* Sample distribution

This provides a global view of how MNIST classes are distributed across the learned topology.

---

## Sample Distribution

Each MNIST sample is projected onto its winning neuron.

Samples are colored according to their true digit class.

This makes it possible to visually inspect:

* Cluster formation
* Class overlap
* Transition regions
* Digit neighborhoods

---

## Class Distribution per Neuron

Pie charts are generated for every active neuron.

Each pie represents the true digit distribution of the samples activating that neuron.

For example:

```text
Neuron
 ├── 80% digit 3
 ├── 12% digit 8
 └──  8% digit 5
```

This provides a more detailed representation than displaying only the dominant class.

---

# SOM Prototypes

Every neuron contains a 784-dimensional centroid.

The weights can be reshaped to:

```text
28 × 28
```

and displayed as images.

These visualizations show what each neuron has learned as its representative handwritten-digit pattern.

The project displays:

* All 100 SOM prototypes
* The five purest prototypes for each digit

This provides a visual interpretation of the internal representation learned by the unsupervised network.

---

# Confusion Matrix

A confusion matrix is generated by treating the dominant label of each BMU as a prediction.

The project includes:

* Absolute confusion matrix
* Row-normalized confusion matrix

This allows identification of digit pairs that the SOM frequently confuses.

For example, visually similar digits may activate neighboring or overlapping regions.

---

# Digit Similarity

A similarity matrix is constructed using symmetric confusion rates.

For two digits (i) and (j):

```text
similarity(i,j) =
(confusion(i→j) + confusion(j→i)) / 2
```

This makes it possible to quantify which digit classes are represented similarly by the SOM.

---

## 5 vs. 4 vs. 8

One of the central questions of the project is:

> Is a handwritten 5 more similar to an 8 or to a 4?

The trained SOM answers this empirically by comparing the corresponding confusion and neighborhood relationships.

Rather than relying on visual intuition, similarity is derived from the topology learned directly from the MNIST samples.

---

# Why SOM Is Useful Here

MNIST images exist in a:

```text
784-dimensional space
```

which humans cannot inspect directly.

The SOM transforms this into a:

```text
10 × 10 two-dimensional map
```

while attempting to preserve local similarity.

This provides an intuitive visualization of relationships between handwritten digits.

Conceptually:

```text
784-dimensional MNIST
        │
        ▼
Self-Organizing Map
        │
        ▼
2D neural topology
        │
        ├── clusters
        ├── prototypes
        ├── class boundaries
        └── similarities
```

---

# Project Structure

```text
som-mnist-clustering/
│
├── som.py
├── SOM_MNIST_Practica.ipynb
└── README.md
```

### `som.py`

Implementation of the Self-Organizing Map used by the project.

It provides functionality including:

* SOM initialization
* Distance metrics
* Training
* Best Matching Unit prediction
* Activation maps
* Prototype extraction
* Distribution summaries
* Map visualization

### `SOM_MNIST_Practica.ipynb`

Main experiment and analysis notebook.

It contains:

* MNIST loading
* Preprocessing
* SOM training
* Neuron labeling
* Purity analysis
* Activation visualization
* Prototype visualization
* Confusion matrices
* Digit-similarity analysis
* Final conclusions

---

# Installation

A Python environment with Jupyter is recommended.

Install the required packages:

```bash
pip install numpy matplotlib scikit-learn jupyter
```

---

# Running the Project

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/som-mnist-clustering.git
cd som-mnist-clustering
```

Start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
SOM_MNIST_Practica.ipynb
```

The notebook automatically retrieves MNIST using OpenML.

An internet connection is therefore required the first time the dataset is downloaded.

---

# Main Results

The experiment demonstrates that a Self-Organizing Map can learn meaningful structure from MNIST without supervised training.

Key results include approximately:

```text
Training samples          12,000
SOM neurons                  100
Training epochs              100
Global accuracy            85.2%
Average neuron purity     83.84%
Neurons above 80% purity     71
```

The map also reveals that:

* Most neurons specialize in a dominant digit.
* Visually similar digits tend to occupy neighboring regions.
* Some neurons represent transitional patterns between classes.
* SOM prototypes resemble recognizable handwritten digits.
* Unsupervised topology can provide meaningful information about inter-class similarity.

---

# Technologies

* Python
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook
* OpenML
* MNIST

---

# Machine Learning Concepts

The project demonstrates concepts including:

* Unsupervised Learning
* Self-Organizing Maps
* Kohonen Networks
* Competitive Learning
* Clustering
* Dimensionality Reduction
* Topology Preservation
* Best Matching Units
* Prototype Learning
* Neural Networks
* Cluster Purity
* Confusion Matrices
* Data Visualization

---

# Limitations

Several limitations should be considered.

### Training subset

Only 12,000 of the 70,000 MNIST samples are used in the main experiment to keep training time manageable.

### SOM size

The selected 10 × 10 map is a design choice.

Different grid sizes could produce different balances between:

* Specialization
* Interpretability
* Empty neurons
* Computation time

### Post-training labels

The SOM itself is unsupervised, but class labels are used after training to interpret the resulting neurons and calculate classification metrics.

Therefore, the reported accuracy should not be interpreted as supervised model training accuracy.

---

# Possible Extensions

Potential future improvements include:

* Training on all 70,000 MNIST samples
* Comparing different map dimensions
* Hyperparameter optimization
* Toroidal SOM topology
* Alternative distance metrics
* U-Matrix visualization
* Quantization error analysis
* Topographic error analysis
* PCA comparison
* t-SNE comparison
* UMAP comparison
* Fashion-MNIST experiments
* Interactive SOM visualization

---

# Academic Context

This project was developed as an educational exercise focused on **Self-Organizing Maps and unsupervised neural learning**.

The objective was to understand how a Kohonen map can organize high-dimensional handwritten-digit data while preserving meaningful topological relationships.

---

# Attribution

The `SOM` class used in this project is based on course material provided by **Francisco Serradilla, Universidad Politécnica de Madrid**, and adapted/reconstructed for the practical exercise.

---

# Disclaimer

This repository is intended for educational and experimental purposes.

Results can vary depending on initialization, selected samples, SOM configuration, and software environment.

---

# License

See the repository license for applicable terms.
