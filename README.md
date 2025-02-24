# **PalmsCNN**

The **PalmsCNN** repository provides the code for training a model designed to segment and detect the crowns of three palm species in Amazonian forests using RGB imagery captured by drones. This work is part of the research conducted by **Tagle et al.** 2025 **"Effective integration of drone technology for mapping and managing palm species in the Peruvian Amazon."**

The proposed approach combines drone-based data with an architecture that integrates **ecoCNN** for data generation, **DeepLabv3+** for semantic segmentation, and **DWT (Deep Watershed)** for image processing. This combination enables precise and efficient detection of palm crowns in aerial imagery.

**Species Detected**
The model is trained to detect three ecologically and economically important palm species in the Amazon:
0. Background (0)
1. **Mauritia Flexuosa** (Class 1)
2. **Euterpe Precatoria** (Class 2)
3. **Oenocarpus Bataua** (Class 3)

#### **Jupyter Notebooks**
The repository includes a step-by-step guide that explains how to:
1. Train a model to detect 3 palm species
2. Train a model to split the crowns of the segmented crowns
3. Apply the trained model to detect and count the crowns of the three palm species.

#### **Python scripts**
For those interested in running the model directly from the original python scripts, they are also available and enumerated to follow the steps for data generation, model training, and predictions.

