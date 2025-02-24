# **PalmsCNN**

The **PalmsCNN** repository provides the code for training a model designed to segment and detect the crowns of three palm species in Amazonian forests using RGB imagery captured by drones. This work is part of the research conducted by **Tagle et al.** 2025 **"Effective integration of drone technology for mapping and managing palm species in the Peruvian Amazon."**

The proposed approach combines drone-based data with an architecture that integrates **ecoCNN** for data generation, **DeepLabv3+** for semantic segmentation, and **DWT (Deep Watershed)** for image processing. This combination enables precise and efficient detection of palm crowns in drone imagery.

**Species Detected**
The model is trained to detect three ecologically and economically important palm species in the Amazon:
0. Background (0)
1. **Mauritia Flexuosa** (Class 1)
2. **Euterpe Precatoria** (Class 2)
3. **Oenocarpus Bataua** (Class 3)

#### **Jupyter Notebooks**
The repository includes a step-by-step guide that explains how to:
1. Train a model to detect 3 palm species (1_PalmsCNN_Tutorial.ipynb, items from 1 to 4)
2. Train a model to split the crowns of the segmented crowns (1_PalmsCNN_Tutorial.ipynb, items from 5 to 7)
3. Apply the trained model to detect and count the crowns of the three palm species (2_PalmsCNN_Tutorial_Prediction.ipynb)

#### **Python scripts**
For those interested in running the models directly from the original python scripts, they are also available and enumerated to follow the steps for data generation, model training, and predictions. They can be found in the folders: 
- Palms_Segment (Scripts numbered from 1 to 4)
- Palms_Quant/E2E_palms (Scripts numbered from 1 to 4)

#### **Google Collab**
The notebooks can be run in Google Collab. 
1st run the 0_PalmsCNN_Getting_Started.ipynb
2nd the 1_PalmsCNN_Tutorial_Google_Colab.ipynb



