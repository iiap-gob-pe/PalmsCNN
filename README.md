Here’s the descriptive text about the **# PalmsCNN** repository in English:
# **PalmsCNN**

The **PalmsCNN** repository provides a comprehensive guide detailing the process of training a model designed to segment and detect the crowns of three palm species in Amazonian forests using RGB imagery captured by drones. This work is part of the research conducted by **Tagle et al.** (currently under review) in their study titled **"Overcoming the Research-Implementation Gap through Drone-based Mapping of Economically Important Amazonian Palms."**

The proposed approach combines drone-based data with an advanced architecture that integrates **ecoCNN** for data generation, **DeepLabv3+** for semantic segmentation, and **DWT (Discrete Wavelet Transform)** for image processing. This combination enables precise and efficient detection of palm crowns in aerial imagery.

#### **Species Detected**
The model is trained to detect three ecologically and economically important palm species in the Amazon:
1. **Mauritia Flexuosa** (Class 1)
2. **Euterpe Precatoria** (Class 2)
3. **Oenocarpus Bataua** (Class 3)

#### **Step-by-Step Guide**
The repository includes a step-by-step guide that explains how to:
1. Load an RGB mosaic obtained by a drone (UAV).
2. Apply the trained model to detect and segment the crowns of the three palm species.
3. Use the **ecoCNN** approach along with the **DeepLabv3+** and **DWT** architectures for image processing and analysis.

#### **Training Tutorial**
For those interested in replicating or understanding the model training process, the repository provides a detailed tutorial in the **PalmsCNN_Tutorial** file. This tutorial covers the necessary steps for data generation, model training, and result validation.

#### **Scientific Contribution**
This work aims to bridge the gap between scientific research and practical application, demonstrating how drone technology and deep learning can be used for mapping and monitoring plant species in complex ecosystems like the Amazon. The code and methodology presented in this repository are a valuable contribution to the scientific community and professionals in the fields of ecology and remote sensing.

---

### **Technologies and Methods Used**
- **Drone (UAV) Data**: High-resolution RGB imagery.
- **ecoCNN**: Data generation for training.
- **DeepLabv3+**: Semantic segmentation architecture.
- **DWT (Discrete Wavelet Transform)**: Image processing for enhanced detection.

---

This repository is an invaluable tool for researchers, ecologists, and professionals interested in vegetation mapping and the application of artificial intelligence in environmental studies. Explore the code and contribute to the advancement of science! 🌿🤖

