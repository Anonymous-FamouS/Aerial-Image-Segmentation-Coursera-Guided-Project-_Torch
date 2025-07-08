# 🛰️ Aerial Image Segmentation (Coursera Guided Project)

This repository contains work based on the [**Aerial Image Segmentation with PyTorch**](https://www.coursera.org/projects/aerial-image-segmentation-with-pytorch) guided project offered on Coursera.

> **Disclaimer:** Due to Coursera's content sharing policy and the agreement accepted upon enrollment, the complete code and final notebook cannot be publicly shared. Instead, this repository presents selected snippets and visual outputs to showcase the workflow and results.

---

## 🧠 Project Overview

The goal of this project was to gain hands-on experience with **semantic image segmentation** using aerial imagery. A deep learning model was trained to extract and segment **road networks** from high-resolution aerial images.

- Framework used: **PyTorch**
- Model architecture: **U-Net**
- Encoder: **EfficientNet** (pretrained on ImageNet)
- Task: Segmenting roads from aerial photographs

<p align="center">
  <img src="https://github.com/user-attachments/assets/66dc940e-9e5e-4540-8e18-ab60bf348c1a" width="1200"/>
</p>

---

## 📂 Dataset

The dataset used in this project is a subset of the **Massachusetts Roads Dataset**, consisting of:

- **200 aerial images** (with corresponding ground-truth masks)
- Image dimensions: `1500 × 1500` pixels
- Coverage: Each image spans approximately **2.25 km²**

> 📎 You can access the **full dataset** here:  
> 🔗 https://www.cs.toronto.edu/~vmnih/data/

---

## 🛠️ Libraries & Their Usage

| Library                            | Purpose                                                                 |
|------------------------------------|-------------------------------------------------------------------------|
| `torch`                            | Deep learning framework used for model definition, training, and inference. |
| `segmentation_models_pytorch`      | Provides the U-Net model architecture with pretrained EfficientNet encoder. |
| `albumentations`                   | Used for image augmentation (resize, flips) in training and validation data. |
| `opencv-python (cv2)`              | For reading, converting, and processing input images and masks.        |
| `numpy`                            | Array operations and pre-processing (e.g., expanding mask dimensions). |
| `pandas`                           | Reading and handling CSV dataset file.                                 |
| `matplotlib.pyplot`                | Visualising input images, masks, and model predictions.                |
| `scikit-learn (train_test_split)` | Splitting the dataset into training and validation subsets.            |
| `tqdm`                             | Adding progress bars during model training.                            |
| `helper` *(custom module)*         | Contains project-specific utility functions like training loop, evaluation, plotting, and loss functions. |
| `torch.utils.data.Dataset`         | Used to create a custom dataset class for loading and augmenting image-mask pairs. |

---

## 🖼️ Sample Outputs

Model input and prediction (road segmentation):


<p align="center">
  <img src="https://github.com/user-attachments/assets/02a6fd8d-b5c0-46a6-824c-3e68664179a2" width="1200"/>
</p>

---

## 📌 Notes

- This project serves as an educational exercise in semantic segmentation using real-world data.
- The final model was trained using transfer learning and data augmentation techniques to improve generalization.

---

## 📚 Acknowledgements

- Guided project by Coursera  
- Dataset by [Volodymyr Mnih](https://www.cs.toronto.edu/~vmnih/) (University of Toronto)

> ⚠️ **For educational use only.** Do not reuse course-specific materials unless permitted by the platform's terms.
