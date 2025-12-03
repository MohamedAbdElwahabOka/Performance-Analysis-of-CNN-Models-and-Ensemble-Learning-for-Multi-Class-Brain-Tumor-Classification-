# Performance Analysis of CNN Models and Ensemble Learning for Multi-Class Brain Tumor Classification

[![Python](https://img.shields.io/badge/python-3.10-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 🚀 Overview

This project conducts a comprehensive performance analysis of multiple Convolutional Neural Network (CNN) models and ensemble learning techniques for multi-class classification of brain tumor MRI images. The goal is to evaluate and compare the accuracy, precision, recall, and F1-score of individual CNN models, as well as an ensemble model combining multiple CNNs, to determine whether ensemble learning improves overall classification performance.

---

## 👂 Project Structure

```
Performance-Analysis-of-CNN-Models-and-Ensemble-Learning/
│
├─ performance-analysis-of-cnn-models-and-ensemble.ipynb  # Main Jupyter Notebook
├─ data/                                                  # Dataset folder (MRI images organized by tumor type)
├─ models/                                                # (Optional) Pre-trained or saved models
├─ results/                                               # (Optional) Generated results, plots, metrics
└─ README.md                                              # Project documentation
```

> Note: Ensure that your dataset is properly organized and accessible to the notebook.

---

## 👨‍💻 Requirements

* Python 3.x
* Essential Python libraries:

  ```bash
  pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras
  ```
* (Optional) GPU with CUDA/cuDNN for faster training
* Adequate storage for MRI images, trained models, and results

---

## 📚 Usage / Getting Started

1. **Prepare the Dataset**

   * Download the MRI dataset and organize it by tumor type (each class in a separate folder or with proper labeling).

2. **Run the Notebook**

   * Open `performance-analysis-of-cnn-models-and-ensemble.ipynb` in Jupyter Notebook, JupyterLab, or VSCode.
   * Execute cells sequentially from top to bottom.

3. **Train and Evaluate Models**

   * Train individual CNN models and evaluate their performance.
   * Build an ensemble model (voting, averaging, or stacking) from multiple CNNs.

4. **Analyze Results**

   * Compare metrics such as Accuracy, Precision, Recall, F1-score for each model and ensemble.
   * Plot confusion matrices and performance graphs for better visualization.

---

## 📊 Features

* Comparison of multiple CNN architectures on the same MRI dataset
* Implementation of ensemble learning to combine multiple CNN models
* Performance metrics evaluation (accuracy, precision, recall, F1-score)
* Confusion matrix visualization
* Flexible notebook that can be adapted to new datasets easily

---

## 🔬 Background & Motivation

* Ensemble learning combines multiple models to achieve better accuracy and robustness than individual models.
* CNNs are highly effective for medical image classification, particularly for MRI scans.
* This project replicates research experiments on CNN ensembles for brain tumor classification, providing a reproducible framework for further experimentation.

---

## ✅ Completed Tasks

* Implemented multiple CNN models for brain tumor classification
* Developed an ensemble model combining CNN predictions
* Evaluated models using standard performance metrics
* Visualized results and compared model performance

### 🔧 Future Improvements

* Experiment with more datasets to improve generalization
* Apply data augmentation or advanced preprocessing techniques
* Implement cross-validation for more reliable evaluation
* Save trained models for inference on new MRI images
* Develop a simple GUI or API for classifying new images

---

## ⚠️ Notes / Limitations

* Results are dataset-specific; performance may vary with other MRI datasets
* Ensemble improvement depends on model diversity
* Large datasets and complex CNN architectures require substantial computational resources

---

## 📚 References

1. [Ensemble Learning - Wikipedia](https://en.wikipedia.org/wiki/Ensemble_learning)
2. [CNNs for Medical Image Classification - Wikipedia](https://en.wikipedia.org/wiki/Convolutional_neural_network)
3. Research on CNN ensembles for brain tumor classification ([example paper](https://www.mdpi.com/2078-2489/15/10/641))

---

## 👤 Author

**Mohamed Abdelwahab Oka**

* GitHub: [https://github.com/MohamedAbdElwahabOka](https://github.com/MohamedAbdElwahabOka)

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
