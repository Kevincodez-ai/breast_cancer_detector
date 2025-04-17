
# 🧠 Breast Cancer Prediction using K-Nearest Neighbors (KNN)

This project implements a **K-Nearest Neighbors (KNN)** classification model to predict whether a tumor is **malignant** or **benign** based on features from a breast cancer dataset.

## 📁 Dataset

The dataset used in this project is assumed to be named `KNNAlgorithmDataset.csv`, based on the [Breast Cancer Wisconsin Diagnostic Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).

## ⚙️ Tech Stack

- Python 🐍
- Pandas
- NumPy
- Scikit-learn (sklearn)

## 📊 Features in Dataset

The dataset contains various attributes extracted from breast mass images like:
- Mean radius
- Texture
- Perimeter
- Area
- Smoothness
- Symmetry
- ... and many more.

Each row represents one tumor sample.

---

## 🚀 How it Works

### 1. **Data Preprocessing**
- Load the dataset using `pandas`.
- Drop unnecessary columns (`id`, `Unnamed: 32`).
- Encode categorical values (`M` = 1, `B` = 0).
- Split dataset into features (`X`) and label (`Y`).

### 2. **Train-Test Split**
- Dataset is split into 80% training and 20% testing using `train_test_split`.

### 3. **Feature Scaling**
- StandardScaler is used to normalize the features for better performance of KNN.

### 4. **Model Training**
- A `KNeighborsClassifier` with `n_neighbors = 5` is trained on the scaled training set.

### 5. **Evaluation**
- Accuracy and a detailed classification report (precision, recall, F1-score) are generated.

### 6. **Prediction**
- A custom sample is passed to the model to predict whether the tumor is **Malignant** or **Benign**.

---

## ✅ Results

- **Accuracy**: ~93-97% (may vary slightly on re-run)
- **Classifier Output**: Classification report includes precision, recall, f1-score.

---

## 📦 Sample Prediction

```python
sample = np.array([[14.2, 20.5, 92.3, 670.2, 0.089, 0.135, 0.075, 0.050, 0.18, 0.062,
    0.45, 1.1, 2.8, 35.6, 0.0, 0.020, 0.015, 0.01, 0.02, 0.003,
    16.9, 27.4, 112.5, 850.3, 0.12, 0.25, 0.18, 0.14, 0.28, 0.09]])
```

Prediction Output:
```
Malignant
```

---

## 🧪 Installation & Run

1. Clone the repository:
```bash
git clone https://github.com/yourusername/breast-cancer-knn.git
cd breast-cancer-knn
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the script:
```bash
python breast_cancer_knn.py
```

---

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [UCI ML Repository: Breast Cancer Wisconsin Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

---

## 💡 Future Improvements

- Add GUI for user-friendly predictions
- Experiment with different values of `k`
- Compare KNN with other models like SVM, Random Forest, etc.
- Deploy the model using Flask or Streamlit

---

## 🧑‍💻 Author

**Kevin James**  
Computer Science & Engineering | Panimalar Engineering College 
ML Enthusiast • Python Programmer  

