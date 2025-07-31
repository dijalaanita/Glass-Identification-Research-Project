# Glass Identification Using Machine Learning

This project explores the classification of glass types based on their chemical properties using machine learning techniques. It is aimed at real-world applications such as forensic analysis and materials engineering, where identifying the right type of glass is crucial.

## 📊 Dataset

The dataset used contains:
- 214 glass samples
- 9 chemical properties (features)
- 7 classes (glass types)

Data Source: UCI Machine Learning Repository - [Glass Identification Data Set](https://archive.ics.uci.edu/ml/datasets/glass+identification)

## 🔍 Objective

To compare the performance of different classification algorithms in identifying the correct glass type, and determine which model provides the best accuracy and generalization for new data.

## ⚙️ Models Used

1. **K-Nearest Neighbors (KNN)**  
   - No training phase (lazy learner)
   - High accuracy: **79.07%**

2. **Decision Tree (DT)**  
   - Interpretable structure
   - Accuracy: **62.79%**

3. **Naive Bayes (NB)**  
   - Based on probability and independence assumption
   - Accuracy: **51.16%**

4. **Support Vector Machine (SVM)**  
   - Margin-based classifier
   - Accuracy: **39.53%**

## 📈 Results Summary

| Model       | Accuracy |
|-------------|----------|
| KNN         | 79.07%   |
| Decision Tree | 62.79% |
| Naive Bayes | 51.16%   |
| SVM         | 39.53%   |

The **KNN model** outperformed the others and is recommended for glass type classification tasks based on this dataset.

## 🧪 Evaluation Method

- Dataset split using a random seed derived from the student ID for reproducibility
- Models trained and tested using scikit-learn
- Accuracy and confusion matrix used for performance comparison

## 🧠 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/glass-identification.git
   cd glass-identification
