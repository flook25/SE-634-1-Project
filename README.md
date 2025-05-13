# Diabetes Prediction and Model Comparison

🎓 **Term Project – SE 634-1: Artificial Intelligence & Data Analytics in Supply Chain Systems**  
📍 **Sirindhorn International Institute of Technology (SIIT)**  
👨‍🏫 **Advisor:** Dr. Warrut Pannakkong  

## 🔍 Overview

This interactive Shiny application allows users to **compare the performance of three machine learning models** — Logistic Regression, Decision Tree, and K-Nearest Neighbors (KNN) — on the **Pima Indians Diabetes dataset**. The app visualizes metrics like accuracy, sensitivity, specificity, and ROC curves, and also allows users to input patient data to receive real-time diabetes predictions from the selected model.

## 📊 Models Used

- **Logistic Regression**: A linear model used for binary classification.
- **Decision Tree**: A tree-based model for decision-making.
- **K-Nearest Neighbors (KNN)**: A non-parametric model based on feature similarity.

Each model is trained using 5-fold cross-validation and evaluated on separate validation and test datasets.

## 🚀 Live Demo

👉 **Try the app here**: [https://xikr3m-bi2f.shinyapps.io/aishiny/](https://xikr3m-bi2f.shinyapps.io/aishiny/)

## 🛠 How It Works

- The dataset is split into:
  - 70% training
  - 15% validation
  - 15% testing
- Features are normalized for KNN.
- Performance is evaluated using confusion matrices, ROC curves, and summary plots.
- You can interactively:
  - Select a model to view performance
  - Enter custom input values and get predictions

## 📦 Technologies Used

- **R** for data analysis and model building
- **Shiny** for interactive web application
- **Caret** for machine learning pipeline
- **ggplot2** and **pROC** for visualization

## 🧪 Sample Input for Prediction

| Feature              | Example |
|----------------------|---------|
| Pregnancies          | 2       |
| Glucose              | 120     |
| Blood Pressure       | 70      |
| Skin Thickness       | 20      |
| Insulin              | 85      |
| BMI                  | 28.0    |
| Diabetes Pedigree    | 0.45    |
| Age                  | 33      |

## 📁 File Structure

```bash
.
├── app.R              # Main Shiny app script
├── README.md          # This file
└── (Optional data or helper files if needed)
