# 🧠 AI & Data Analytics in Supply Chain Systems — Term Project

[![Shiny App](https://img.shields.io/badge/Shiny-Running-green?logo=R)](https://xikr3m-bi2f.shinyapps.io/aishiny/)

---

<div align="center">

<img src="https://admissions.siit.tu.ac.th/wp-content/uploads/2023/06/cropped-TU-SIIT1992-01.png" height="100"/>

## Term Project - SE 634-1: AI & Data Analytics in Supply Chain Systems  
### Sirindhorn International Institute of Technology (SIIT)  
### Semester 2/2024  

</div>

---

## 📋 Project Title

**Comparison of Machine Learning Models for Diabetes Prediction**  
Using the **Pima Indians Diabetes dataset**, we built and evaluated three models to classify whether a patient has diabetes based on health parameters.

---

## 🧪 Models Used

1. **Logistic Regression**
2. **Decision Tree (CART)**
3. **K-Nearest Neighbors (KNN)**

All models were trained with cross-validation and evaluated on a held-out test set. Performance was compared based on accuracy and confusion matrix results.

---

## 📈 Live Shiny App

🔗 [Click here to open the live Shiny app](https://xikr3m-bi2f.shinyapps.io/appnew/)

This interactive Shiny application allows users to:

- View model accuracy comparisons  
- See confusion matrices for each model  
- Explore how each model performs on the validation and test datasets

---

## 📽 Presentation Slides

[![Watch the Slide Presentation](https://img.icons8.com/fluency/240/google-slides.png)](https://docs.google.com/presentation/d/16jZ11q2jrd408cw8Sdvptqg1ORq0ukkUBdBD8ptAQxI/edit?usp=sharing)

🔗 [Click here to view the Google Slides](https://docs.google.com/presentation/d/16jZ11q2jrd408cw8Sdvptqg1ORq0ukkUBdBD8ptAQxI/edit?usp=sharing)

---

## 👨‍🏫 Project Advisor

<div align="center">
<img src="https://drive.google.com/uc?export=view&id=16ebl13r_yph1SOclCfOy0EF3btFdpffO" height="160"/>
<br/>
<b>Dr. Warrut Pannakkong</b>  
Sirindhorn International Institute of Technology (SIIT)  
</div>

---

## 🗃 Dataset

- 📦 **PimaIndiansDiabetes** from the `mlbench` R package  
- Includes medical diagnostic measurements and diabetes outcomes for female patients of Pima Indian heritage.

---

## ⚙️ Technologies Used

- **R** for modeling  
- **Shiny** for web app  
- **caret**, **rpart**, **class** for machine learning  
- **RStudio** for development  
- **shinyapps.io** for deployment  

---

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

---

## 📁 File Structure

```bash
.
├── app.R              # Main Shiny app script
├── README.md          # This file


