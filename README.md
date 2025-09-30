# 💰 Machine Learning Salary Prediction

A comprehensive machine learning project that predicts salaries based on demographic and professional features using Random Forest Regression with extensive hyperparameter tuning.

## 📌 Project Overview

This project builds a predictive model to estimate salaries based on:
- **Age** and **Years of Experience** (numerical features)
- **Gender**, **Education Level**, and **Job Title** (categorical features)

The model achieves an impressive **98.27% R² score** on test data with a **Mean Absolute Error of $3,840**.

### 🎯 Problem Statement
Salary prediction is crucial for:
- HR departments setting competitive compensation
- Job seekers understanding market rates
- Companies ensuring fair pay structures
- Career planning and salary negotiations

## 📊 Dataset Information

- **Size**: 6,704 samples with 6 features
- **Target Variable**: Salary (ranging from $350 to $250,000)
- **Features**:
  - `Age`: 21-62 years (mean: 33.6)
  - `Gender`: Male, Female, Other
  - `Education Level`: High School, Bachelor's Degree, Master's Degree, PhD
  - `Job Title`: 200+ unique job titles (Software Engineer, Data Scientist, etc.)
  - `Years of Experience`: 0-34 years (mean: 8.1)

### 📈 Key Statistics
- **Mean Salary**: $115,327
- **Salary Range**: $350 - $250,000
- **Missing Values**: Successfully handled using median (numerical) and mode (categorical) imputation

## 🛠️ Technical Implementation

### Model Architecture
- **Algorithm**: Random Forest Regression
- **Preprocessing Pipeline**:
  - OneHotEncoder for categorical features (200+ categories)
  - StandardScaler for numerical features
  - Integrated preprocessing + model pipeline

### Hyperparameter Optimization
- **GridSearchCV** with 5-fold cross-validation
- **Parameters Tuned**: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features
- **Best Parameters**:
  ```python
  {
      'model__max_depth': 30,
      'model__max_features': 'sqrt',
      'model__min_samples_leaf': 1,
      'model__min_samples_split': 5,
      'model__n_estimators': 200
  }
  ```

### 📊 Model Performance
- **Test R² Score**: 0.9827 (98.27% variance explained)
- **Mean Absolute Error**: $3,840
- **Mean Squared Error**: $47,826,315
- **Cross-Validation Mean R²**: 0.9771 (±0.0039)

## 🔍 Feature Importance Analysis

Top predictive features:
1. **Years of Experience**: 75.57% importance
2. **Age**: 4.52% importance  
3. **Job Title - Data Scientist**: 3.37% importance
4. **Job Title - Data Analyst**: 3.34% importance
5. **Job Title - Software Engineer**: 2.97% importance

## ⚙️ Installation & Setup

### Requirements
```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

### Dependencies
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- joblib >= 1.1.0

## ▶️ How to Run

### 1. Run the Jupyter Notebook
```bash
jupyter notebook "Salaries_ML Model.ipynb"
```

### 2. Train the Model
The notebook will automatically:
- Load and preprocess the data
- Perform hyperparameter tuning
- Train the final model
- Save the model as `salary_prediction_model.pkl`

### 3. Make Predictions
```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("salary_prediction_model.pkl")

# Make a prediction
def predict_salary(Age, Gender, Edu_Level, Job_Title, Years_Exp):
    input_data = pd.DataFrame([{
        "Age": Age,
        "Gender": Gender,
        "Education Level": Edu_Level,
        "Job Title": Job_Title,
        "Years of Experience": Years_Exp
    }])
    return model.predict(input_data)[0]

# Example usage
predicted_salary = predict_salary(
    Age=30, 
    Gender="Male", 
    Edu_Level="PhD", 
    Job_Title="Data Scientist", 
    Years_Exp=20
)
print(f"Predicted salary: ${predicted_salary:,.2f}")
```

## 🖥️ Streamlit Web Application

### Setup Streamlit App
```bash
pip install streamlit
streamlit run app.py  # Create your Streamlit interface
```

### Sample Streamlit Code Structure
```python
import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("salary_prediction_model.pkl")

st.title("💰 Salary Prediction App")

# Create input widgets
age = st.slider("Age", 21, 62, 30)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
education = st.selectbox("Education Level", 
    ["High School", "Bachelor's Degree", "Master's Degree", "PhD"])
job_title = st.text_input("Job Title", "Data Scientist")
experience = st.slider("Years of Experience", 0, 34, 5)

if st.button("Predict Salary"):
    prediction = predict_salary(age, gender, education, job_title, experience)
    st.success(f"Predicted Salary: ${prediction:,.2f}")
```

## 📁 Project Structure
```
├── Salaries_ML Model.ipynb          # Main notebook
├── Salary_Data.csv                  # Dataset (add your own)
├── salary_prediction_model.pkl      # Trained model
├── README.md                        # This file
├── requirements.txt                 # Dependencies
└── app.py                          # Streamlit application
```

## 📊 Key Visualizations

The notebook includes:
- **Actual vs Predicted Comparison**: Bar chart showing model accuracy on sample predictions
- **Feature Importance Plot**: Visualization of top contributing factors
- **Cross-validation Scores**: Model consistency across different data splits

## 🔧 Model Features

- **Robust Preprocessing**: Handles missing values and categorical encoding automatically
- **Feature Engineering**: OneHot encoding for 200+ job titles
- **Cross-validation**: 10-fold validation ensures model generalizability  
- **Hyperparameter Tuning**: GridSearchCV optimization for best performance
- **Production Ready**: Saved model with prediction function

## 📈 Results Summary

The Random Forest model demonstrates excellent predictive power:
- **High Accuracy**: 98.27% of salary variance explained
- **Low Error**: Average prediction error of $3,840
- **Feature Insights**: Experience dominates salary predictions (75.6%)
- **Stable Performance**: Consistent across cross-validation folds

## 🚀 Future Improvements

- Add more features (location, company size, industry)
- Implement ensemble methods (XGBoost, LightGBM)
- Create interactive web dashboard
- Add model interpretability tools (SHAP values)
- Implement real-time prediction API

## 📝 Usage Examples

### Basic Prediction
```python
# Predict salary for a Senior Data Scientist
salary = predict_salary(
    Age=35,
    Gender="Female", 
    Edu_Level="Master's Degree",
    Job_Title="Senior Data Scientist",
    Years_Exp=8
)
```

### Batch Predictions
```python
# Predict for multiple candidates
candidates = pd.DataFrame({
    'Age': [28, 45, 33],
    'Gender': ['Male', 'Female', 'Male'],
    'Education Level': ['Bachelor\'s Degree', 'PhD', 'Master\'s Degree'],
    'Job Title': ['Software Engineer', 'Data Scientist', 'Product Manager'],
    'Years of Experience': [3, 15, 7]
})

predictions = model.predict(candidates)
```

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Built** **using Python, Scikit-learn, and Machine Learning best practices**
