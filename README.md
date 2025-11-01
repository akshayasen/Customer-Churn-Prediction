# Customer Churn Prediction using Artificial Neural Network (ANN) 

Hey there! 👋  
This project is all about predicting **whether a customer is likely to leave (churn)** or stay with a bank — using **deep learning**.  
I wanted to go beyond basic models and build an **end-to-end churn prediction pipeline** that’s both accurate and deployment-ready.


---

##  Project Overview

Customer churn is one of the biggest challenges for any business. The goal here is to **analyze customer behavior**, identify key patterns that lead to churn, and use an **Artificial Neural Network (ANN)** to predict it with high accuracy.

This project walks through:
- Data preprocessing (encoding, scaling)
- Model building using TensorFlow/Keras
- Model optimization with EarlyStopping
- Evaluation and prediction
- Saving encoders, scalers, and model for deployment
- Creating an interactive Streamlit web app

---

##  Dataset

The dataset contains **10,000 customer records** with details like:
- Customer demographics (Age, Gender, Geography)
- Financial information (Credit Score, Balance, Estimated Salary)
- Account activity (Number of Products, Active Member, Tenure)
- Target variable: **Exited (1 = Churned, 0 = Stayed)**

---

## Tech Stack

- **Python**  
- **TensorFlow / Keras** – Model building  
- **Scikit-learn** – Preprocessing & metrics  
- **Pandas & NumPy** – Data handling  
- **Pickle** – Saving encoders & scaler  
- **Streamlit** – Web app deployment  
- **Jupyter Notebook / PyCharm** – Development environment  

---

##  Project Workflow

### **1. Data Preprocessing**
- Dropped unnecessary columns like `RowNumber`, `CustomerId`, and `Surname`.
- Handled categorical data:
  - **Gender** → Label Encoded
  - **Geography** → One-Hot Encoded
- Standardized features using **StandardScaler**.

### **2. Model Building**
Built a simple but effective **Sequential Neural Network**:
- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Metric: Accuracy
- Added EarlyStopping and TensorBoard callbacks to monitor performance.
- python
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

 **3. Training**

- Trained the model for **multiple epochs** using validation data.  
- Implemented **EarlyStopping** to prevent overfitting by halting training when the validation loss stopped improving.  
- Achieved approximately **86% validation accuracy**, demonstrating strong model generalization.

---

### **4. Model Evaluation**

- The model was evaluated based on key performance metrics:
  -  **Accuracy**
  - **Validation Loss**  
- Additionally, tested with **real-world customer inputs** to verify prediction reliability and consistency.

---

### **5. Saving Artifacts**

All essential components were saved to ensure smooth deployment and reproducible predictions:
 - label_encoder.pkl
 - onehot_encoder.pkl
 - scaler.pkl
 - model.h5

These artifacts allow for consistent preprocessing and model inference during deployment.

---

### **6. Streamlit Deployment**

- Integrated the trained model into a **Streamlit web application** for a user-friendly experience.  
- **Deployed on Streamlit Community Cloud**, allowing anyone to interact with the app easily.  
- The application accepts **customer details** (e.g., Age, Geography, Balance, Tenure, etc.) and returns:
  - The **churn prediction**
  - The **predicted probability score**

> Example Output:  
> ✅ *The customer is not likely to churn.*  
> ⚠️ *The customer is likely to churn.*

---





