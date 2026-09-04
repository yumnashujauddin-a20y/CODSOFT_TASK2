"# CODSOFT_TASK2 "
🛡️ FraudGuard AI — Credit Card Fraud Detection

CODSOFT Internship — Task 2

FraudGuard AI is a machine-learning-powered credit card transaction fraud detection application. It loads a trained model, accepts transaction feature values through an interactive Streamlit interface, and classifies transactions as potentially fraudulent or safe.

The application also provides model-performance and dataset-explorer pages for inspecting the trained model and available test data.

✨ Features

🔐 Login screen with username/password authentication

🛡️ Modern FraudGuard AI dashboard

🤖 Pre-trained model loaded from fraud_model.joblib

🔮 Live transaction prediction

📈 Model confidence when probability prediction is available

📊 Model performance dashboard

🧩 Confusion matrix visualization

📉 ROC curve visualization

📋 Classification report

📁 Test dataset explorer

🔎 Dataset preview and column information

⚡ Streamlit caching for efficient model/data loading

🚀 Suitable for local Streamlit deployment

🧠 How It Works

Transaction Data
       │
       ▼
Feature Preparation
       │
       ▼
Trained ML Model
       │
       ▼
Fraud Prediction
       │
       ├──────────────► 🚨 Potential Fraud
       │
       └──────────────► ✅ Appears Safe

The application loads the serialized model from:

fraud_model.joblib

When available, the test dataset is loaded from:

fraudTest.csv

The application attempts to obtain feature names from the saved estimator. If feature names are not stored in the model, it can fall back to the columns of fraudTest.csv, excluding common target-column names.

🖥️ Application Pages

🏠 Dashboard

Provides a quick overview of:

Model name

Number of model features

Number of test records

Model readiness/status

Available saved model reports

🔮 Live Prediction

Users can enter values for the model's expected features and select:

🛡️ Analyze Transaction

The application displays:

Prediction result

Fraud/safe classification

Model confidence when predict_proba() is available

📊 Model Performance

The performance page can display:

Model metrics

Confusion matrix

ROC curve

Classification report

These artifacts are read from:

fraud_model_output/

📁 Data Explorer

When fraudTest.csv is available, the Data Explorer shows:

Number of rows

Number of columns

Missing cells

A preview of the first 200 rows

Column data types

Missing-value counts

Unique-value counts

🔐 Demo Login

The current application contains demo credentials:

Username: admin
Password: admin123

Security note: These credentials are for demonstration/development only. For production, use secure authentication and store secrets outside the source code.

📂 Project Structure

CODSOFT_TASK2/
│
├── app.py
├── train_fraud_model.py
├── fraud_model.joblib
├── fraudTest.csv
├── requirements.txt
├── README.md
│
└── fraud_model_output/
    ├── model_metrics.csv
    ├── classification_report.csv
    ├── confusion_matrix.png
    └── roc_curve.png

The exact contents of fraud_model_output/ depend on which training and evaluation artifacts have been generated.

🛠️ Technologies Used

Programming

Python

Data Processing

Pandas

NumPy

Machine Learning

Scikit-learn

Joblib

Application

Streamlit

Evaluation

Classification report

Confusion matrix

ROC curve

Model metrics

⚙️ Installation

1. Clone the repository

git clone https://github.com/yumnashujauddin-a20y/CODSOFT_TASK2.git

2. Navigate to the project directory

cd CODSOFT_TASK2

3. Create a virtual environment

python -m venv venv

4. Activate the environment

Windows

venv\Scripts\activate

macOS / Linux

source venv/bin/activate

5. Install dependencies

pip install -r requirements.txt

▶️ Run the Application

Start the Streamlit application:

streamlit run app.py

Then open the local URL shown in the terminal, normally:

http://localhost:8501

🔧 Required Project Files

For the main application to start successfully, make sure the trained model is available in the project directory:

fraud_model.joblib

For dataset exploration and automatic feature fallback, also place:

fraudTest.csv

in the project directory.

For the performance page, generated evaluation files can be placed inside:

fraud_model_output/

📊 Model Evaluation

The application supports common classification evaluation outputs:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

ROC Curve

Classification Report

Specific accuracy and F1 values are intentionally not hard-coded in this README because they depend on the model artifacts generated for the project.

🔮 Live Prediction Logic

When a transaction is submitted, the application:

Collects the entered feature values.

Builds a one-row Pandas DataFrame.

Passes the DataFrame to the trained model.

Reads the model prediction.

Normalizes common fraud labels such as 1, true, yes, fraud, and fraudulent.

Displays either a fraud warning or safe-transaction message.

Displays model confidence when probability prediction is supported.

🚀 Deployment

The application can be deployed to a Streamlit-compatible hosting platform.

Before deployment, verify that the repository contains all files required by the application:

app.py
requirements.txt
fraud_model.joblib
fraudTest.csv
fraud_model_output/

Set the application entry point to:

app.py

🔒 Security Considerations

This project is intended as an educational/internship demonstration.

For production use:

Do not hard-code usernames or passwords.

Store secrets using a secure secrets manager.

Validate and sanitize user inputs.

Protect sensitive transaction data.

Restrict access to model and dataset files.

Monitor model performance after deployment.

Retrain the model when transaction patterns change.

Add proper authentication and authorization.

Do not treat model predictions as a final financial decision without appropriate review.

📌 Limitations

Machine-learning fraud detection has practical limitations:

Fraud patterns can change over time.

False positives and false negatives are possible.

Model confidence is not necessarily the same as real-world certainty.

Performance depends on the quality and representativeness of the training data.

Live prediction inputs must match the features and data types expected by the trained model.

This application should therefore be considered a fraud-risk classification tool, not a guaranteed fraud detector.

🔮 Future Enhancements

Possible improvements include:

Real-time transaction monitoring

Real-time fraud alerts

Advanced authentication

Role-based access control

Transaction history

Fraud probability/risk scoring

Feature-importance visualization

Precision-recall curve

Model comparison dashboard

Automated model retraining

Database integration

API-based prediction service

Cloud deployment

Data/model drift monitoring
>>>>>>> 5567077 (Update professional README)
