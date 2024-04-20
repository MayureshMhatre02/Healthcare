import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load the dataset
data = pd.read_csv('healthcare_dataset.csv')

# Data preprocessing with OneHotEncoder for categorical variables
categorical_features = ['Gender', 'Blood Group Type']
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot",
                                  one_hot,
                                  categorical_features)],
                                remainder='passthrough')

# Split data into features and target variable
X = data[categorical_features]  # Features
y = data['Medical Condition']   # Target variable

# Create a Pipeline
model = Pipeline([
    ('transformer', transformer),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
model.fit(X, y)

# Save the trained model to a file
joblib.dump(model, 'medical_condition_prediction_model.pkl')
