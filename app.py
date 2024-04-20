from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load('medical_condition_prediction_model.pkl')

# Define route for the web form
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Receive user input
        name = request.form['name']
        gender = request.form['gender']
        blood_group = request.form['blood_group']
        
        # Predict medical condition
        user_data = {'Name': name, 'Gender': gender, 'Blood Group Type': blood_group}
        user_data_df = pd.DataFrame([user_data])  # Create a DataFrame
        user_data_array = user_data_df[['Gender', 'Blood Group Type']]  # Select the required columns
        medical_condition = model.predict(user_data_array)[0]

        # Store user data along with predicted medical condition
        user_data['Medical Condition'] = medical_condition
        pd.DataFrame([user_data]).to_csv('user_data.csv', mode='a', index=False, header=False)

        # Display prediction result
        return render_template('result.html', name=name, gender=gender, blood_group=blood_group, medical_condition=medical_condition)
    
    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)
