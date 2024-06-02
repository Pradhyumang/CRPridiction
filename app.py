from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)
pio.templates.default = "plotly_white"

# Load data
data = pd.read_csv("credit_scoring.csv")

# Define the mapping for categorical features
education_level_mapping = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
employment_status_mapping = {'Unemployed': 0, 'Employed': 1, 'Self-Employed': 2}

# Apply mapping to categorical features
data['Education Level'] = data['Education Level'].map(education_level_mapping)
data['Employment Status'] = data['Employment Status'].map(employment_status_mapping)

# Calculate credit scores using the complete FICO formula
credit_scores = []

for index, row in data.iterrows():
    payment_history = row['Payment History']
    credit_utilization_ratio = row['Credit Utilization Ratio']
    number_of_credit_accounts = row['Number of Credit Accounts']
    education_level = row['Education Level']
    employment_status = row['Employment Status']

    # Apply the FICO formula to calculate the credit score
    credit_score = (payment_history * 0.35) + (credit_utilization_ratio * 0.30) + (number_of_credit_accounts * 0.15) + (education_level * 0.10) + (employment_status * 0.10)
    credit_scores.append(credit_score)

# Add the credit scores as a new column to the DataFrame
data['Credit Score'] = credit_scores

# Perform KMeans clustering
X = data[['Credit Score']]
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
kmeans.fit(X)
data['Segment'] = kmeans.labels_
data['Segment'] = data['Segment'].map({2: 'Very Low', 0: 'Low', 1: 'Good', 3: "Excellent"})
data['Segment'] = data['Segment'].astype('category')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    payment_history = float(request.form['payment_history'])
    credit_utilization_ratio = float(request.form['credit_utilization_ratio'])
    number_of_credit_accounts = int(request.form['number_of_credit_accounts'])
    education_level = int(request.form['education_level'])
    employment_status = int(request.form['employment_status'])

    # Calculate credit score
    credit_score = (payment_history * 0.35) + (credit_utilization_ratio * 0.30) + (number_of_credit_accounts * 0.15) + (education_level * 0.10) + (employment_status * 0.10)

    # Determine segment based on credit score
    if credit_score >= 800:
        segment_name = 'Excellent'
    elif credit_score >= 700:
        segment_name = 'Good'
    elif credit_score >= 600:
        segment_name = 'Low'
    else:
        segment_name = 'Very Low'

    # Visualize the distribution of credit scores and segments
    fig = px.scatter(data, x='Credit Score', color='Segment', title='Credit Score Distribution by Segment')
    fig.update_layout(xaxis_title='Credit Score', yaxis_title='Frequency')
    graph_json = fig.to_json()

    return render_template('result.html', credit_score=credit_score, segment=segment_name, graph_json=graph_json)

if __name__ == '__main__':
    app.run(debug=True)
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Assuming 'actual_scores' contains the actual credit scores from your dataset
    actual_scores = data['Credit Score']

    # Assuming 'predicted_scores' contains the predicted credit scores from your model
    predicted_scores = data['Credit Score']  # Replace this with the predicted scores from your model

    # Calculate metrics
    mae = mean_absolute_error(actual_scores, predicted_scores)
    mse = mean_squared_error(actual_scores, predicted_scores)
    rmse = mean_squared_error(actual_scores, predicted_scores, squared=False)  # RMSE requires 'squared=False'
    r2 = r2_score(actual_scores, predicted_scores)

    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("R-squared:", r2)

