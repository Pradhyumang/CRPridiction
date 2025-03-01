# paymentHistroy      => 35%  ==>315 max
# creditUtilization   => 30%  ==>270 max
# Number of Account   => 15%  ==>135 max
# eduaction Level     => 10%  ==>90 max
# employment Status   => 10%  ==>90 max
# -----------------------100%---900 max
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
dataMaxPaymentHistroy= max(data['Payment History'])
print(dataMaxPaymentHistroy ,'<- max')
# Calculate credit scores using the complete FICO formula
credit_scores = []

for index, row in data.iterrows():
    payment_history = row['Payment History']
    credit_utilization_ratio = row['Credit Utilization Ratio']
    number_of_credit_accounts = row['Number of Credit Accounts']
    education_level = row['Education Level']
    employment_status = row['Employment Status']

    # Apply the FICO formula to calculate the credit score
    # credit_score = (payment_history * 0.35) + (credit_utilization_ratio * 0.30) + (number_of_credit_accounts * 0.15) + (education_level * 0.10) + (employment_status * 0.10)
    credit_score = ((payment_history * 315)/dataMaxPaymentHistroy) + ((1-credit_utilization_ratio) * 270) + ((number_of_credit_accounts * 135)/15) + (education_level * 22.5) + (employment_status * 45)
    credit_scores.append(credit_score)

# Add the credit scores as a new column to the DataFrame
data['Credit Score'] = credit_scores

# Perform KMeans clustering
X = data[['Credit Score']]
# print(X,"Xrest")

kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
kmeans.fit(X)
data['Segment'] = kmeans.labels_
data['Segment'] = data['Segment'].map({2: 'Poor', 0: 'Fair', 1: 'Good', 3: "Excellent",4:"very Good"})
data['Segment'] = data['Segment'].astype('category')
print(data)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    payment_history = float(request.form['payment_history'])
    credit_utilization_ratio = float(request.form['credit_utilization_ratio'])
    number_of_credit_account_entered = int(request.form['number_of_credit_accounts'])
    education_level = int(request.form['education_level'])
    employment_status = int(request.form['employment_status'])
    number_of_credit_accounts=15 if number_of_credit_account_entered>=15 else number_of_credit_account_entered
    # print("====>",number_of_credit_accounts)

    # Calculate credit score
    
    # credit_score = (payment_history * 0.35) + (credit_utilization_ratio * 0.30) + (number_of_credit_accounts * 0.15) + (education_level * 0.10) + (employment_status * 0.10)
    credit_score = ((payment_history * 315)/dataMaxPaymentHistroy) + ((1-credit_utilization_ratio) * 270) + ((number_of_credit_accounts * 135)/15) + (education_level * 22.5) + (employment_status * 45)
    # print(credit_score,"<=========Credit Score by input")
    # Determine segment based on credit score
    if credit_score >= 800:
        segment_name = 'Excellent'
    elif credit_score >=740 and credit_score<=799:
        segment_name = 'Very Good'
    elif credit_score >=670 and credit_score<=739:
        segment_name = 'Good'
    elif credit_score >=580 and credit_score<=669:
        segment_name = 'Fair'
    else:
        segment_name = 'Poor'
    return render_template('result.html', credit_score=credit_score, segment=segment_name)

@app.route('/allData', methods=['POST'])
def allData():
     # Visualize the distribution of credit scores and segments
    fig = px.scatter(data, x='Credit Score', color='Segment', title='Credit Score Distribution by Segment')
    fig.update_layout(xaxis_title='Credit Score', yaxis_title='Frequency')
    graph_json = fig.to_json()
    return render_template('allData.html', graph_json=graph_json)


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

