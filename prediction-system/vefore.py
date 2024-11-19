import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from flask import Flask, request, jsonify

# Data
data = {
    'Friday': {'Nasi Lemak': 100, 'Nasi Kandar': 40, 'Nasi Goreng Kampung': 80, 'Mee Goreng': 10, 'Nasi Ayam Geprek': 20},
    'Monday': {'Nasi Kandar': 40, 'Nasi Lemak': 80, 'Mee Goreng': 90, 'Nasi Ayam Geprek': 30, 'Nasi Goreng Kampung': 30},
    'Saturday': {'Nasi Ayam Geprek': 120, 'Nasi Lemak': 80, 'Mee Goreng': 30, 'Nasi Kandar': 60, 'Nasi Goreng Kampung': 50},
    'Sunday': {'Nasi Lemak': 90, 'Mee Goreng': 110, 'Nasi Kandar': 40, 'Nasi Ayam Geprek': 50, 'Nasi Goreng Kampung': 30},
    'Thursday': {'Nasi Kandar': 150, 'Mee Goreng': 60, 'Nasi Goreng Kampung': 70, 'Nasi Lemak': 160, 'Nasi Ayam Geprek': 40},
    'Tuesday': {'Nasi Ayam Geprek': 30, 'Nasi Kandar': 90, 'Nasi Lemak': 60, 'Nasi Goreng Kampung': 110, 'Mee Goreng': 40},
    'Wednesday': {'Nasi Goreng Kampung': 40, 'Nasi Ayam Geprek': 100, 'Mee Goreng': 20, 'Nasi Kandar': 10, 'Nasi Lemak': 60}
}

# Convert data to DataFrame and determine the top-selling product each day
records = []
for day, products in data.items():
    top_product = max(products, key=products.get)  # Get the product with the maximum sales
    records.append({'Day': day, 'TopProduct': top_product})
df = pd.DataFrame(records)

# One-hot encoding for the day
df_encoded = pd.get_dummies(df, columns=['Day'])
print(df_encoded)

# Prepare the features and target
X = df_encoded.drop('TopProduct', axis=1)
y = df_encoded['TopProduct']
print(X)
print('sini')
print(y)

# Encoding the labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save the model and encoder
joblib.dump(model, 'top_product_model.joblib')
joblib.dump(encoder, 'product_encoder.joblib')

# Set up the Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    content = request.get_json()
    features = content['features']  # e.g., 'Friday'
    days = list(features.keys())  # e.g., 'Friday'
    print(days)
    results = []
    
    for day in days:
        # Prepare input data for prediction
        input_df = pd.DataFrame([{'Day': day}])
        input_df = pd.get_dummies(input_df, columns=['Day'])

        # Ensure all columns are present
        missing_cols = set(X.columns) - set(input_df.columns)  # X.columns from the model training script
        for c in missing_cols:
            input_df[c] = 0
        input_df = input_df.reindex(columns=X.columns, fill_value=0)  # Reorder columns to match training data

        # Make prediction
        probabilities = model.predict_proba(input_df)[0]
        print(probabilities)
        predicted_index = probabilities.argmax()
        predicted_product = encoder.inverse_transform([predicted_index])[0]
        predicted_probability = probabilities[predicted_index]

        # Format the probability to two decimal places
        formatted_probability = f"{predicted_probability:.2%}"
        
        # Collect the result
        results.append({
            'day': day,
            'predicted_top_selling_product': predicted_product,
            'probability': formatted_probability
        })

    # Return all results
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)