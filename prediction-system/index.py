from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Step 1: Receive data
    content = request.get_json()
    data = content['features']  # Incoming data structured per day with product sales

    # Step 2: Prepare the data for training
    records = []
    for day, products in data.items():
        for product, sales in products.items():
            records.append({'Day': day, 'Product': product, 'Sales': sales})

    df = pd.DataFrame(records)

    # Encoding data
    onehot_encoder = OneHotEncoder()
    label_encoder = LabelEncoder()
    
    # Prepare features and labels
    X = onehot_encoder.fit_transform(df[['Day', 'Product']])
    y = df['Sales'].values
    print(df)
    print(X)

    # Step 3: Train the model
    model = RandomForestClassifier(n_estimators=80, random_state=42)
    model.fit(X, y)

    # Step 4: Make predictions using the trained model
    predictions = []
    for day in set(df['Day']):
        day_df = pd.DataFrame([{'Day': day, 'Product': p} for p in set(df['Product'])])
        day_encoded = onehot_encoder.transform(day_df)
        # Make sure every necessary column is included and transformed correctly.
        print(day_df.head())  # Check the DataFrame before encoding
        print(day_encoded.toarray())  # View the encoded features to confirm correctness
        pred_sales = model.predict_proba(day_encoded)[:, 1]  # Get probabilities for class 1 (e.g., top-selling)
        print(pred_sales)
        top_product_index = np.argmax(pred_sales)
        top_product = day_df.iloc[top_product_index]['Product']
        top_sales_probability = pred_sales[top_product_index]

        predictions.append({
            'day': day,
            'predicted_top_selling_product': top_product,
            'probability': f"{top_sales_probability:.2%}"
        })

    # Step 5: Send response
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
