import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)
all_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

@app.route('/train', methods=['POST'])
def train():
    # Receive data from POST request
    content = request.get_json()
    data = content['features']  # Expecting data to be in the same format as the original 'data' dictionary

    # Convert data to DataFrame and determine the top-selling product each day
    records = []
    for day, products in data.items():
        top_product = max(products, key=products.get)  # Get the product with the maximum sales
        records.append({'Day': day, 'TopProduct': top_product})
    df = pd.DataFrame(records)
    
    # One-hot encoding for the day
    df_encoded = pd.get_dummies(df, columns=['Day'])

    # Define manual indices for train and test
    train_indices = [0, 1, 2, 3, 4, 5, 6]  # Includes all products with at least one representation
    test_indices = [1, 3, 5]

    # Splitting the DataFrame using loc for label-based indexing
    X_train = df_encoded.loc[train_indices].drop('TopProduct', axis=1)
    X_test = df_encoded.loc[test_indices].drop('TopProduct', axis=1)
    y_train = df_encoded.loc[train_indices]['TopProduct']
    y_test = df_encoded.loc[test_indices]['TopProduct']

    # Encoding the labels
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train_encoded)

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Evaluate the accuracy
    accuracy = accuracy_score(y_test_encoded, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Save the model and encoder
    joblib.dump(model, 'top_product_model.joblib')
    joblib.dump(encoder, 'product_encoder.joblib')
    joblib.dump(X_train.columns.tolist(), 'X_columns.joblib')  # Save the feature names for later use

    # Optionally, returning results if needed
    return {
        'message': 'Model trained and evaluated successfully',
        'accuracy': accuracy
    }

@app.route('/predict', methods=['POST'])
def predict():
    # Load model, encoder, and X_columns
    model = joblib.load('top_product_model.joblib')
    encoder = joblib.load('product_encoder.joblib')
    X_columns = joblib.load('X_columns.joblib')  # Assuming X_columns contains all day columns

    content = request.get_json()
    features = content['features']  # e.g., {'Monday': some_value}
    results = []

    for day in all_days: # Iterate over all_days to maintain order
        if day in features.keys():
            # Prepare input data for prediction
            input_df = pd.DataFrame([{'Day': day}])
            input_df['Day'] = pd.Categorical(input_df['Day'], categories=all_days, ordered=True)
            input_df = pd.get_dummies(input_df, columns=['Day'])

            # Ensure all columns are present that were in training data
            missing_cols = set(X_columns) - set(input_df.columns)
            for c in missing_cols:
                input_df[c] = 0
            input_df = input_df.reindex(columns=X_columns, fill_value=0)

            # Make prediction
            probabilities = model.predict_proba(input_df)[0]
            predicted_index = probabilities.argmax()
            predicted_product = encoder.inverse_transform([predicted_index])[0]
            predicted_probability = probabilities[predicted_index]

            # Collect the result
            results.append({
                'day': day,
                'predicted_top_selling_product': predicted_product,
                'probability': f"{predicted_probability:.2%}"
            })

    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)
