import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load or define your data and products
data = {
    'Friday': {'Nasi Lemak': 10, 'Nasi Kandar': 4, 'Nasi Goreng Kampung': 8, 'Mee Goreng': 1, 'Nasi Ayam Geprek': 2},
    'Monday': {'Nasi Kandar': 4, 'Nasi Lemak': 8, 'Mee Goreng': 1, 'Nasi Ayam Geprek': 3, 'Nasi Goreng Kampung': 3},
    'Saturday': {'Nasi Ayam Geprek': 12, 'Nasi Lemak': 8, 'Mee Goreng': 3, 'Nasi Kandar': 3, 'Nasi Goreng Kampung': 5},
    'Sunday': {'Nasi Lemak': 9, 'Mee Goreng': 11, 'Nasi Kandar': 4, 'Nasi Ayam Geprek': 5},
    'Thursday': {'Nasi Kandar': 15, 'Mee Goreng': 6, 'Nasi Goreng Kampung': 7, 'Nasi Lemak': 16, 'Nasi Ayam Geprek': 4},
    'Tuesday': {'Nasi Ayam Geprek': 3, 'Nasi Kandar': 11, 'Nasi Lemak': 12, 'Nasi Goreng Kampung': 11, 'Mee Goreng': 4},
    'Wednesday': {'Nasi Goreng Kampung': 4, 'Nasi Ayam Geprek': 10, 'Mee Goreng': 2, 'Nasi Kandar': 1, 'Nasi Lemak': 6}
}
records = []
for day, products in data.items():
    for product, quantity in products.items():
        records.append({'day': day, 'product': product, 'quantity': quantity})
df = pd.DataFrame(records)

# Prepare your data
encoder = LabelEncoder()
df['product_id'] = encoder.fit_transform(df['product'])
X = pd.get_dummies(df[['day', 'product']], columns=['day', 'product'])
print(X)
y = df['product_id']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train your model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Saving the model for later use
joblib.dump(model, 'model.joblib')

# Prediction on a comprehensive test set
test_data = pd.DataFrame(records)  # Use the same records to ensure all combinations
X_test_full = pd.get_dummies(test_data, columns=['day', 'product'])
X_test_full = X_test_full.reindex(columns=X.columns, fill_value=0)
probabilities = model.predict_proba(X_test_full)
predicted_product_ids = probabilities.argmax(axis=1)
predicted_products = encoder.inverse_transform(predicted_product_ids)
predicted_probabilities = probabilities.max(axis=1)

# Determine the highest probability product for each day
top_products = {}
for i, row in test_data.iterrows():
    day = row['day']
    product = predicted_products[i]
    probability = predicted_probabilities[i]
    if day not in top_products or top_products[day][1] < probability:
        top_products[day] = (product, probability)
        
# Print the top product for each day
for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
    product, probability = top_products.get(day, ("No prediction", 0))
    print(f"Day: {day}, Top-Selling Product: {product}, Probability: {probability:.2%}")
