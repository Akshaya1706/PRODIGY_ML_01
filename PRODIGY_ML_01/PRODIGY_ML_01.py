import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')

X = data[['BedroomAbvGr', 'FullBath', 'GrLivArea']]
y = data['SalePrice']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training R-squared: {train_score:.3f}")
print(f"Testing R-squared: {test_score:.3f}")

def predict_price(num_bedrooms, num_bathrooms, living_area):
    input_data = [[num_bedrooms, num_bathrooms, living_area]]
    input_data_scaled = scaler.transform(input_data)
    predicted_price = model.predict(input_data_scaled)
    return predicted_price[0]

bedrooms = int(input("Enter the number of bedrooms: "))
bathrooms = int(input("Enter the number of bathrooms: "))
living_area = int(input("Enter the living area in square feet: "))

predicted_price = predict_price(bedrooms, bathrooms, living_area)
print("Predicted price:", predicted_price)

y_pred = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs. Predicted Sale Price')
plt.show()
