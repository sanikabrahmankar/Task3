import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Import and preprocess the dataset
# Make sure the filename exactly matches what you downloaded
df = pd.read_csv('House-Price-Dataset.csv') 

print("--- First 5 rows of dataset ---")
print(df.head())

# Define independent variable (X) and dependent variable (y)
X = df[['SquareFeet']] # This MUST be a 2D array (double brackets)
y = df['Price']        # This is a 1D Series

# 2. Split data into train-test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Fit a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# 4. Evaluate model using MAE, MSE, R-squared
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared Score: {r2:.4f}")

# 5. Interpret coefficients
print("\n--- Model Interpretation ---")
print(f"Coefficient (Slope): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Equation: Price = ({model.coef_[0]:.2f} * SquareFeet) + {model.intercept_:.2f}")

# 6. Plot regression line
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual Prices')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line (Predicted)')
plt.title('Simple Linear Regression: House Prices vs Square Footage')
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()