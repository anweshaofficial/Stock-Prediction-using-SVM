import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Load the dataset
df = pd.read_csv('RPOWER.NS.csv')


# Select features and target variable
X = df[['Open', 'High', 'Low', 'Volume']]
Y = df['Close']


# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# Feature scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
Y_train = sc_y.fit_transform(Y_train.values.reshape(-1, 1)).ravel()


# Initialize and train the SVM model
model = SVR(kernel='rbf')
model.fit(X_train, Y_train)
#rbf is Radial Basis Function






# Compute the average values for 'Open', 'High', 'Low', and 'Volume' columns
average_open = df['Open'].mean()
average_high = df['High'].mean()
average_low = df['Low'].mean()
average_volume = df['Volume'].mean()


# Use these averages as the example features
example_features = [[average_open, average_high, average_low, average_volume]]


# Scale features before making predictions
example_features = sc_X.transform(example_features)


# Predict the closing price using the model
# Assuming this is where you make predictions
predictions = model.predict(X_test)


# Before inverse transforming the predictions, reshape them to 2D
predictions_reshaped = predictions.reshape(-1, 1)
predictions = sc_y.inverse_transform(predictions_reshaped)  # Now this should work without error


predicted_close = model.predict(example_features)
# Ensure predicted_close is a 2D array before inverse transforming
predicted_close_reshaped = predicted_close.reshape(-1, 1)
predicted_close = sc_y.inverse_transform(predicted_close_reshaped)


mse = mean_squared_error(Y_test, predictions)
print(f"Mean Squared Error: {mse}")




#Actual Vs Predicted
plt.figure(figsize=(10, 6))
plt.plot(Y_test.values, label='Actual Close Price', color='blue', marker='o')
plt.plot(predictions, label='Predicted Close Price', color='red', linestyle='--', marker='x')
plt.title('Actual vs Predicted Close Prices')
plt.xlabel('Sample Index')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
# Since predicted_close is now a 2D array, access the first element as needed
print(f"Predicted Close Price: {predicted_close[0][0]}")


days_ahead = int(input("Enter the number of days ahead to predict: "))


# Predict the closing price for 'days_ahead' using the average features
predicted_closes = []
for _ in range(days_ahead):
    predicted_close = model.predict(example_features)
    predicted_close = sc_y.inverse_transform(predicted_close.reshape(-1, 1))
    predicted_closes.append(predicted_close[0][0])
    
    # Update 'example_features' based on the prediction for the next prediction
    # Assuming the closing price affects the next day's 'Open', 'High', 'Low', 'Volume' similarly
    # This is a simplification and may not reflect real-world stock movements accurately
    example_features = [[predicted_close[0][0], predicted_close[0][0], predicted_close[0][0], average_volume]]
    example_features = sc_X.transform(example_features)


# Plotting the predicted closing prices for the days ahead
plt.figure(figsize=(10, 6))
plt.plot(range(days_ahead), predicted_closes, label='Predicted Close Price', color='green', marker='x')
plt.title(f'Predicted Close Prices for the Next {days_ahead} Days')
plt.xlabel('Days from Today')
plt.ylabel('Predicted Stock Price')
plt.legend()
plt.show()
