import numpy as np 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras.datasets import boston_housing 
import matplotlib.pyplot as plt 
 
# Step 2: Load the Boston Housing dataset 
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data() 
 
print("Training data shape:", train_data.shape) 
print("Test data shape:", test_data.shape) 
 
# Step 3: Normalize the data (important for Neural Networks) 
mean = train_data.mean(axis=0) 
std = train_data.std(axis=0) 
 
train_data = (train_data - mean) / std 
test_data = (test_data - mean) / std 

# Step 4: Build the Neural Network model 
model = keras.Sequential([ 
    layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)), 
    layers.Dense(64, activation='relu'), 
    layers.Dense(1)  # Output layer for regression 
]) 
 
# Step 5: Compile the model 
model.compile( 
    optimizer='adam', 
    loss='mse', 
    metrics=['mae'] 
) 
 
# Step 6: Train the model 
history = model.fit( 
    train_data, 
    train_targets, 
    epochs=100, 
    batch_size=16, 
    validation_split=0.2, 
    verbose=1 
) 
 
# Step 7: Evaluate the model 
test_loss, test_mae = model.evaluate(test_data, test_targets) 
print("\nTest MAE:", test_mae)
# Step 8: Make predictions 
predictions = model.predict(test_data) 
 
print("\nFirst 5 predictions:") 
print(predictions[:5]) 
 
print("\nActual house prices:") 
print(test_targets[:5]) 
 
# Step 9: Plot Training and Validation Loss 
plt.figure(figsize=(8, 5)) 
plt.plot(history.history['loss'], label='Training Loss') 
plt.plot(history.history['val_loss'], label='Validation Loss') 
plt.xlabel('Epochs') 
plt.ylabel('Mean Squared Error') 
plt.title('Training vs Validation Loss') 
plt.legend() 
plt.show()