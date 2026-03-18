import numpy as np 
from matplotlib import pyplot as plt 
from tensorflow.keras.datasets import mnist 
 
# Load MNIST dataset 
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
 
# Count training labels 
unique, counts = np.unique(y_train, return_counts=True) 
print("Train:", dict(zip(unique, counts))) 
 
# Count test labels 
unique, counts = np.unique(y_test, return_counts=True) 
print("\nTest labels:", dict(zip(unique, counts))) 
 
# Select 25 random images 
indexes = np.random.randint(0, x_train.shape[0], size=25) 
images = x_train[indexes] 
labels = y_train[indexes] 
 
# Plot images 
plt.figure(figsize=(5, 5)) 
 
for i in range(len(indexes)): 
    plt.subplot(5, 5, i + 1) 
    image = images[i] 
    plt.imshow(image, cmap="gray") 
    plt.axis("off")
    # Save and show figure 
 
plt.savefig("mnist_sample.png") 
plt.show() 
 
# Close all figures 
plt.close("all")