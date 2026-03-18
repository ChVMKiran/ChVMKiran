from numpy.random import seed 
seed(0xdeadbeef) 
 
import tensorflow as tf 
tf.random.set_seed(0xdeadbeef) 
 
from tensorflow import keras 
import matplotlib.pyplot as plt 
 
# Load IMDB dataset 
imdb = keras.datasets.imdb 
num_words = 20000 
 
(train_data, train_labels), (test_data, test_labels) = imdb.load_data( 
    seed=1,
    num_words=num_words 
) 
print(train_data[0]) 
print("Label:", train_labels[0]) 
 
# Get word index dictionary 
vocabulary = imdb.get_word_index() 
 
# Shift indices by 3 (reserve special tokens) 
vocabulary = {k: (v + 3) for k, v in vocabulary.items()} 
vocabulary["<PAD>"] = 0 
vocabulary["<START>"] = 1 
vocabulary["<UNK>"] = 2 
vocabulary["<UNUSED>"] = 3 
 
# Create reverse dictionary (number → word) 
index = dict((value, key) for (key, value) in vocabulary.items()) 
 
# Function to convert numbers back to words 
def decode_review(text): 
    return ' '.join([index.get(i, '?') for i in text]) 
 
# Print decoded review 
print(decode_review(train_data[0])) 
 
# Pad sequences to same length (256 words) 
train_data = keras.preprocessing.sequence.pad_sequences( 
    train_data, 
    value=vocabulary["<PAD>"], 
    padding='post',
        maxlen=256 
) 
 
test_data = keras.preprocessing.sequence.pad_sequences( 
    test_data, 
    value=vocabulary["<PAD>"], 
    padding='post', 
    maxlen=256 
) 
# Build the model 
model = keras.Sequential() 
 
# Embedding layer (word → vector) 
model.add(keras.layers.Embedding(len(vocabulary), 2, input_length=256)) 
 
# Flatten output 
model.add(keras.layers.Flatten()) 
 
# Add dropout to prevent overfitting 
model.add(keras.layers.Dropout(0.5)) 
 
# Hidden dense layer 
model.add(keras.layers.Dense(5)) 
 
# Output layer (binary classification) 
model.add(keras.layers.Dense(1, activation='sigmoid')) 
model.summary() 
 
# Compile model 
model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy'] 
) 
 
# Train model 
history = model.fit( 
    train_data, 
    train_labels, 
    epochs=5, 
    batch_size=100, 
    validation_data=(test_data, test_labels), 
    verbose=1 
) 
 
# Plot accuracy graph 
def plot_accuracy(history): 
    acc = history.history['accuracy'] 
    val_acc = history.history['val_accuracy'] 
    epochs = range(len(acc)) 
    plt.plot(epochs, acc, label='Training Accuracy') 
    plt.plot(epochs, val_acc, label='Validation Accuracy') 
    plt.title('Accuracy') 
    plt.xlabel('Epoch') 
    plt.legend() 
    plt.show() 
 
plot_accuracy(history)