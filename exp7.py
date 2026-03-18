import tensorflow as tf 
from tensorflow.keras.applications import VGG16 
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Flatten, Dense, Dropout 
from tensorflow.keras.utils import plot_model 
from tensorflow.keras.preprocessing.image import load_img 
# from google.colab import files 
import matplotlib.pyplot as plt 
#LOAD PRETRAINED VGG16 
base_model = VGG16( 
    weights='imagenet', 
    include_top=False, 
    input_shape=(224, 224, 3) 
) 
# FREEZE BASE MODEL LAYERS 
for layer in base_model.layers: 
    layer.trainable = False 
# ADD CUSTOM CLASSIFICATION LAYERS  
x = base_model.output 
x = Flatten()(x)
x = Dense(256, activation='relu')(x) 
x = Dropout(0.5)(x) 
output = Dense(3, activation='softmax')(x) 
# CREATE FINAL MODEL 
model = Model( 
    inputs=base_model.input, 
    outputs=output 
) 
# COMPILE MODEL 
model.compile( 
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy'] 
) 
# MODEL SUMMARY 
model.summary() 
#  VISUALIZE MODEL ARCHITECTURE 
plot_model( 
    model, 
    show_shapes=True, 
    show_layer_names=True 
) 
#UPLOAD IMAGE 
# uploaded = files.upload() 
# LOAD AND DISPLAY IMAGE 
img = load_img('mnist_sample.png', target_size=(224, 224)) 
plt.imshow(img) 
plt.axis('off') 
plt.show()
