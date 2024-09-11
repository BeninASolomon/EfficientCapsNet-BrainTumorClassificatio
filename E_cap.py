import tensorflow as tf
import os
from tensorflow.keras import layers, models
import cv2 as cv
import numpy as np
# Data paths
dataset_path = 'Data//'

# Load images and labels
images_ct = []
labels = []
org_img=[]

for r, d, f in os.walk(dataset_path):
    for file in f:
        images_ct.append(os.path.join(r, file))
        if r == "Data//glioma":
            labels.append(0)
        elif r == "Data//meningioma":
            labels.append(1)
        elif r == "Data//notumor":
            labels.append(2)
        elif r == "Data//pituitary":
            labels.append(3)
labels=np.array(labels).astype('float32')
for i in range(0,len(images_ct)):
    img = cv.imread(images_ct[i])
    resized_image = cv.resize(img, (128, 128))
    gray_img = cv.cvtColor(resized_image, cv.COLOR_BGR2GRAY)
    org_img.append(gray_img)
org_img = np.expand_dims(np.array(org_img), axis=-1)
org_img=np.array(org_img).astype('float32')
class EfficientCapsNet(tf.keras.Model):
    def __init__(self, input_shape, n_classes):
        super(EfficientCapsNet, self).__init__()
        self.conv1 = layers.Conv2D(64, kernel_size=3, strides=1, activation='relu', input_shape=input_shape)
        self.conv2 = layers.Conv2D(128, kernel_size=3, strides=1, activation='relu')
        self.primary_caps = layers.Conv2D(256, kernel_size=3, strides=2, activation='relu')
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(256, activation='relu')
        self.capsule_layer = layers.Dense(n_classes, activation='softmax') 

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.primary_caps(x)
        x = self.flatten(x)
        x = self.dense(x)
        output = self.capsule_layer(x)
        return output

# Example usage:
input_shape = (128, 128, 1)  # Example input shape
n_classes = 4  # Example number of leukemia classes
caps_net = EfficientCapsNet(input_shape, n_classes)
caps_net.build((None, 128, 128, 1))
caps_net.summary()

# Compile the model before training
optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.1,nesterov=True)
caps_net.compile(optimizer=optimizer, 
                 loss='sparse_categorical_crossentropy', 
                 metrics=['accuracy'])
# Train the model
history = caps_net.fit(org_img, labels, epochs=100, batch_size=32)
trainPredict = caps_net.predict(org_img)
y_pred=np.round(abs(trainPredict))
