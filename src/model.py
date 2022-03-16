import tensorflow.keras as keras

from src.mnist import MNIST
from src.utils.util_path import PathUtils


x_train, y_train, _, _ = MNIST.instance().load_with_preprocess()

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.save(PathUtils.model_path() + 'demo_classifier.h5')
