import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ReLU, Flatten, Dense

# Load Fashion-MNIST
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize & reshape
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

x_train = x_train.reshape(-1,28,28,1)
x_test  = x_test.reshape(-1,28,28,1)

# Build simplified CNN
model = Sequential([
    Conv2D(8, (3,3), strides=1, padding="valid", input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2), strides=2),
    ReLU(),
    Flatten(),
    Dense(10, activation=None)  # logits, softmax done later
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train small # epochs for class project
model.fit(x_train, y_train, epochs=50, batch_size=128, validation_split=0.1)

# Save weights
model.save_weights("fashion_cnn.weights.h5")

# Export numpy weights for Vortex
for layer in model.layers:
    weights = layer.get_weights()
    if len(weights) > 0:
        W = weights[0]
        b = weights[1]
        np.save(layer.name + "_W.npy", W)
        np.save(layer.name + "_b.npy", b)
