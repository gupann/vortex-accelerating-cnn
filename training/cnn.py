import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models


def main():
    # 1. Load Fashion-MNIST
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # Normalize & reshape to (N, 28, 28, 1)
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_test = (x_test.astype("float32") / 255.0)[..., None]

    # 2. Build CNN that matches Vortex pipeline
    model = models.Sequential(
        [
            layers.Conv2D(
                8,
                (3, 3),
                strides=1,
                padding="valid",
                input_shape=(28, 28, 1),
                name="conv2d",
            ),
            layers.MaxPooling2D(pool_size=(2, 2), strides=2),
            layers.ReLU(),
            layers.Flatten(),
            layers.Dense(10, activation="softmax", name="dense"),
        ]
    )

    model.summary()

    # 3. Train on full dataset (60k train, 10% held out for val)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=30,
        batch_size=128,
        validation_split=0.1,
        verbose=2,
    )

    # 4. Evaluate on full test set (10k)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    # 5. Save Keras weights
    model.save_weights("fashion_cnn.weights.h5")

    # 6. Export weights in the .npy format
    conv_layer = model.get_layer("conv2d")
    dense_layer = model.get_layer("dense")

    conv_W, conv_b = conv_layer.get_weights()   # (3,3,1,8), (8,)
    dense_W, dense_b = dense_layer.get_weights()  # (1352,10), (10,)

    np.save("conv2d_W.npy", conv_W)
    np.save("conv2d_b.npy", conv_b)
    np.save("dense_W.npy", dense_W)
    np.save("dense_b.npy", dense_b)


if __name__ == "__main__":
    main()
