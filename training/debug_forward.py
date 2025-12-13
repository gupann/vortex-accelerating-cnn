import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist

np.set_printoptions(precision=6, suppress=True)


def build_model():
    """Functional model matching your Vortex CNN."""
    inputs = keras.Input(shape=(28, 28, 1), name="input")

    x = layers.Conv2D(
        8,
        (3, 3),
        strides=1,
        padding="valid",
        name="conv2d",
    )(inputs)

    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, name="maxpool")(x)
    x = layers.ReLU(name="relu")(x)
    x = layers.Flatten(name="flatten")(x)
    logits = layers.Dense(10, activation=None, name="dense")(x)

    model = models.Model(inputs=inputs, outputs=logits, name="fashion_cnn")
    return model


def stats(name, arr):
    print(f"{name} shape: {arr.shape}")
    print(f"{name} min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}")


def main(idx=0):
    # 1. Load test data
    (_, _), (x_test, y_test) = fashion_mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    x = x_test[idx:idx+1, ..., None]   # (1,28,28,1)
    label = int(y_test[idx])

    # 2. Build and load weights
    model = build_model()
    model.load_weights("fashion_cnn.weights.h5")

    conv = model.get_layer("conv2d")
    pool = model.get_layer("maxpool")
    relu = model.get_layer("relu")
    dense = model.get_layer("dense")

    # 3. Sub-models for each layer’s output
    conv_model = keras.Model(model.inputs, conv.output)
    pool_model = keras.Model(model.inputs, pool.output)
    relu_model = keras.Model(model.inputs, relu.output)
    logits_model = keras.Model(model.inputs, dense.output)

    conv_out = conv_model(x).numpy()[0]     # (26,26,8) NHWC
    pool_out = pool_model(x).numpy()[0]     # (13,13,8)
    relu_out = relu_model(x).numpy()[0]     # (13,13,8)
    logits = logits_model(x).numpy()[0]   # (10,)
    probs = keras.activations.softmax(logits).numpy()
    pred = int(np.argmax(probs))

    print("=== Python reference ===")
    print(f"Index: {idx}")
    print(f"Label: {label}")
    print(f"Pred : {pred}")
    print("probs:", probs)

    # Conv
    stats("conv_out", conv_out)
    print("conv_out[0,0,:8] =", conv_out[0, 0, :8])

    # Pool
    stats("pool_out", pool_out)
    print("pool_out[0,0,:8] =", pool_out[0, 0, :8])

    # ReLU
    stats("relu_out", relu_out)
    print("relu_out[0,0,:8] =", relu_out[0, 0, :8])

    # Logits
    stats("logits", logits)
    print("logits =", logits)


if __name__ == "__main__":
    main(idx=0)
