import os
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

# Load Fashion-MNIST test split
(_, _), (x_test, y_test) = fashion_mnist.load_data()

# Normalize to float32 in [0, 1]
x_test = x_test.astype(np.float32) / 255.0

N = 100  # number of test images to export

# Path to your demo CNN folder (from host side)
DEMO_DIR = "../demo/CNN"
IMG_DIR = os.path.join(DEMO_DIR, "images")

os.makedirs(IMG_DIR, exist_ok=True)

labels_path = os.path.join(IMG_DIR, "test_labels_100.txt")

with open(labels_path, "w") as lf:
    for i in range(N):
        # Image shape: (28, 28)
        img = x_test[i]

        # Explicit CHW layout: (C=1, H=28, W=28)
        img_chw = img.reshape(1, 28, 28)

        img_path = os.path.join(IMG_DIR, f"test_image_{i}.bin")
        img_chw.tofile(img_path)

        lf.write(f"{int(y_test[i])}\n")

print(f"Exported {N} test images to {IMG_DIR}")
print(f"Labels written to {labels_path}")
