import numpy as np
from tensorflow.keras.datasets import fashion_mnist

(_, _), (x_test, y_test) = fashion_mnist.load_data()

# Normalize
x_test = x_test.astype("float32") / 255.0

# Pick any test image index
idx = 0
img = x_test[idx]  # shape (28,28)

# Save as float32 binary (CHW with C=1)
img.astype(np.float32).tofile("images/test_image.bin")

print(f"Saved test image {idx}, label={y_test[idx]}")
