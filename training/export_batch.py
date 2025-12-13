# export_batch.py
import os
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

N = 100          # start with 100 (later try 1000)
OUT_DIR = "../demo/CNN/images"

os.makedirs(OUT_DIR, exist_ok=True)


(_, _), (x_test, y_test) = fashion_mnist.load_data()
x_test = x_test.astype("float32") / 255.0

for i in range(N):
    img = x_test[i]              # (28,28)
    label = int(y_test[i])

    img.astype(np.float32).tofile(f"{OUT_DIR}/img_{i}.bin")
    with open(f"{OUT_DIR}/label_{i}.txt", "w") as f:
        f.write(str(label))

print(f"Exported {N} test images")
