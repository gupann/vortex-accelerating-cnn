import os
import numpy as np

# Ensure output directory exists
os.makedirs("weights", exist_ok=True)

# Load Keras saved weights
conv_W = np.load("conv2d_W.npy")     # shape (3,3,1,8)
conv_b = np.load("conv2d_b.npy")     # shape (8,)
fc_W = np.load("dense_W.npy")      # shape (1352,10)
fc_b = np.load("dense_b.npy")      # shape (10,)

# ---- Conv Weights Reorder ----
# Keras:  (Ky, Kx, C_in, C_out)
# Vortex: (C_out, C_in, Ky, Kx)
Ky, Kx, C_in, C_out = conv_W.shape
assert (Ky, Kx) == (3, 3), f"Expected 3x3 conv, got {Ky}x{Kx}"

conv_reordered = np.zeros((C_out, C_in, Ky, Kx), dtype=np.float32)

# conv_reordered[oc, ic, ky, kx] = conv_W[ky, kx, ic, oc]
for oc in range(C_out):
    for ic in range(C_in):
        for ky in range(Ky):
            for kx in range(Kx):
                conv_reordered[oc, ic, ky, kx] = conv_W[ky, kx, ic, oc]

# Flatten in row-major order to match:
# wt_idx = ((oc * C_in + ic) * 3 + ky) * 3 + kx
conv_flat = conv_reordered.ravel(order="C")

# Save Conv Weights + Bias
conv_flat.astype(np.float32).tofile("weights/conv1_w.bin")
conv_b.astype(np.float32).tofile("weights/conv1_b.bin")

# ---- FC Weights Reorder ----
# Keras Dense: W is (in_dim, out_dim)
# Vortex FC expects row-major W[o][i] => (out_dim, in_dim)
fc_W_t = fc_W.T.astype(np.float32)   # (10,1352)

fc_W_t.ravel(order="C").tofile("weights/fc_w.bin")
fc_b.astype(np.float32).tofile("weights/fc_b.bin")

print("Export complete!")
print(f"conv_W: {conv_W.shape} -> exported conv1_w.bin floats: {conv_flat.size}, conv1_b.bin floats: {conv_b.size}")
print(f"fc_W:   {fc_W.shape} -> exported fc_w.bin floats: {fc_W_t.size}, fc_b.bin floats: {fc_b.size}")
