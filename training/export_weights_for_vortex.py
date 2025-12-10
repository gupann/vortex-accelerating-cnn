import numpy as np

# --------------------------
# Load Keras saved weights
# --------------------------
conv_W = np.load("conv2d_W.npy")     # shape (3,3,1,8)
conv_b = np.load("conv2d_b.npy")     # shape (8,)
fc_W   = np.load("dense_W.npy")      # shape (1352,10)
fc_b   = np.load("dense_b.npy")      # shape (10,)

# --------------------------
# Reorder Conv Weights
# Keras: (K, K, C_in, C_out)
# Vortex: (C_out, C_in, K, K)
# --------------------------

K = 3
C_in = 1
C_out = 8

# conv_W_keras[kx, ky, ic, oc]
# convert to conv_W_vortex[oc, ic, ky, kx]

conv_reordered = np.zeros((C_out, C_in, K, K), dtype=np.float32)

for oc in range(C_out):
    for ic in range(C_in):
        for ky in range(K):
            for kx in range(K):
                conv_reordered[oc, ic, ky, kx] = conv_W[ky, kx, ic, oc]

# Flatten in correct order
conv_flat = conv_reordered.flatten()  # matches your kernel indexing

# --------------------------
# Save Conv Weights + Bias
# --------------------------

conv_flat.astype(np.float32).tofile("weights/conv1_w.bin")
conv_b.astype(np.float32).tofile("weights/conv1_b.bin")

# --------------------------
# FC Weights for Vortex
# Keras gives shape = (1352,10)
# We want = (10, 1352), because FC expects row-major [o][i]
# --------------------------

# Keras W: [in_dim, out_dim]
# We need W[o][i]

fc_W_t = fc_W.T  # shape becomes (10,1352)

fc_W_t.astype(np.float32).tofile("weights/fc_w.bin")
fc_b.astype(np.float32).tofile("weights/fc_b.bin")

print("Export complete!")
