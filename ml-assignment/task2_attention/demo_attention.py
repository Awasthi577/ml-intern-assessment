import numpy as np
from attention import scaled_dot_product_attention

if __name__ == "__main__":
    print("--- Scaled Dot-Product Attention Demo ---")

    np.random.seed(42)

    batch_size = 2
    seq_len = 3
    d_k = 4
    d_v = 5

    Q = np.random.rand(batch_size, seq_len, d_k).astype(np.float32)
    K = np.random.rand(batch_size, seq_len, d_k).astype(np.float32)
    V = np.random.rand(batch_size, seq_len, d_v).astype(np.float32)

    print(f"\nInput Shapes:\nQ: {Q.shape}\nK: {K.shape}\nV: {V.shape}")

    print("\n1. Attention WITHOUT mask")
    output, weights = scaled_dot_product_attention(Q, K, V)
    print("\nOutput (batch 0):\n", output[0])
    print("\nWeights (batch 0):\n", weights[0])
    print("Rows sum to 1? →", np.allclose(np.sum(weights, axis=-1), 1.0))

    print("\n2. Attention WITH mask")
    mask = np.ones((batch_size, seq_len, seq_len))
    mask[0, :, 2] = 0

    output_masked, weights_masked = scaled_dot_product_attention(Q, K, V, mask)
    print("\nMasked Weights (batch 0):\n", weights_masked[0])
    print("Last column should be zeros.")
