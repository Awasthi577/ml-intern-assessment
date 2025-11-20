import numpy as np

def softmax(x):
    """Standard softmax over the last axis."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / (np.sum(e_x, axis=-1, keepdims=True) + 1e-9)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention.
    Attention(Q, K, V) = softmax( (QK^T)/sqrt(d_k) ) @ V
    """
    Q = Q.astype(np.float32, copy=False)
    K = K.astype(np.float32, copy=False)
    V = V.astype(np.float32, copy=False)

    assert Q.shape[-1] == K.shape[-1], "Q/K dimension mismatch"
    assert K.shape[-2] == V.shape[-2], "Key/Value seq_len mismatch"

    d_k = Q.shape[-1]

    K_T = K.swapaxes(-2, -1)
    scores = np.matmul(Q, K_T)
    scores /= np.sqrt(d_k)

    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    attention_weights = softmax(scores)
    output = np.matmul(attention_weights, V)

    return output, attention_weights
