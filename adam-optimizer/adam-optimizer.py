import numpy as np

def adam_step(param, grad, m, v, t,
              lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Perform one Adam optimization step.

    Parameters
    ----------
    param : float or np.ndarray
        Current parameter(s) θ_{t-1}
    grad : float or np.ndarray
        Current gradient g_t
    m : float or np.ndarray
        First moment estimate m_{t-1}
    v : float or np.ndarray
        Second moment estimate v_{t-1}
    t : int
        Time step (1-based)
    lr : float
        Learning rate α
    beta1 : float
        Exponential decay rate for first moment
    beta2 : float
        Exponential decay rate for second moment
    eps : float
        Small constant for numerical stability

    Returns
    -------
    param_new, m_new, v_new : same type/shape as inputs
    """

    # Convert to numpy arrays for vectorized ops
    param = np.asarray(param, dtype=float)
    grad = np.asarray(grad, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)

    # 1. Update biased moments
    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * (grad ** 2)

    # 2. Bias correction
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)

    # 3. Parameter update
    param_new = param - lr * m_hat / (np.sqrt(v_hat) + eps)

    return param_new, m_new, v_new
