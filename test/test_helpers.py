from jax.numpy import asarray, zeros_like

def eval_numerical_gradient_complex(f, x, h=1e-5):
    """
    Numerically approximate the gradient of a real-valued function f(x),
    where x is complex-valued. Uses the Wirtinger convention.

    Parameters:
        f : function mapping complex array to real scalar
        x : complex array (jax.numpy)
        h : finite difference step size (float)

    Returns:
        grad : complex-valued gradient (Wirtinger)
    """
    x = asarray(x)
    flat_x = x.ravel()
    grad = []

    for i in range(flat_x.size):
        ei = zeros_like(flat_x).at[i].set(1.0)

        fpos = f((flat_x + h * ei).reshape(x.shape))
        fneg = f((flat_x - h * ei).reshape(x.shape))
        f_up = f((flat_x + 1j * h * ei).reshape(x.shape))
        f_dn = f((flat_x - 1j * h * ei).reshape(x.shape))

        df = ((fpos - fneg) - 1j * (f_up - f_dn)) / (4 * h)
        grad.append(df)

    return 2*asarray(grad).reshape(x.shape).conj()