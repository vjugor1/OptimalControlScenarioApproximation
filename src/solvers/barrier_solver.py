import numpy as np


def objective_barrier_multiple(c, x, constr_foos, r):
    """
    Returns the value of penalized objective function. Penalization ensures the constraint satistaction
    c^\top x + r * max(0, constr_foo(x))**2
    args:
        x(n,) array-like floats: control variable (optimization variable)
        c(n,) array-like floats: vector of costs c
        constr_foo(x) function: g(x) from a constraint of form g(x) <= 0
        r float: penalization parameter
    """
    return np.dot(c, x[: len(c)]) + r * sum(
        [np.max((0.0, constr_foo(x))) ** 2 for constr_foo in constr_foos]
    )


def objective_barrier_multiple_grad(
    c, x, constr_foos, constr_foos_grads, r, eta_var=True
):
    """
    Returns the gradient of penalized objective function at x.
    args:
        x(n,) array-like floats: control variable (optimization variable)
        c(n,) array-like floats: vector of costs c
        constr_foo(x) function: g(x) from a constraint of form g(x) <= 0
        r float: penalization parameter
    """
    if eta_var:
        return np.hstack((c, np.array([0.0, 0.0]))) + 2 * r * sum(
            [
                np.max((0.0, constr_foo(x))) * constr_foo_grad(x)
                for constr_foo, constr_foo_grad in zip(constr_foos, constr_foos_grads)
            ]
        )
    else:
        return c + 2 * r * sum(
            [
                np.max((0.0, constr_foo(x))) * constr_foo_grad(x)
                for constr_foo, constr_foo_grad in zip(constr_foos, constr_foos_grads)
            ]
        )
