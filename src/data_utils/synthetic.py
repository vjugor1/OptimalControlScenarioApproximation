import numpy as np


def regular_polyhedron(J, tau):
    """Generate 2d regular polyhedron centered at (0, 0) with defined distance from origin to planes

    Args:
        J (int): Number of planes
        tau (float): Distance from origin to planes

    Returns:
        tuple: Matrix (J, 2) containing normals vector of planes, Tau (J,) vector of planes' constants
    """
    Gamma = []
    Beta = np.array([tau] * J)
    for j in range(J):
        Gamma.append([np.cos(2 * np.pi * j / J), np.sin(2 * np.pi * j / J)])
    Gamma = np.array(Gamma)
    return Gamma, Beta
