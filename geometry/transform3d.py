# -*- coding: utf-8 -*-
# Filename: transform3d.py
""" Module for 3-D geometric transformations.
"""
import numpy as np

def translate(points, amount):
    if points.size == 3:
        out = points + amount
    else:
        out = points + np.tile(amount, (points.shape[0], 1))
    return out


def rotate(px, py, pz, order='xyx',
           alpha=0, beta=0, gamma=0,
           dtype='float32'):
    """Euler rotation.
    """
    s1 = np.sin(alpha)
    s2 = np.sin(beta)
    s3 = np.sin(gamma)
    c1 = np.cos(alpha)
    c2 = np.cos(beta)
    c3 = np.cos(gamma)

    Ax = np.array([[1,    0,   0],
                   [0,   c1,  s1],
                   [0,  -s1,  c1]])
    Ay = np.array([[c2,   0, -s2],
                   [ 0,   1,   0],
                   [s2,   0,  c2]])
    Az = np.array([[ c3, s3,   0],
                   [-s3, c3,   0],
                   [  0,  0,   1]])

    # Classic Euler rotations:
    if order == 'xzx':
        A = reduce(np.dot, [Ax, Az, Ax])
    elif order == 'xyx':
        A = reduce(np.dot, [Ax, Ay, Ax])
    elif order == 'yxy':
        A = reduce(np.dot, [Ay, Ax, Ay])
    elif order == 'yzy':
        A = reduce(np.dot, [Ay, Az, Ay])
    elif order == 'zyz':
        A = reduce(np.dot, [Az, Ay, Az])
    elif order == 'zxz':
        A = reduce(np.dot, [Az, Ax, Az])

    # Talt-Bryan rotations:
    if order == 'xyz':
        A = reduce(np.dot, [Ax, Ay, Az])
    elif order == 'xzy':
        A = reduce(np.dot, [Ax, Az, Ay])
    elif order == 'yxz':
        A = reduce(np.dot, [Ay, Ax, Az])
    elif order == 'yzx':
        A = reduce(np.dot, [Ay, Az, Ax])
    elif order == 'zxy':
        A = reduce(np.dot, [Az, Ax, Ay])
    elif order == 'zyx':
        A = reduce(np.dot, [Az, Ay, Ax])

    return np.dot(points, A).astype(dtype)


def rotate(points, order='xyx',
           alpha=0, beta=0, gamma=0,
           dtype='float32'):
    """Euler rotation.
    """
    s1 = np.sin(alpha)
    s2 = np.sin(beta)
    s3 = np.sin(gamma)
    c1 = np.cos(alpha)
    c2 = np.cos(beta)
    c3 = np.cos(gamma)

    Ax = np.array([[1,    0,   0],
                   [0,   c1,  s1],
                   [0,  -s1,  c1]])
    Ay = np.array([[c2,   0, -s2],
                   [ 0,   1,   0],
                   [s2,   0,  c2]])
    Az = np.array([[ c3, s3,   0],
                   [-s3, c3,   0],
                   [  0,  0,   1]])

    # Classic Euler rotations:
    if order == 'xzx':
        A = reduce(np.dot, [Ax, Az, Ax])
    elif order == 'xyx':
        A = reduce(np.dot, [Ax, Ay, Ax])
    elif order == 'yxy':
        A = reduce(np.dot, [Ay, Ax, Ay])
    elif order == 'yzy':
        A = reduce(np.dot, [Ay, Az, Ay])
    elif order == 'zyz':
        A = reduce(np.dot, [Az, Ay, Az])
    elif order == 'zxz':
        A = reduce(np.dot, [Az, Ax, Az])

    # Talt-Bryan rotations:
    if order == 'xyz':
        A = reduce(np.dot, [Ax, Ay, Az])
    elif order == 'xzy':
        A = reduce(np.dot, [Ax, Az, Ay])
    elif order == 'yxz':
        A = reduce(np.dot, [Ay, Ax, Az])
    elif order == 'yzx':
        A = reduce(np.dot, [Ay, Az, Ax])
    elif order == 'zxy':
        A = reduce(np.dot, [Az, Ax, Ay])
    elif order == 'zyx':
        A = reduce(np.dot, [Az, Ay, Ax])

    return np.dot(points, A).astype(dtype)
