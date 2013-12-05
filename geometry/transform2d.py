# -*- coding: utf-8 -*-
# Filename: transform2d.py
""" Module for 2-D geometric transformations.
"""
import numpy as np

def translate(points, amount):
    if points.size == 2:
        out = points + amount
    else:
        out = points + np.tile(amount, (points.shape[0], 1))
    return out

def rotate(points, alpha=0, dtype='float32'):
    """2-D rotation.
    """
    s1 = np.sin(alpha)
    c1 = np.cos(alpha)
    A = np.array([[c1,  s1], [-s1,  c1]])
    return np.dot(points, A).astype(dtype)
