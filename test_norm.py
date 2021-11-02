
import timeit

setup1 = """
from tomopy.util.extern.normalize import normalize_bg
import numpy as np
x = np.random.rand(100, 2048, 2048,).astype('float32')
"""

setup = """
from tomopy import normalize_bg
import numpy as np
x = np.random.rand(100, 2048, 2048,).astype('float32')
"""


stmt = """
y = normalize_bg(x, ncore=8)
"""

print(timeit.timeit(
    setup=setup,
    stmt=stmt,
    number=1,
))

print(timeit.timeit(
    setup=setup1,
    stmt=stmt,
    number=1,
))
