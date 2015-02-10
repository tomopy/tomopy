"""
To Generate phantoms. You can call the following functions with the
desired phantom shape as input :

- modified_shepp_logan
- shepp_logan
- yu_ye_wang

You can generate a custom phantom by specifying a list of
ellipsoid parameters by calling the phantom function.

Ellipsoid parameters are as follows:
- A : value inside the ellipsoid
- a, b, c : axis length of the ellipsoid (in % of the cube shape)
- x0, y0, z0 : position of the center (in % of the cube shape)
- phi, theta, psi : Euler angles defining the orientation (in degrees)

Alternatively, you can generate only one ellipsoid by calling
the ellipsoid function.

Exemple
-------
To generate a phantom cube of size 32 * 32 * 32 :

>>> from siddon.phantom import *
>>> my_phantom = shepp_logan((32, 32, 32))
>>> assert my_phantom[16, 16, 16] == -0.8

Notes
-----
You can take a look at those links for explanations:
http://en.wikipedia.org/wiki/Imaging_phantom
http://en.wikipedia.org/wiki/Ellipsoid
http://en.wikipedia.org/wiki/Euler_angles

This module is largely inspired by :
http://www.mathworks.com/matlabcentral/fileexchange/9416-3d-shepp-logan-phantom

Author
------
Nicolas Barbey

"""
import numpy as np

__all__ = ['phantom', 'shepp_logan', 'modified_shepp_logan', 'yu_ye_wang']

def phantom(shape, parameters_list, dtype=np.float64):
    """
    Generate a cube of given shape using a list of ellipsoid
    parameters.

    Inputs
    ------
    shape: tuple of ints
        Shape of the output cube.

    parameters_list: list of dictionaries
        List of dictionaries with the parameters defining the ellipsoids to
        include in the cube.

    dtype: data-type
        Data type of the output ndarray.

    Output
    ------
    cube: 3-dimensional ndarray
        A 3-dimensional ndarray filled with the specified ellipsoids.

    See Also
    --------
    shepp_logan : Generates the Shepp Logan phantom in any shape.
    modified_shepp_logan : Modified Shepp Logan phantom in any shape.
    yu_ye_wang : The Yu Ye Wang phantom in any shape.
    ellipsoid : Generates a cube filled with an ellipsoid of any shape.

    Notes
    -----
    http://en.wikipedia.org/wiki/Imaging_phantom
    """
    # instantiate ndarray cube
    cube = np.zeros(shape, dtype=dtype)
    # define coordinates
    coordinates = define_coordinates(shape)
    # recursively add ellipsoids to cube
    for parameters in parameters_list:
        ellipsoid(parameters, out=cube, coordinates=coordinates)
    return cube

def ellipsoid(parameters, shape=None, out=None, coordinates=None):
    """
    Generate a cube containing an ellipsoid defined by its parameters.
    If out is given, fills the given cube instead of creating a new
    one.
    """
    # handle inputs
    if shape is None and out is None:
        raise ValueError("You need to set shape or out")
    if out is None:
        out = np.zeros(shape)
    if shape is None:
        shape = out.shape
    if len(shape) == 1:
        shape = shape, shape, shape
    elif len(shape) == 2:
        shape = shape[0], shape[1], 1
    elif len(shape) > 3:
        raise ValueError("input shape must be lower or equal to 3")
    if coordinates is None:
        coordinates = define_coordinates(shape)
    # rotate coordinates
    coords = transform(coordinates, parameters)
    # recast as ndarray
    coords = [np.asarray(u) for u in coords]
    # reshape coordinates
    x, y, z = coords
    x.resize(shape)
    y.resize(shape)
    z.resize(shape)
    # fill ellipsoid with value
    out[(x ** 2 + y ** 2 + z ** 2) <= 1.] += parameters['A']
    return out

def rotation_matrix(p):
    """
    Defines an Euler rotation matrix from angles phi, theta and psi.

    Notes
    -----
    http://en.wikipedia.org/wiki/Euler_angles
    """
    cphi = np.cos(np.radians(p['phi']))
    sphi = np.sin(np.radians(p['phi']))
    ctheta = np.cos(np.radians(p['theta']))
    stheta = np.sin(np.radians(p['theta']))
    cpsi = np.cos(np.radians(p['psi']))
    spsi = np.sin(np.radians(p['psi']))
    alpha = [[cpsi * cphi - ctheta * sphi * spsi,
              cpsi * sphi + ctheta * cphi * spsi,
              spsi * stheta],
             [-spsi * cphi - ctheta * sphi * cpsi,
               -spsi * sphi + ctheta * cphi * cpsi,
               cpsi * stheta],
             [stheta * sphi,
              -stheta * cphi,
              ctheta]]
    return np.asarray(alpha)

def define_coordinates(shape):
    """
    Generate a tuple of coordinates in 3d with a given shape
    """
    mgrid = np.lib.index_tricks.nd_grid()
    cshape = np.asarray(1j) * shape
    x, y, z = mgrid[-1:1:cshape[0], -1:1:cshape[1], -1:1:cshape[2]]
    return x, y, z

def transform(coordinates, p):
    """
    Apply rotation, translation and rescaling to a 3-tuple of
    coordinates.
    """
    alpha = rotation_matrix(p)
    x, y, z = coordinates
    ndim = len(coordinates)
    out_coords = [sum([alpha[j, i] * coordinates[i] for i in xrange(ndim)])
                  for j in xrange(ndim)]
    M0 = [p['x0'], p['y0'], p['z0']]
    sc = [p['a'], p['b'], p['c']]
    out_coords = [(u - u0) / su for u, u0, su in zip(out_coords, M0, sc)]
    return out_coords

# specific phantom parameters

# mandatory parameters to define an ellipsoid
parameters_tuple = ['A', 'a', 'b', 'c', 'x0', 'y0', 'z0', 'phi', 'theta', 'psi']

# arrays
modified_shepp_logan_array = [
    [  1,  .6900,  .920,  .810,    0.,      0.,     0,     0,     0,     0],
    [-.8,  .6624,  .874,  .780,    0.,  -.0184,     0,     0,     0,     0],
    [-.2,  .1100,  .310,  .220,   .22,      0.,     0,   -18,     0,    10],
    [-.2,  .1600,  .410,  .280,  -.22,      0.,     0,    18,     0,    10],
    [ .1,  .2100,  .250,  .410,    0.,     .35,  -.15,     0,     0,     0],
    [ .1,  .0460,  .046,  .050,    0.,      .1,   .25,     0,     0,     0],
    [ .1,  .0460,  .046,  .050,    0.,     -.1,   .25,     0,     0,     0],
    [ .1,  .0460,  .023,  .050,  -.08,   -.605,     0,     0,     0,     0],
    [ .1,  .0230,  .023,  .020,    0.,   -.606,     0,     0,     0,     0],
    [ .1,  .0230,  .046,  .020,   .06,   -.605,     0,     0,     0,     0]]

shepp_logan_array = np.copy(modified_shepp_logan_array)
shepp_logan_array[0] = [1, -.98, -.02, -.02, .01, .01, .01, .01, .01, .01]

yu_ye_wang_array = [
    [  1,  .6900,  .920,  .900,     0,      0,      0,     0,     0,     0],
    [-.8,  .6624,  .874,  .880,     0,      0,      0,     0,     0,     0],
    [-.2,  .4100,  .160,  .210,  -.22,      0,   -.25,   108,     0,     0],
    [-.2,  .3100,  .110,  .220,   .22,      0,   -.25,    72,     0,     0],
    [ .2,  .2100,  .250,  .500,     0,    .35,   -.25,     0,     0,     0],
    [ .2,  .0460,  .046,  .046,     0,     .1,   -.25,     0,     0,     0],
    [ .1,  .0460,  .023,  .020,  -.08,   -.65,   -.25,     0,     0,     0],
    [ .1,  .0460,  .023,  .020,   .06,   -.65,   -.25,    90,     0,     0],
    [ .2,  .0560,  .040,  .100,   .06,  -.105,   .625,    90,     0,     0],
    [-.2,  .0560,  .056,  .100,     0,   .100,   .625,     0,     0,     0]]

# to convert to list of dicts
def _array_to_parameters(array):
    array = np.asarray(array)
    out = []
    for i in xrange(array.shape[0]):
        tmp = dict()
        for k, j in zip(parameters_tuple, xrange(array.shape[1])):
            tmp[k] = array[i, j]
        out.append(tmp)
    return out

modified_shepp_logan_parameters = _array_to_parameters(modified_shepp_logan_array)
shepp_logan_parameters = _array_to_parameters(shepp_logan_array)
yu_ye_wang_parameters = _array_to_parameters(yu_ye_wang_array)

# define specific functions
def modified_shepp_logan(shape, **kargs):
    return phantom(shape, modified_shepp_logan_parameters, **kargs)

def shepp_logan(shape, **kargs):
    return phantom(shape, shepp_logan_parameters, **kargs)

def yu_ye_wang(shape, **kargs):
    return phantom(shape, yu_ye_wang_parameters, **kargs)

# add docstrings to specific phantoms
common_docstring = """
    Generates a %(name)s phantom with a given shape and
    dtype.

    Inputs
    ------
    shape: 3-tuple of ints
       Shape of the 3d output cube.
    dtype: data-type
       Data type of the output cube.

    Output
    ------
    cube: ndarray
       3-dimensional phantom.

"""

modified_shepp_logan_docstring = common_docstring % {'name': 'Modified Shepp-Logan'}
shepp_logan_docstring = common_docstring % {'name': 'Shepp-Logan'}
yu_ye_wang_docstring = common_docstring % {'name': 'Yu Ye Wang'}

#np.add_docstring(modified_shepp_logan, modified_shepp_logan_docstring)
#np.add_docstring(shepp_logan, shepp_logan_docstring)
#np.add_docstring(yu_ye_wang, yu_ye_wang_docstring)
