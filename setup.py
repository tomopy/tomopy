from skbuild import setup

setup(
    # These keys must remain in setup.py because scikit-build intercepts them.
    # https://scikit-build.readthedocs.io/en/latest/usage.html#setuptools-options
    # For other package metadata see setup.cfg
    packages=['tomopy'],
    package_dir={"": "source"},
    include_package_data=True,
    zip_safe=False,
)
