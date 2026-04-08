from setuptools import setup

setup(
    name="rqcopt_mps",
    version="1.0.0",
    author="Isabel Nha Minh Le",
    author_email="isabel.le@tum.de",
    packages=["rqcopt_mps"],
    install_requires=[
        "scipy",
        "matplotlib",
        "PyYAML",
        "pytest",
        "h5py",
        "jax",
        "jaxlib",
        "orbax-checkpoint",
    ],
)
