"""Package metadata for the Rear-End ADAS Collision Warning System."""

from setuptools import find_packages, setup

setup(
    name="rear-end-adas",
    version="0.1.0",
    description=(
        "Camera-Based Rear-End ADAS Collision Warning System "
        "using YOLO detection and monocular depth estimation."
    ),
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "ultralytics>=8.1.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "PyYAML>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "rear-end-adas=src.main:main",
        ],
    },
)
