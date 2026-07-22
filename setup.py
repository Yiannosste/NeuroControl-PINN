from setuptools import setup, find_packages

setup(
    name="neurocontrol-pinn",
    version="1.0.0",
    description="Physics-Informed Neural Network Surrogate for Nonlinear MPC",
    author="Yiannosste",
    license="MIT",
    packages=find_packages(exclude=["tests*", "notebooks*", "experiments*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0", "pytest-cov>=4.0.0"],
        "notebook": ["jupyter>=1.0.0", "jupyterlab>=3.6.0", "seaborn>=0.12.0"],
    },
)
