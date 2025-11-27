from setuptools import setup, find_packages

setup(
    name="ml_project",
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "scikit-learn",
        "pyyaml",
        "fastapi",
    ],
)
