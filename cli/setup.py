from setuptools import setup, find_packages

setup(
    name="potpie-cli",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'potpie=potpie_cli:main',
        ],
    },
    install_requires=[
        "requests>=2.28.0",
    ],
    python_requires=">=3.11",
)
