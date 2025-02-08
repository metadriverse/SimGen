from setuptools import setup, find_packages

setup(
    name="simgen",
    version="0.1.0",
    description="simgen",
    author="Sicheng Mo",
    author_email="smo3@cs.ucla.edu",
    packages=find_packages(),  # Automatically discover all packages and subpackages
    install_requires=[  # Dependencies
        "numpy",
        "torch",
        "xformers",
        "torchvision",
        "Pillow",
        "transformers",
        "einops",
        "opencv-python",
        "lightning",
        "omegaconf",
        "huggingface_hub",
        "open-clip-torch==2.0.2"
        # Add other dependencies here
    ],
    python_requires=">=3.10",  # Minimum Python version
    classifiers=[  # Metadata
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
