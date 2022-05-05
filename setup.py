import setuptools

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="delphi.ai",
    version="0.2.1.9",
    author="Patroklos Stefanou",
    author_email="patstefanou@gmail.com",
    description="Package for Robust Statistics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pstefanou12/delphi",
    project_urls={
        "Repository": "https://github.com/pstefanou12/delphi",
    },
    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%4Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=['delphi',
              'delphi.utils',
              'delphi.distributions',
              'delphi.stats',
              'delphi.imagenet_models',
              'delphi.cifar_models',],
    python_requires=">=3.6",
    py_modules=['train', 'oracle', 'attacker', 'grad' 'attacker_steps'],
    setup_requires=['tqdm', 'grpcio', 'psutil', 'gitpython','py3nvml', 'cox',
                    'scikit-learn', 'seaborn', 'torch', 'torchvision', 'pandas',
                    'numpy', 'scipy', 'GPUtil', 'dill', 'tensorboardX', 'tables',
                    'matplotlib', 'orthnet'],
    install_requires=['tqdm', 'grpcio', 'psutil', 'gitpython','py3nvml', 'cox',
                    'scikit-learn', 'seaborn', 'torch', 'torchvision', 'pandas',
                    'numpy', 'scipy', 'GPUtil', 'dill', 'tensorboardX', 'tables',
                    'matplotlib', 'orthnet'],
)
