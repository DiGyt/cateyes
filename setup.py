import os
from setuptools import setup

# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "cateyes",
    version = "0.0.4",

    author = "Dirk Gütlin",
    author_email = "dirk.guetlin@gmail.com",
    description = ("Uniform Categorization of Eyetracking in Python."),
    license = "BSD-3",
    keywords = "Eyetracking classification",
    url = "https://github.com/DiGyt/cateyes",
    packages=['cateyes'],
    include_package_data=True,
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "remodnav",
        #"nslr @ git+https://github.com/pupil-labs/nslr",
        #"nslr_hmm @ git+https://github.com/pupil-labs/nslr-hmm",
    ],
)
