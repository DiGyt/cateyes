import os
from setuptools import setup

# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

# define reqs
REQS = [
    "numpy",
    "scipy",
    "matplotlib",
    "remodnav",
    ]
NSLR_REQS = [
    #"nslr @ git+https://github.com/pupil-labs/nslr",
    "nslr @ git+https://gitlab.com/nslr/nslr.git",
    "nslr_hmm @ git+https://github.com/pupil-labs/nslr-hmm",
    #"nslr @ git+https://gitlab.com/nslr/nslr.git@b99f0b7b",
    #"nslr_hmm @ git+https://github.com/pupil-labs/nslr-hmm.git@main",
]
UNEYE_REQS = [
    "uneye @ git+https://github.com/DiGyt/uneye.git",
    #"uneye @ git+https://github.com/DiGyt/uneye.git@492c0268c8c0f3a6271d4b8f5832e24f1bc62848",
]

setup(
    name = "cateyes",
    version = "0.0.5",

    author = "Dirk Gütlin",
    author_email = "dirk.guetlin@gmail.com",
    description = ("Uniform Categorization of Eyetracking in Python."),
    license = "BSD-3",
    keywords = "Eyetracking classification",
    url = "https://github.com/DiGyt/cateyes",
    packages = ["cateyes"],
    include_package_data=True,
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
    install_requires=REQS,
    extras_require={
        "nslr_hmm":NSLR_REQS,
        "uneye":UNEYE_REQS,
        "full":NSLR_REQS + UNEYE_REQS,
    },
    #dependency_links= NSLR_REQS + UNEYE_REQS,
)
