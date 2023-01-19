from setuptools import setup

def readme():
    with open("README.md") as f:
        return f.read()

setup(
    name="ordinal_quantification",
    version="0.0.1",
    description="", # TODO
    long_description=readme(),
    classifiers=[
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 3 - Alpha",
    ],
    keywords=[
        "machine learning",
        "supervised learning",
        "ordinal quantification",
        "supervised prevalence estimation",
    ],
    url="https://github.com/bertocast/ordinal_quantification",
    author="", # TODO
    author_email="", # TODO
    license="", # TODO
    packages=setuptools.find_packages(),
    install_requires=[
        "cvxpy",
        "Cython",
        "imbalanced-learn",
        "numpy",
        "quadprog",
        "scikit-learn",
        "scipy",
        "six",
    ],
    python_requires=">=3.6",
    include_package_data=True,
    zip_safe=False,
    test_suite="nose.collector",
    extras_require = {
        "experiments" : ["matplotlib", "pandas"],
        "tests" : ["nose", "pandas"]
    }
)
