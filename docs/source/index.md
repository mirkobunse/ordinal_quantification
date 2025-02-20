```{toctree}
:hidden:

self
api
developer-guide
```

# Quickstart

This fork of [https://github.com/bertocast/ordinal_quantification](https://github.com/bertocast/ordinal_quantification) allows you to install the upstream code via pip

```bash
pip install git+https://github.com/mirkobunse/ordinal_quantification.git
```

**Updating:** To update an existing installation of `ordinal_quantification`, run

```
pip install --force-reinstall --no-deps git+https://github.com/mirkobunse/ordinal_quantification.git
```

**Troubleshooting:** Starting from `pip 23.1.2`, you have to install `setuptools` and `wheel` explicitly. If you receive a "NameError: name 'setuptools' is not defined", you need to execute the following command before installing `ordinal_quantification`.

```
pip install --upgrade pip setuptools wheel
```


## Usage

You can use the methods as follows:

```python
from ordinal_quantification import factory

# X_trn, y_trn = your_training_data()
# X_tst, y_tst = your_testing_data()

# tune the hyper-parameters of the original classifier from the paper
classifier = factory.estimator(X_trn, y_trn)

# fit an instance of the adjusted classify & count method
method = factory.AC(classifier) # other classifiers could also be used
method.fit(X_trn, y_trn) # all quantifiers implement fit(X, y) and predict(X)

# estimate the prevalences of classes in X_tst
p_est = method.predict(X_tst)
```


## Citing

This repository implements the experiments of

```bibtex
@Article{castano2022matching,
  author  = {Alberto Casta{\~{n}}o and Pablo Gonz{\'{a}}lez and Jaime Alonso Gonz{\'{a}}lez and Juan Jos{\'{e}} del Coz},
  journal = {{IEEE} Transactions on Neural Networks and Learning Systems},
  title   = {Matching Distributions Algorithms Based on the Earth Mover's Distance for Ordinal Quantification},
  year    = {2022},
  doi     = {10.1109/tnnls.2022.3179355},
}
```
