[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://mirkobunse.github.io/ordinal_quantification)
[![CI](https://github.com/mirkobunse/ordinal_quantification/actions/workflows/ci.yml/badge.svg)](https://github.com/mirkobunse/ordinal_quantification/actions/workflows/ci.yml)


# bertocast/ordinal_quantification

This fork of https://github.com/bertocast/ordinal_quantification allows you to install the upstream code via pip

```
pip install git+https://github.com/mirkobunse/ordinal_quantification.git
```


## Usage

For detailed information, visit [the documentation](https://mirkobunse.github.io/ordinal_quantification).

Basically, you can use the methods as follows:

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

```
@Article{castano2022matching,
  author  = {Alberto Casta{\~{n}}o and Pablo Gonz{\'{a}}lez and Jaime Alonso Gonz{\'{a}}lez and Juan Jos{\'{e}} del Coz},
  journal = {{IEEE} Transactions on Neural Networks and Learning Systems},
  title   = {Matching Distributions Algorithms Based on the Earth Mover's Distance for Ordinal Quantification},
  year    = {2022},
  doi     = {10.1109/tnnls.2022.3179355},
}
```
