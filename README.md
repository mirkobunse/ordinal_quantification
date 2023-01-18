# bertocast/ordinal_quantification

This fork of https://github.com/bertocast/ordinal_quantification allows you to install the upstream code via pip

```
pip install https://github.com/mirkobunse/ordinal_quantification
```


## Usage

You can use the methods as follows:

```python
from ordinal_quantification.classify_and_count import (AC, CC)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier() # or some other classifier
# X_trn, y_trn = your_training_data()
# X_tst, y_tst = your_testing_data()

cc = CC(clf)
ac = AC(clf, clf) # AC distinguishes training and testing classifiers

for method in (cc, ac):
  method.fit(X_trn, y_trn) # all quantifiers implement fit(X, y) and predict(X)
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


## Development / unit testing

Run tests locally with the `unittest` package.

```
python -m venv venv
venv/bin/pip install .[tests]
venv/bin/python -m unittest
```
