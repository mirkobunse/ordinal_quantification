import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from unittest import TestCase

import ordinal_quantification
from ordinal_quantification.classify_and_count import (AC, CC)
from ordinal_quantification.metrics.ordinal import emd

RNG = np.random.RandomState(876) # make tests reproducible

def _read_data(name):
  """Read data (X, y) from datasets/ordinal/{name}.csv"""
  df = pd.read_csv(f"datasets/ordinal/{name}.csv", sep=';', header=0)
  X = df.iloc[:, :-1].values.astype(np.float64)
  y = df.iloc[:, -1].values.astype(np.int64)
  return X, y

def _two_clones(classifier):
  """Clone the classifier two times, for UsingClassifiers instances"""
  classifier = sklearn.base.clone(classifier)
  return classifier, classifier

class TestAC(TestCase):
  def testAC(self):
    seed = RNG.randint(np.iinfo(np.int32).max)
    clf = RandomForestClassifier(5, random_state=seed)
    X, y = _read_data("ESL")
    if len(y) > 1000:
      i = RNG.permutation(len(y))
      X = X[i[:1000],:]
      y = y[i[:1000]]
    p_true = np.unique(y, return_counts=True)[1] / len(y)
    print(f"p_true = {p_true}")
    methods = [
      ("CC", CC(sklearn.base.clone(clf), verbose=1)),
      ("AC", AC(*_two_clones(clf), verbose=1)),
      ("AC_HD", AC(*_two_clones(clf), distance="HD", verbose=1)),
    ]
    for (name, method) in methods:
      method.fit(X, y)
      p_est = method.predict(X)
      print(f"p_{name} = {p_est} (EMD = {emd(p_true, p_est)})")
