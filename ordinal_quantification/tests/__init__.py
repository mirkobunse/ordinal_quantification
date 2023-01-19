import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from unittest import TestCase

import ordinal_quantification
from ordinal_quantification import factory
from ordinal_quantification.metrics.ordinal import emd

RNG = np.random.RandomState(876) # make tests reproducible

def _read_data(name):
  """Read data (X, y) from datasets/ordinal/{name}.csv"""
  df = pd.read_csv(f"datasets/ordinal/{name}.csv", sep=';', header=0)
  X = df.iloc[:, :-1].values.astype(np.float64)
  y = df.iloc[:, -1].values.astype(np.int64)
  return X, y

def _classifier(seed):
  return RandomForestClassifier(5, random_state=seed)

class TestAC(TestCase):
  def testAC(self):
    seed = RNG.randint(np.iinfo(np.int32).max)
    X, y = _read_data("ESL")
    if len(y) > 1000:
      i = RNG.permutation(len(y))
      X = X[i[:1000],:]
      y = y[i[:1000]]
    p_true = np.unique(y, return_counts=True)[1] / len(y)
    print(f"p_true = {p_true}")
    methods = [
      ("CC", factory.create_CC(_classifier(seed), verbose=1)),
      ("AC_L2", factory.create_AC(_classifier(seed), verbose=1)),
      ("AC_L2 (FH tree)", factory.create_AC(_classifier(seed), decomposer=factory.Decomposer.fh_tree, verbose=1)),
      ("AC_HD", factory.create_AC(_classifier(seed), distance="HD", verbose=1)),
      ("PAC_L2", factory.create_PAC(_classifier(seed), verbose=1)),
    ]
    for (name, method) in methods:
      method.fit(X, y)
      p_est = method.predict(X)
      print(f"p_{name} = {p_est} (EMD = {emd(p_true, p_est)})")
