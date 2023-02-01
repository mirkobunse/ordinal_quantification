import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from unittest import TestCase

import ordinal_quantification
from ordinal_quantification import factory
from ordinal_quantification.metrics.ordinal import emd, emd_distances

RNG = np.random.RandomState(876) # make tests reproducible

def _read_data(name):
  """Read data (X, y) from datasets/ordinal/{name}.csv"""
  df = pd.read_csv(f"datasets/ordinal/{name}.csv", sep=';', header=0)
  X = df.iloc[:, :-1].values.astype(np.float64)
  y = df.iloc[:, -1].values.astype(np.int64)
  return X, y

def _classifier(seed):
  return RandomForestClassifier(5, random_state=seed)

class TestFactory(TestCase):
  def test_factory(self):
    seed = RNG.randint(np.iinfo(np.int32).max)
    X, y = _read_data("ESL")
    if len(y) > 1000:
      i = RNG.permutation(len(y))
      X = X[i[:1000],:]
      y = y[i[:1000]]
    p_true = np.unique(y, return_counts=True)[1] / len(y)
    print(f"p_true = {p_true}")

    tuned = factory.estimator(X, y, random_state=seed)
    print("Tuned the default estimator\n")

    methods = [
      ("CC", factory.CC(_classifier(seed), decomposer=factory.Decomposer.none, verbose=1)), # classify and count
      ("CC (tuned)", factory.CC(tuned, verbose=1)),
      ("AC_L2", factory.AC(_classifier(seed), decomposer=factory.Decomposer.none, verbose=1)),
      ("AC_L2 (tuned)", factory.AC(tuned, verbose=1)),
      ("AC_L2 (FH tree)", factory.AC(_classifier(seed), decomposer=factory.Decomposer.fh_tree, verbose=1)),
      ("AC_HD", factory.AC(_classifier(seed), distance="HD", verbose=1)),
      ("PAC_L2", factory.PAC(_classifier(seed), decomposer=factory.Decomposer.none, verbose=1)),
      ("CvMy_Eu", factory.CvMy(_classifier(seed), verbose=1)), # distribution matching
      ("EDX", factory.EDX(verbose=1)),
      ("EDy_Eu", factory.EDy(_classifier(seed), verbose=1)),
      ("EDy_EMD", factory.EDy(_classifier(seed), decomposer=factory.Decomposer.none, distances=emd_distances, verbose=1)),
      ("HDX", factory.HDX(2)),
      ("HDy", factory.HDy(_classifier(seed), 2, verbose=1)),
      ("AC_Ord", factory.OrdinalAC(_classifier(seed), verbose=1)), # ordinal methods
      ("PDF_EMD", factory.PDF(_classifier(seed), 2, verbose=1)),
      ("PDF_L2", factory.PDF(_classifier(seed), 2, distance="L2", verbose=1)),
    ]
    for (name, method) in methods:
      method.fit(X, y)
      p_est = method.predict(X)
      print(f"p_{name} = {p_est} (EMD = {emd(p_true, p_est)})\n")
