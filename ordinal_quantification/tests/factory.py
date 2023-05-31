import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

def _classifier(seed, option):
  if option == factory.Option.cv_decomp:
    n_trees = 5
    oob_score = False
  elif option == factory.Option.bagging_decomp:
    n_trees = 100
    oob_score = True
  return RandomForestClassifier(n_trees, random_state=seed, oob_score=oob_score)

class TestFactory(TestCase):
  def test_local_random_state(self):
    np.random.seed(123)
    v1 = np.random.rand()
    v2 = np.random.rand()
    np.random.seed(123)
    self.assertEqual(np.random.rand(), v1)
    with factory._local_random_state(np.random.RandomState()):
      self.assertNotEqual(np.random.rand(), v2) # the random state above is a different one
    self.assertEqual(np.random.rand(), v2) # check that the global RandomState is unaffected
  def test_cv(self):
    option = factory.Option.cv_decomp
    print(f"\noption = {option}")
    seed = 0 # RNG.randint(np.iinfo(np.int32).max)
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
      ("CC", factory.CC(_classifier(seed, option), decomposer=factory.Decomposer.none, verbose=1, option=option)), # classify and count
      ("CC (tuned)", factory.CC(tuned, verbose=1, option=option)),
      ("AC_L2", factory.AC(_classifier(seed, option), decomposer=factory.Decomposer.none, verbose=1, option=option)),
      ("AC_L2 (tuned)", factory.AC(tuned, verbose=1, option=option)),
      ("AC_L2 (FH tree)", factory.AC(_classifier(seed, option), decomposer=factory.Decomposer.fh_tree, verbose=1, option=option)),
      ("AC_HD", factory.AC(_classifier(seed, option), distance="HD", verbose=1, option=option)),
      ("PAC_L2", factory.PAC(_classifier(seed, option), decomposer=factory.Decomposer.none, verbose=1, option=option)),
      ("CvMy_Eu", factory.CvMy(_classifier(seed, option), verbose=1, option=option)), # distribution matching
      ("EDX", factory.EDX(verbose=1)),
      ("EDy_Eu", factory.EDy(_classifier(seed, option), verbose=1, option=option)),
      ("EDy_EMD", factory.EDy(_classifier(seed, option), decomposer=factory.Decomposer.none, distances=emd_distances, verbose=1, option=option)),
      ("HDX", factory.HDX(2)),
      ("HDy", factory.HDy(_classifier(seed, option), 2, verbose=1, option=option)),
      ("AC_Ord", factory.OrdinalAC(_classifier(seed, option), verbose=1, option=option)), # ordinal methods
      ("PDF_EMD", factory.PDF(_classifier(seed, option), 2, verbose=1, option=option)),
      ("PDF_EMD_L2", factory.PDF(_classifier(seed, option), 2, distance="EMD_L2", verbose=1, option=option)),
      ("PDF_L2", factory.PDF(_classifier(seed, option), 2, distance="L2", verbose=1, option=option)),
    ]
    for (name, method) in methods:
      method.fit(X, y)
      p_est = method.predict(X)
      print(f"p_{name} = {p_est} (EMD = {emd(p_true, p_est)})\n")

  def test_bagging(self): # the default estimator is a bagging classifier, already
    option = factory.Option.bagging_decomp
    print(f"\noption = {option}")
    seed = 0 # RNG.randint(np.iinfo(np.int32).max)
    X, y = _read_data("ESL")
    if len(y) > 1000:
      i = RNG.permutation(len(y))
      X = X[i[:1000],:]
      y = y[i[:1000]]
    p_true = np.unique(y, return_counts=True)[1] / len(y)
    print(f"p_true = {p_true}")

    methods = [ # all methods with Decomposer.none
      ("CC", factory.CC(_classifier(seed, option), decomposer=factory.Decomposer.none, verbose=1, option=option)), # classify and count
      ("CC [CV]", factory.CC(_classifier(seed, factory.Option.cv_decomp), decomposer=factory.Decomposer.none, option=factory.Option.cv_decomp)),
      ("AC_L2", factory.AC(_classifier(seed, option), decomposer=factory.Decomposer.none, verbose=1, option=option)),
      ("AC_L2 [CV]", factory.AC(_classifier(seed, factory.Option.cv_decomp), decomposer=factory.Decomposer.none, option=factory.Option.cv_decomp)),
      ("PAC_L2", factory.PAC(_classifier(seed, option), decomposer=factory.Decomposer.none, verbose=1, option=option)),
      ("PAC_L2 [fallback]", factory.PAC(LogisticRegression(random_state=seed), decomposer=factory.Decomposer.none, verbose=1, option=option)),
      ("PAC_L2 [CV]", factory.PAC(_classifier(seed, factory.Option.cv_decomp), decomposer=factory.Decomposer.none, option=factory.Option.cv_decomp)),
      ("EDy_EMD", factory.EDy(_classifier(seed, option), decomposer=factory.Decomposer.none, distances=emd_distances, verbose=1, option=option)),
      ("EDy_EMD [CV]", factory.EDy(_classifier(seed, factory.Option.cv_decomp), decomposer=factory.Decomposer.none, distances=emd_distances, option=factory.Option.cv_decomp)),
    ]
    for (name, method) in methods:
      method.fit(X, y)
      p_est = method.predict(X)
      print(f"p_{name} = {p_est} (EMD = {emd(p_true, p_est)})\n")
