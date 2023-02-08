"""
A bagging-based drop-in replacement for the *CV_estimator* class.
"""

# Authors: Mirko Bunse <mirko.bunse@cs.tu-dortmund.de>
# License: /

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import issparse

class BaggingEstimator(BaseEstimator, ClassifierMixin):
    """A bagging-based drop-in replacement for the *CV_estimator* class.

    This drop-in replacement uses out-of-bag (OOB) predictions of a bagging ensemble for fitting the quantifier.

    Args:
        estimator: Any bagging classifier with an attribute *oob_score*.
    """
    def __init__(self, estimator):
        if not hasattr(estimator, "oob_score") or not estimator.oob_score:
            raise ValueError("Only bagging classifiers with oob_score=True are supported")
        self.estimator = estimator
        self.label_encoder_ = None
        self.X_trn_ = None
    def fit(self, X, y):
        self.label_encoder_ = LabelEncoder().fit(y)
        self.estimator.fit(X, self.label_encoder_.transform(y))
        self.X_trn_ = X
        return self
    def predict(self, X):
        if self.label_encoder_ is None:
            raise NotFittedError('BaggingEstimator not fitted')
        return self.label_encoder_.inverse_transform(np.argmax(self.predict_proba(X), axis=1))
    def predict_proba(self, X):
        if issparse(X):
            X_equals_X_trn_ = (X!=self.X_trn_).nnz==0
        else:
            X_equals_X_trn_ = np.array(X == self.X_trn_).all()
        if X_equals_X_trn_:
            return self.estimator.oob_decision_function_
        else:
            return self.estimator.predict_proba(X)
