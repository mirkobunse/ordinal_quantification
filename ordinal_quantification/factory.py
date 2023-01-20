"""This module creates instances of quantification methods for the outside world."""

from enum import Enum
from imblearn.metrics import geometric_mean_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from . import (classify_and_count, estimators)

Decomposer = Enum("Decomposer", ["monotone", "fh_tree", "dag", "dag_lv"])
Option = Enum("Option", ["cv_decomp", "decomp_cv"])

def _create_estimators(
        estimator,
        decomposer = Decomposer.monotone,
        option = Option.cv_decomp,
        n_folds = 20,
        random_state = None
        ):
    """TODO"""
    skf_trn = StratifiedKFold(
        n_splits = n_folds,
        shuffle = True,
        random_state = random_state
    )
    est_tst = _create_decomposer(estimator, decomposer)
    if option == Option.cv_decomp:
        est_trn = estimators.cross_validation.CV_estimator(estimator=est_tst, cv=skf_trn)
    elif option == Option.decomp_cv:
        print("WARNING: Option.decomp_cv is not used in bertocast/ordinal_quantification")
        est_trn = _create_decomposer(
            estimators.cross_validation.CV_estimator(estimator=estimator, cv=skf_trn),
            decomposer
        )
    else:
        raise ValueError('Unknown option {option}')
    return est_trn, est_tst

def _create_decomposer(estimator, decomposer = Decomposer.monotone):
    """Wrap the given estimator in a decomposer."""
    if decomposer == Decomposer.monotone:
        return estimators.frank_and_hall.FrankAndHallMonotoneClassifier(estimator)
    elif decomposer == Decomposer.fh_tree:
        return estimators.frank_and_hall.FrankAndHallTreeClassifier(estimator)
    elif decomposer == Decomposer.dag:
        return estimators.ordinal_ddag.DDAGClassifier(estimator)
    elif decomposer == Decomposer.dag_lv:
        return estimators.ordinal_ddag.DDAGClassifier(estimator, predict_method='winner_node')
    else:
        raise ValueError('Unknown decomposer {decomposer}')

def estimator(X, y, estimator=None, param_grid=None, random_state=None):
    """Take out the grid search of the original experiments."""
    if estimator is None:
        estimator = RandomForestClassifier(
            n_estimators = 100,
            class_weight = "balanced",
            random_state = random_state,
        )
    if param_grid is None:
        param_grid = { # learning theory doesn't want us to tune n_estimators
            "max_depth": [1, 5, 10, 15, 20, 25, 30],
            "min_samples_leaf": [1, 2, 5, 10, 20],
        }
    gs_tst = GridSearchCV(
        estimator,
        param_grid = param_grid,
        scoring = make_scorer(geometric_mean_score),
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state),
        n_jobs = -1,
    )
    gs_tst.fit(X, y)
    return gs_tst.best_estimator_

def CC(estimator, *, verbose=0, **kwargs):
    return classify_and_count.CC(
        _create_estimators(estimator, **kwargs)[1],
        verbose = verbose
    )

def PCC(estimator, *, verbose=0, **kwargs):
    return classify_and_count.PCC(
        _create_estimators(estimator, **kwargs)[1],
        verbose = verbose
    )

def AC(estimator, *, distance="L2", verbose=0, **kwargs):
    return classify_and_count.AC(
        *_create_estimators(estimator, **kwargs),
        distance = distance,
        verbose = verbose
    )

def PAC(estimator, *, distance="L2", verbose=0, **kwargs):
    return classify_and_count.PAC(
        *_create_estimators(estimator, **kwargs),
        distance = distance,
        verbose = verbose
    )

def DeBias(estimator, *, verbose=0, **kwargs):
    print("WARNING: DeBias is not used in bertocast/ordinal_quantification")
    return classify_and_count.DeBias(
        *_create_estimators(estimator, **kwargs),
        verbose = verbose
    )
