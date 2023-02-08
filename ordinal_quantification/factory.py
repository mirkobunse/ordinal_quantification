"""This module creates instances of quantification methods for the outside world."""

from enum import Enum
from imblearn.metrics import geometric_mean_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from . import (classify_and_count, distribution_matching, estimators, ordinal)

Decomposer = Enum("Decomposer", ["monotone", "fh_tree", "dag", "dag_lv", "none"])
Option = Enum("Option", ["cv_decomp", "decomp_cv", "bagging_decomp"])

def _create_estimators(
        estimator,
        decomposer = Decomposer.monotone,
        option = Option.cv_decomp,
        n_folds = 20,
        random_state = None
        ):
    """Wrap two instances of the given estimator with a given order of CV and decomposer."""
    skf_trn = StratifiedKFold(
        n_splits = n_folds,
        shuffle = True,
        random_state = random_state
    )
    est_tst = _create_decomposer(estimator, decomposer)
    if option == Option.cv_decomp:
        est_trn = estimators.CV_estimator(estimator=est_tst, cv=skf_trn)
    elif option == Option.decomp_cv:
        print("WARNING: Option.decomp_cv is not used in bertocast/ordinal_quantification")
        est_trn = _create_decomposer(
            estimators.CV_estimator(estimator=estimator, cv=skf_trn),
            decomposer
        )
    elif option == Option.bagging_decomp:
        if decomposer is not None and decomposer != Decomposer.none:
            raise ValueError("Option.bagging_decomp is only supported with Decomposer.none")
        est_trn = estimators.BaggingEstimator(estimator)
        est_trn = est_tst
    else:
        raise ValueError('Unknown option {option}')
    return est_trn, est_tst

def _create_decomposer(estimator, decomposer=Decomposer.monotone):
    """Wrap the given estimator in a decomposer."""
    if decomposer == Decomposer.monotone:
        return estimators.frank_and_hall.FrankAndHallMonotoneClassifier(estimator)
    elif decomposer == Decomposer.fh_tree:
        return estimators.frank_and_hall.FrankAndHallTreeClassifier(estimator)
    elif decomposer == Decomposer.dag:
        return estimators.ordinal_ddag.DDAGClassifier(estimator)
    elif decomposer == Decomposer.dag_lv:
        return estimators.ordinal_ddag.DDAGClassifier(estimator, predict_method='winner_node')
    elif decomposer == Decomposer.none or decomposer is None:
        return estimator
    else:
        raise ValueError('Unknown decomposer {decomposer}')

def estimator(X, y, estimator=None, param_grid=None, random_state=None):
    """
    Take out the grid search of the original experiments.

    This grid search is looking for the hyper-parameters that optimize the geometric mean of class-wise accuracies in a stratified 3-fold cross validation.

    Args:
        X: The feature matrix for which the estimator will be optimized.
        y: The labels for which the estimator will be optimized.
        estimator (optional): The scikit-learn classifier to optimize. Defaults to a RandomForestClassifier.
        param_grid (optional): The parameter grid to optimize over. Defaults to the parameter grid that is used in the paper.
        random_state (optional): The numpy RandomState. Defaults to None.

    Returns:
        An estimator to be used in any of the quantification methods, optimized for (X, y).
    """
    if estimator is None:
        estimator = RandomForestClassifier(
            n_estimators = 100,
            class_weight = "balanced",
            random_state = random_state,
            oob_score = True, # does not hurt anybody; enables BaggingEstimator
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
    """
    Create an instance of the Classify-and-Count method.

    Args:
        estimator: The estimator, usually a classifier.
        verbose (optional): The logging level. Defaults to 0.
        decomposer (optional): How to decompose ordinal tasks into binary classification tasks. Defaults to Decomposer.monotone.
        option (optional): The order of cross validation and ordinal decomposition. Defaults to Option.cv_decomp.
        n_folds (optional): The number of folds for fitting the quantifier. Defaults to 20.
        random_state (optional): The numpy RandomState. Defaults to None.

    Returns:
        A configured instance of type CC.
    """
    return classify_and_count.CC(
        _create_estimators(estimator, **kwargs)[1],
        verbose = verbose
    )

def PCC(estimator, *, verbose=0, **kwargs):
    """
    Create an instance of the Probabilistic Classify-and-Count method.

    Args:
        estimator: The estimator, usually a classifier.
        verbose (optional): The logging level. Defaults to 0.
        decomposer (optional): How to decompose ordinal tasks into binary classification tasks. Defaults to Decomposer.monotone.
        option (optional): The order of cross validation and ordinal decomposition. Defaults to Option.cv_decomp.
        n_folds (optional): The number of folds for fitting the quantifier. Defaults to 20.
        random_state (optional): The numpy RandomState. Defaults to None.

    Returns:
        A configured instance of type PCC.
    """
    return classify_and_count.PCC(
        _create_estimators(estimator, **kwargs)[1],
        verbose = verbose
    )

def AC(estimator, *, distance="L2", verbose=0, **kwargs):
    """
    Create an instance of the Adjusted Classify-and-Count method.

    Args:
        estimator: The estimator, usually a classifier.
        distance (optional): The distance metric to optimize. Defaults to "L2".
        verbose (optional): The logging level. Defaults to 0.
        decomposer (optional): How to decompose ordinal tasks into binary classification tasks. Defaults to Decomposer.monotone.
        option (optional): The order of cross validation and ordinal decomposition. Defaults to Option.cv_decomp.
        n_folds (optional): The number of folds for fitting the quantifier. Defaults to 20.
        random_state (optional): The numpy RandomState. Defaults to None.

    Returns:
        A configured instance of type AC.
    """
    return classify_and_count.AC(
        *_create_estimators(estimator, **kwargs),
        distance = distance,
        verbose = verbose
    )

def PAC(estimator, *, distance="L2", verbose=0, **kwargs):
    """
    Create an instance of the Probabilistic Adjusted Classify-and-Count method.

    Args:
        estimator: The estimator, usually a classifier.
        distance (optional): The distance metric to optimize. Defaults to "L2".
        verbose (optional): The logging level. Defaults to 0.
        decomposer (optional): How to decompose ordinal tasks into binary classification tasks. Defaults to Decomposer.monotone.
        option (optional): The order of cross validation and ordinal decomposition. Defaults to Option.cv_decomp.
        n_folds (optional): The number of folds for fitting the quantifier. Defaults to 20.
        random_state (optional): The numpy RandomState. Defaults to None.

    Returns:
        A configured instance of type PAC.
    """
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

def CvMy(estimator, *, distances=euclidean_distances, verbose=0, **kwargs):
    """
    Create an instance of the CvMy method (Castaño et al., 2019).

    Args:
        estimator: The estimator, usually a classifier.
        distances (optional): The distance metric for each pair of samples. Defaults to sklearn.metrics.pairwise.euclidean_distances.
        verbose (optional): The logging level. Defaults to 0.
        decomposer (optional): How to decompose ordinal tasks into binary classification tasks. Defaults to Decomposer.monotone.
        option (optional): The order of cross validation and ordinal decomposition. Defaults to Option.cv_decomp.
        n_folds (optional): The number of folds for fitting the quantifier. Defaults to 20.
        random_state (optional): The numpy RandomState. Defaults to None.

    Returns:
        A configured instance of type CvMy.
    """
    return distribution_matching.CvMy(
        *_create_estimators(estimator, **kwargs),
        distance = distances,
        verbose = verbose
    )

def EDX(*, distances=euclidean_distances, verbose=0):
    """
    Create an instance of the energy distance method EDX (Kawakubo et al., 2016).

    Args:
        distances (optional): The distance metric for each pair of samples. Defaults to sklearn.metrics.pairwise.euclidean_distances.
        verbose (optional): The logging level. Defaults to 0.

    Returns:
        A configured instance of type EDX.
    """
    return distribution_matching.EDX(
        distance = distances,
        verbose = verbose
    )

def EDy(estimator, *, distances=euclidean_distances, verbose=0, **kwargs):
    """
    Create an instance of the energy distance method EDy (Castaño et al., 2016).

    Args:
        estimator: The estimator, usually a classifier.
        distances (optional): The distance metric for each pair of samples. Defaults to sklearn.metrics.pairwise.euclidean_distances. Another suitable value is ordinal_quantification.metrics.ordinal.emd_distances.
        verbose (optional): The logging level. Defaults to 0.
        decomposer (optional): How to decompose ordinal tasks into binary classification tasks. Defaults to Decomposer.monotone.
        option (optional): The order of cross validation and ordinal decomposition. Defaults to Option.cv_decomp.
        n_folds (optional): The number of folds for fitting the quantifier. Defaults to 20.
        random_state (optional): The numpy RandomState. Defaults to None.

    Returns:
        A configured instance of type EDy.
    """
    return distribution_matching.EDy(
        *_create_estimators(estimator, **kwargs),
        distance = distances,
        verbose = verbose
    )

def HDX(n_bins):
    """
    Create an instance of the hellinger distance method HDX (González-Castro et al., 2013).

    Args:
        n_bins: The number of bins per feature.

    Returns:
        A configured instance of type HDX.
    """
    return distribution_matching.HDX(n_bins)

def HDy(estimator, n_bins, *, verbose=0, **kwargs):
    """
    Create an instance of the hellinger distance method HDy (González-Castro et al., 2013).

    Args:
        estimator: The estimator, usually a classifier.
        n_bins: The number of bins per class.
        verbose (optional): The logging level. Defaults to 0.
        decomposer (optional): How to decompose ordinal tasks into binary classification tasks. Defaults to Decomposer.monotone.
        option (optional): The order of cross validation and ordinal decomposition. Defaults to Option.cv_decomp.
        n_folds (optional): The number of folds for fitting the quantifier. Defaults to 20.
        random_state (optional): The numpy RandomState. Defaults to None.

    Returns:
        A configured instance of type HDy.
    """
    return distribution_matching.HDy(
        *_create_estimators(estimator, **kwargs),
        n_bins = n_bins,
        verbose = verbose
    )

def OrdinalAC(estimator, *, verbose=0, **kwargs):
    """
    Create an instance of the ordinal version of AC (Castaño et al., 2022).

    Args:
        estimator: The estimator, usually a classifier.
        verbose (optional): The logging level. Defaults to 0.
        decomposer (optional): How to decompose ordinal tasks into binary classification tasks. Defaults to Decomposer.monotone.
        option (optional): The order of cross validation and ordinal decomposition. Defaults to Option.cv_decomp.
        n_folds (optional): The number of folds for fitting the quantifier. Defaults to 20.
        random_state (optional): The numpy RandomState. Defaults to None.

    Returns:
        A configured instance of type ACOrdinal.
    """
    return ordinal.ACOrdinal(
        *_create_estimators(estimator, **kwargs),
        verbose = verbose
    )

def PDF(estimator, n_bins, *, distance="EMD", verbose=0, **kwargs):
    """
    Create an instance of the ordinal method PDF (Castaño et al., 2022).

    Args:
        estimator: The estimator, usually a classifier.
        n_bins: The number of bins per class.
        distance (optional): The distance metric to optimize. Defaults to "EMD". Another suitable value is "L2".
        verbose (optional): The logging level. Defaults to 0.
        decomposer (optional): How to decompose ordinal tasks into binary classification tasks. Defaults to Decomposer.monotone.
        option (optional): The order of cross validation and ordinal decomposition. Defaults to Option.cv_decomp.
        n_folds (optional): The number of folds for fitting the quantifier. Defaults to 20.
        random_state (optional): The numpy RandomState. Defaults to None.

    Returns:
        A configured instance of type PDFOrdinaly.
    """
    return ordinal.PDFOrdinaly(
        *_create_estimators(estimator, **kwargs),
        n_bins = n_bins,
        distance = distance,
        verbose = verbose
    )
