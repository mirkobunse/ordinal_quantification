"""
Util functions for different kinds of quantifiers
"""

# Authors: Alberto Castaño <bertocast@gmail.com>
#          Jaime Alonso <jalonso@uniovi.es>
#          Juan José del Coz <juanjo@uniovi.es>
# License: BSD 3 clause

import numpy as np
import numbers
import cvxpy
import quadprog

from sklearn.utils import check_X_y


############
#  Functions for solving optimization problems with different loss functions
############
def solve_l1(train_distrib, test_distrib, n_classes, solver='ECOS'):
    """ Solves AC, PAC, PDF and Friedman optimization problems for L1 loss function

        min   |train_distrib * prevalences - test_distrib|
        s.t.  prevalences_i >=0
              sum prevalences_i = 1

        Parameters
        ----------
        train_distrib : array, shape depends on the optimization problem
            Represents the distribution of each class in the training set
            PDF: shape (n_bins * n_classes, n_classes)
            AC, PAC, Friedman: shape (n_classes, n_classes)

        test_distrib : array, shape depends on the optimization problem
            Represents the distribution of the testing set
            PDF: shape shape (n_bins * n_classes, 1)
            AC, PAC, Friedman: shape (n_classes, 1)

        n_classes : int
            Number of classes

        solver : str, (default='ECOS')
            The solver used to solve the optimization problem. The following solvers have been tested:
            'ECOS', 'ECOS_BB', 'CVXOPT', 'GLPK', 'GLPK_MI', 'SCS' and 'OSQP', but it seems that 'CVXOPT' does not
            work

        Returns
        -------
        prevalences : array, shape=(n_classes, )
           Vector containing the predicted prevalence for each class
    """
    prevalences = cvxpy.Variable(n_classes)
    objective = cvxpy.Minimize(cvxpy.norm(np.squeeze(test_distrib) - train_distrib @ prevalences, 1))

    contraints = [cvxpy.sum(prevalences) == 1, prevalences >= 0]

    prob = cvxpy.Problem(objective, contraints)
    prob.solve(solver=solver)
    return np.array(prevalences[0:n_classes].value).squeeze()


def solve_l2cvx(train_distrib, test_distrib, n_classes, solver='ECOS'):
    prevalences = cvxpy.Variable(n_classes)
    objective = cvxpy.Minimize(cvxpy.sum_squares(train_distrib @ prevalences - test_distrib))

    contraints = [cvxpy.sum(prevalences) == 1, prevalences >= 0]

    prob = cvxpy.Problem(objective, contraints)
    prob.solve(solver=solver)
    return np.array(prevalences[0:n_classes].value).squeeze()


def solve_l2(train_distrib, test_distrib, G, C, b):
    """ Solves AC, PAC, PDF and Friedman optimization problems for L2 loss function

        min    (test_distrib - train_distrib * prevalences).T (test_distrib - train_distrib * prevalences)
        s.t.   prevalences_i >=0
               sum prevalences_i = 1

        Expanding the objective function, we obtain:

        prevalences.T train_distrib.T train_distrib prevalences
        - 2 prevalences train_distrib.T test_distrib + test_distrib.T test_distrib

        Notice that the last term is constant w.r.t prevalences.

        Let G = 2 train_distrib.T train_distrib  and a = 2 * train_distrib.T test_distrib, we can use directly
        quadprog.solve_qp because it solves the following kind of problems:

        Minimize     1/2 x^T G x - a^T x
        Subject to   C.T x >= b

        `solve_l2` just computes the term a, shape (n_classes,1), and then calls quadprog.solve_qp.
        G, C and b were computed by `compute_l2_param_train` before, in the 'fit' method` of the PDF/Friedman object

        Parameters
        ----------
        train_distrib : array, shape depends on the optimization problem
            Represents the distribution of each class in the training set
            PDF: shape (n_bins * n_classes, n_classes)
            AC, PAC Friedman: shape (n_classes, n_classes)

        test_distrib : array, shape depends on the optimization problem
            Represents the distribution of the testing set
            PDF: shape shape (n_bins * n_classes, 1)
            AC, PAC, Friedman: shape (n_classes, 1)

        G : array, shape (n_classes, n_classes)

        C : array, shape (n_classes, n_constraints)
            n_constraints will be n_classes + 1

        b : array, shape (n_constraints,)

        G, C and b are computed by `compute_l2_param_train` in the 'fit' method

        Returns
        -------
        prevalences : array, shape=(n_classes, )
           Vector containing the predicted prevalence for each class
    """
    a = 2 * train_distrib.T.dot(test_distrib)
    a = np.squeeze(a)
    prevalences = quadprog.solve_qp(G=G, a=a, C=C, b=b, meq=1)
    return prevalences[0]


def compute_l2_param_train(train_distrib, classes):
    """ Computes params related to the train distribution for solving PDF optimization problems using
        L2 loss function

        Parameters
        ----------
        train_distrib : array, shape (n_bins * n_classes, n_classes)
            Represents the distribution of each class in the training set

        classes : ndarray, shape (n_classes, )
            Class labels

        Returns
        -------
        G : array, shape (n_classes, n_classes)

        C : array, shape (n_classes, n_constraints)
            n_constraints will be n_classes + 1  (n_classes constraints to guarantee that prevalences_i>=0, and
            an additional constraints for ensuring that sum(prevalences)==1

        b : array, shape (n_constraints,)

        quadprog.solve_qp solves the following kind of problems:

        Minimize     1/2 x^T G x  a^T x
        Subject to   C.T x >= b

        Thus, the values of G, C and b must be the following

        G = train_distrib.T train_distrib
        C = [[ 1, 1, ...,  1],
             [ 1, 0, ...,  0],
             [ 0, 1, 0,.., 0],
             ...
             [ 0, 0, ..,0, 1]].T
        C shape (n_classes+1, n_classes)
        b = [1, 0, ..., 0]
        b shape (n_classes, )
    """
    G = 2 * train_distrib.T.dot(train_distrib)
    if not is_pd(G):
        G = nearest_pd(G)
    #  constraints, sum prevalences = 1, every prevalence >=0
    n_classes = len(classes)
    C = np.vstack([np.ones((1, n_classes)), np.eye(n_classes)]).T
    b = np.array([1] + [0] * n_classes, dtype=np.float64)
    return G, C, b


############
# Functions to check if a matrix is positive definite and to compute the nearest positive definite matrix
# if it is not
############
def nearest_pd(A):
    """ Find the nearest positive-definite matrix to input

        A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which credits [2].

        References
        ----------
        [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

        [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if is_pd(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    indendity_matrix = np.eye(A.shape[0])
    k = 1
    while not is_pd(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += indendity_matrix * (-mineig * k ** 2 + spacing)
        k += 1

    return A3


def dpofa(m):
    """ Factors a symmetric positive definite matrix

        This is a version of the dpofa function included in quadprog library. Here, it is mainly used to check
        whether a matrix is positive definite or not

        Parameters
        ----------
        m : symmetric matrix, typically the shape is (n_classes, n_classes)
            The matrix to be factored. Only the diagonal and upper triangle are used

        Returns
        -------
        k : int,
            == 0  m is positive definite and the factorization has been completed
            >  0  the leading minor of order k is not positive definite

        r : array, an upper triangular matrix
            When k==0, the factorization is complete and r.T.dot(r) == m
            The strict lower triangle is unaltered (it is equal to the strict lower triangle of matrix m), so it
            could be different from 0.
   """
    r = np.array(m, copy=True)
    n = len(r)
    for k in range(n):
        s = 0.0
        if k >= 1:
            for i in range(k):
                t = r[i, k]
                if i > 0:
                    t = t - np.sum(r[0:i, i] * r[0:i, k])
                t = t / r[i, i]
                r[i, k] = t
                s = s + t * t
        s = r[k, k] - s
        if s <= 0.0:
            return k+1, r
        r[k, k] = np.sqrt(s)
    return 0, r


def is_pd(m):
    """ Checks whether a matrix is positive definite or not

        It is based on dpofa function, a version of the dpofa function included in quadprog library. When dpofa
        returns 0 the matrix is positive definite.

        Parameters
        ----------
        m : symmetric matrix, typically the shape is (n_classes, n_classes)
            The matrix to check whether it is positive definite or not

        Returns
        -------
        A boolean, True when m is positive definite and False otherwise

    """
    return dpofa(m)[0] == 0


############
# Functions for solving HD-based methods
############
def solve_hd(train_distrib, test_distrib, n_classes, solver='ECOS'):
    """ Solves the optimization problem for PDF methods using Hellinger Distance

        This method just uses cvxpy library

        Parameters
        ----------
        train_distrib : array, shape (n_bins * n_classes, n_classes)
            Represents the distribution of each class in the training set

        test_distrib : array, shape (n_bins * n_classes, 1)
            Represents the distribution of the testing set

        n_classes : int
            Number of classes

        solver : str, optional (default='ECOS')
            The solver to use. For example, 'ECOS', 'SCS', or 'OSQP'.

        Returns
        -------
        prevalences : array, shape=(n_classes, )
            Vector containing the predicted prevalence for each class
    """
    prevalences = cvxpy.Variable(n_classes)
    s = cvxpy.multiply(np.squeeze(test_distrib), train_distrib @ prevalences)
    objective = cvxpy.Minimize(1 - cvxpy.sum(cvxpy.sqrt(s)))
    contraints = [cvxpy.sum(prevalences) == 1, prevalences >= 0]

    prob = cvxpy.Problem(objective, contraints)
    prob.solve(solver=solver)
    return np.array(prevalences.value).squeeze()


def probs2crisps(preds, labels):
    """ Convert probability predictions to crisp predictions

        Parameters
        ----------
        preds : ndarray, shape (n_examples, 1) or (n_examples,) for binary problems, (n_examples, n_classes) multiclass
            The matrix with the probability predictions

        labels : ndarray, shape (n_classes, )
            Class labels
    """
    if len(preds) == 0:
        return preds
    if preds.ndim == 1 or preds.shape[1] == 1:
        #  binary problem
        if preds.ndim == 1:
            preds_mod = np.copy(preds)
        else:
            preds_mod = np.copy(preds.squeeze())
        if isinstance(preds_mod[0], np.float64):
            # it contains probs
            preds_mod[preds_mod >= 0.5] = 1
            preds_mod[preds_mod < 0.5] = 0
            return preds_mod.astype(int)
        else:
            return preds_mod
    else:
        # multiclass problem
        if isinstance(preds[0, 0], np.float64):
            # are probs
            #  preds_mod = np.copy(preds)
            return labels.take(preds.argmax(axis=1), axis=0)
        else:
            raise TypeError("probs2crips: error converting probabilities, the type of the values is int")


def create_bags_with_multiple_prevalence(X, y, n=1001, rng=None):
    """ Create bags of examples given a dataset with different prevalences

        The method proposed by Kramer is used to generate a uniform distribution of the prevalences

        Parameters
        ----------
        X : array-like, shape (n_examples, n_features)
            Data

        y : array-like, shape (n_examples, )
            True classes

        n : int, default (n=1001)
            Number of bags

        rng : int, RandomState instance, (default=None)
            To generate random numbers
            If type(rng) is int, rng is the seed used by the random number generator;
            If rng is a RandomState instance, rng is the own random number generator;

        Raises
        ------
        ValueError
            When rng is neither a int nor a RandomState object

        References
        ----------
        http://www.cs.cmu.edu/~nasmith/papers/smith+tromble.tr04.pdf

        http://blog.geomblog.org/2005/10/sampling-from-simplex.html
    """
    if isinstance(rng, (numbers.Integral, np.integer)):
        rng = np.random.RandomState(rng)
    if not isinstance(rng, np.random.RandomState):
        raise ValueError("Invalid random generaror object")

    X, y = check_X_y(X, y)
    classes = np.unique(y)
    n_classes = len(classes)
    m = len(X)

    for i in range(n):
        # Kraemer method:

        # to soft limits
        low = round(m * 0.05)
        high = round(m * 0.95)

        ps = rng.randint(low, high, n_classes - 1)
        ps = np.append(ps, [0, m])
        ps = np.diff(np.sort(ps))  # number of samples for each class
        prev = ps / m  # to obtain prevalences
        idxs = []
        for n, p in zip(classes, ps.tolist()):
            if p != 0:
                idx = rng.choice(np.where(y == n)[0], p, replace=True)
                idxs.append(idx)

        idxs = np.concatenate(idxs)
        yield X[idxs], y[idxs], prev
