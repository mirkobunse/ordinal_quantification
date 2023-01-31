# API

Instantiate quantification methods through the `ordinal_quantification.factory` module. Then, use their `fit` and `predict` methods.

## Creating estimators

Each quantification method requires an estimator, usually a classifier. For this purpose, the `estimator` function of the `factory` module takes out the hyper-parameter optimization of the experiments in [https://github.com/bertocast/ordinal_quantification](https://github.com/bertocast/ordinal_quantification).

However, you can also go with any other classifier.

```{eval-rst}
.. autofunction:: ordinal_quantification.factory.estimator
```

## Creating quantifiers

All quantifiers require a decomposition of the ordinal task into multiple binary classification tasks. For this purpose, you can specify the `decomposer` of the quantifier to be any value of the `Decomposer` enum:

```{eval-rst}
.. py:data:: ordinal_quantification.factory.Decomposer
    :type: Enum
    :value: ["monotone", "fh_tree", "dag", "dag_lv"]
```

The quantifiers are created through the following `factory` functions. We have categorized them into [](#classify-and-count) methods, [](#distribution-matching) methods, and [](#ordinal-methods)

### Classify and count

```{eval-rst}
.. autofunction:: ordinal_quantification.factory.AC

.. autofunction:: ordinal_quantification.factory.CC

.. autofunction:: ordinal_quantification.factory.PAC

.. autofunction:: ordinal_quantification.factory.PCC
```

### Distribution matching

```{eval-rst}
.. autofunction:: ordinal_quantification.factory.CvMy

.. autofunction:: ordinal_quantification.factory.EDX

.. autofunction:: ordinal_quantification.factory.EDy

.. autofunction:: ordinal_quantification.factory.HDX

.. autofunction:: ordinal_quantification.factory.HDy
```

### Ordinal methods

```{eval-rst}
.. autofunction:: ordinal_quantification.factory.OrdinalAC

.. autofunction:: ordinal_quantification.factory.PDF
```

## Using quantifiers

All instantiated quantifiers share the following methods:

```{eval-rst}
.. py:method:: fit(X, y)

   Fit the quantifier to data.

   :param X: The feature matrix to which the quantifier will be fitted.
   :param y: The labels to which the quantifier will be fitted.
   :return: The fitted quantifier itself.

.. py:method:: predict(X)

   Predict the class prevalences in a data set.

   :param X: The feature matrix for which the quantifier will predict.
   :return: A numpy array of class prevalences.
```
