# API

Instantiate quantification methods through the [`ordinal_quantification.factory`](#ordinal_quantificationfactory) module.

## ordinal_quantification.factory

```{eval-rst}
.. automodule:: ordinal_quantification.factory
```

### Estimators

Each quantification method requires an estimator, usually a classifier. For this purpose, the `estimator` function of the `factory` module takes out the hyper-parameter optimization of the experiments in [https://github.com/bertocast/ordinal_quantification](https://github.com/bertocast/ordinal_quantification).

However, you can also go with any other classifier.

```{eval-rst}
.. autofunction:: ordinal_quantification.factory.estimator
```

### Quantifiers

```{eval-rst}
.. autofunction:: ordinal_quantification.factory.AC

.. autofunction:: ordinal_quantification.factory.CC

.. autofunction:: ordinal_quantification.factory.PAC

.. autofunction:: ordinal_quantification.factory.PCC
```
