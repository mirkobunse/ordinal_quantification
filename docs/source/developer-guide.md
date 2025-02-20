# Developer guide

## Unit testing

Run tests locally with the `unittest` package.

```bash
python -m venv venv
venv/bin/pip install -e .[tests]
venv/bin/python -m unittest
```

As soon as you push to the `main` branch, GitHub Actions will take out these unit tests, too.

## Documentation

Inspect your changes before pushing them to the `main` branch. After building the documentation, open `docs/build/index.html` in your browser.

```bash
. venv/bin/activate
pip install sphinx myst-parser sphinxcontrib-napoleon sphinx-rtd-theme
cd docs/
sphinx-apidoc --force --output-dir source/ ../ordinal_quantification
make html
```

As soon as you push to the `main` branch, GitHub Actions will build the documentation, push it to the `gh-pages` branch, and publish the result on GitHub Pages: [https://mirkobunse.github.io/ordinal_quantification](https://mirkobunse.github.io/ordinal_quantification)
