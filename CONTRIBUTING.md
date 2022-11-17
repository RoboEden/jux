# Development Setup

We are going to support `python>=3.7,<3.11`.

## Install requirements
Upgrade pip and install requirements JUX in editable mode with optional-dependencies for `dev`.
```console
$ pip install --upgrade pip
$ pip install -e .[dev]
```

## Setup pre-commit hooks
Install pre-commit hooks.
```console
$ pre-commit install
$ pre-commit run --all-files
```
It may take a while for the first run.

# Before commit
Make sure your code passes all tests.
```console
$ pytest
```

Check your code with pytype and make it happy (Optional).
```console
$ pytype jux -k -j 4
```
If you are sure that pytype is complaining about a false positive, you can add a `# pytype: disable=...` comment to the offending line.

If you successfully go through all above check, congratulations! You can commit your code now.
```console
$ git commit -m "your commit message"
```

# Before merge into master
Test your code against different python versions.
```console
$ tox
```
If it fails in a specific python version (e.g. `py38`), you can run `tox -e py38` to solely test against that version. Only after tests on all python versions pass, you can merge your code into master.
