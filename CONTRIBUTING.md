# Development Setup

We are going to support `python>=3.8,<3.12`.

### Install JAX
JAX is a main dependency of JUX, and must be installed by user manually.
```sh
pip install --upgrade "jax[cuda11_cudnn82]==0.4.7" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

You can test whether jax is installed successfully by running the following command.
```sh
python -c "import jax.numpy as jnp; \
    a = jnp.convolve(jnp.array([1, 2, 3]), jnp.array([0, 1, 0.5])); \
    print(a); \
    print(a.device());"
# You shall get something like this:
# [0.  1.  2.5 4.  1.5]
# gpu:0
```

### Install JUX
Finally, upgrade your pip and install JUX.
```sh
pip install --upgrade pip
pip install -e '.[dev,torch]'
```

## Setup pre-commit hooks
Install pre-commit hooks.
```sh
pre-commit install
pre-commit run --all-files
```
It may take a while for the first run.

# Before commit
Make sure your code passes all tests.
```sh
pytest
```

Check your code with pytype and make it happy (Optional).
```sh
pytype jux -k -j 4
```
If you are sure that pytype is complaining about a false positive, you can add a `# pytype: disable=...` comment to the offending line.

If you successfully go through all above check, congratulations! You can commit your code now.
```sh
git commit -m "your commit message"
```

# Before merge into master
Test your code against different python versions.
```sh
tox
```
If it fails in a specific python version (e.g. `py38`), you can run `tox -e py38` to solely test against that version. Only after tests on all python versions pass, you can merge your code into master.
