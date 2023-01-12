# Development Setup

We are going to support `python>=3.7,<3.11`.

## Install requirements
Make sure you have nvcc, cuda-toolkit and cudnn installed. There are two ways to get them ready, either by conda or docker (recommended).

For conda users, you can install them with the following commands.
```console
$ conda install -c nvidia cuda-nvcc cuda-python
$ conda install cudnn
```
For docker users, you can use the [NVIDIA CUDA docker image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cuda) or the [PyTorch docker image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch), which has all of them ready and compatible with each other.

### Install JAX
Please follow the [official installation guide](https://github.com/google/jax#installation) to install JAX.


### Install JUX
Finally, upgrade your pip and install JUX.
```console
$ pip install --upgrade pip
$ pip install -e '.[dev,torch]'
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
