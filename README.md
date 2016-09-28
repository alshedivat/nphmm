# Hidden Markov Models with Nonparametric Emissions

We provide two learning algorithms for HMMs with nonparametric emissions: a naive nonparameteric EM and the spectral learning algorithm based on decompositions of continuous matrices (as described in [the original paper](https://arxiv.org/abs/1609.06390)).
Continuous linear algebra is implemented using [chebfun](https://github.com/chebfun/chebfun).
KDE estimators are based on [if-estimators](https://github.com/kirthevasank/if-estimators).

### Download

To get the code, clone the repository as follows:
```bash
$ git clone --recursive git@github.com:alshedivat/nphmm.git
```
**Note:** You need to clone *recursively* since `nphmm` depends on `chebfun` as a submodule.

Alternatively, the code can be donwloaded directly as a [zip-file](https://github.com/alshedivat/nphmm/releases/download/v0.1/latest.zip).

### Installation

To use the library, add `lib` to the MATLAB path and setup `chebfun` as follows:
```matlab
>> addpath(genpath('lib'))
>> chebfun_setup()
```
To ensure that the library is functional, you can run tests as follows:
```matlab
>> runtests('tests')
```

### Usage

The library provides two main classes, `npHMM` and `npObsHMM`, that correspond the nonparameteric HMM and its observable representation, respectively.
Additionally, `learnNPHMM` trains a model from the provided data using either EM or spectral method.
For usecases and examples check the scripts in `examples/` and `tests/`.

### Citation

```bibtex
@article{kax2016nphmm,
  title={Learning HMMs with Nonparametric Emissions via Spectral Decompositions of Continuous Matrices},
  author={Kandasamy, Kirthevasan and Al-Shedivat, Maruan and Xing, Eric P},
  journal={arXiv preprint arXiv:1609.06390},
  year={2016}
}
```

### License

MIT (for details, please refer to [LICENSE](https://github.com/alshedivat/nphmm/blob/master/LICENSE))

Copyright (c) 2016 Maruan Al-Shedivat, Kirthevasan Kandasamy
