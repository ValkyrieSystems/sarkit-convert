<div align="center">

<img src="https://raw.githubusercontent.com/ValkyrieSystems/sarkit/main/docs/source/_static/sarkit_logo.png" width=200>

[![PyPI - Version](https://img.shields.io/pypi/v/sarkit-convert)](https://pypi.org/project/sarkit-convert/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sarkit-convert)
[![PyPI - License](https://img.shields.io/pypi/l/sarkit-convert)](./LICENSE)
[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)
<br>
[![Tests](https://github.com/ValkyrieSystems/sarkit-convert/actions/workflows/test.yml/badge.svg)](https://github.com/ValkyrieSystems/sarkit-convert/actions/workflows/test.yml)

</div>

**sarkit-convert** is a Python library for converting SAR data to standard formats.

## Install
While some `sarkit-convert` functionality is available using base dependencies, many converter-specific dependencies are
declared in the packaging extras defined in the [`pyproject.toml`](./pyproject.toml).
`sarkit-convert` can be installed with one or more of these dependencies using pip:

```sh
$ python -m pip install sarkit-convert[cosmo,iceye,sentinel,terrasar]
$ python -m pip install sarkit-convert[all]
```

`sarkit-convert` can also be installed using conda and the conda-forge channel:

```sh
$ conda install --channel conda-forge sarkit-convert
```

The conda-forge package comes with all packaging extras.

## License
This repository is licensed under the [MIT license](./LICENSE).

## Contributing and Development
Contributions are welcome.
A few tips for getting started using [PDM](https://pdm-project.org/en/latest/) are below:


```shell
$ pdm install -G:all  # install SARkit-convert with optional & dev dependencies
$ pdm run nox  # run lint and tests
```
