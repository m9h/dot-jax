# Contributing to dot-jax

Thank you for your interest in contributing to dot-jax! This document explains
how to set up a development environment, run tests, and submit changes.

## 1. Getting Started

Clone the repository and install in development mode using [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/m9h/dot-jax.git
cd dot-jax
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev,docs]"
```

Enable 64-bit precision (required for DOT computations):

```python
import jax
jax.config.update("jax_enable_x64", True)
```

## 2. Development Workflow

1. Create a feature branch from `main`.
2. Make your changes with tests.
3. Run the test suite (see below).
4. Submit a pull request with a clear description of the change.

We follow a linear history; please rebase on `main` before submitting.

## 3. Code Style

- **Docstrings**: NumPy-style (parsed by `numpydoc`). Every public function
  and class must have a docstring with Parameters, Returns, and (where
  applicable) Notes and Examples sections.
- **Type annotations**: Use `jaxtyping` annotations for array-valued
  arguments (`Float[Array, "n_nodes 3"]`, etc.).
- **Formatting**: We do not enforce a specific formatter, but please keep
  lines under 100 characters and use 4-space indentation.
- **Naming**: Follow JAX/Equinox conventions. Modules are lowercase
  (`forward.py`), classes are CamelCase (`FEMMesh`), functions are
  snake_case (`forward_cw`).

## 4. Testing

Run the full test suite:

```bash
python -m pytest tests/ -v
```

Run a specific module's tests:

```bash
python -m pytest tests/test_forward.py -v
```

Tests are organised by layer:
- **Mathematical properties** (symmetry, positivity, conservation)
- **Known-value validation** against analytical formulas
- **Cross-validation** against redbirdpy and scipy
- **JIT/grad/vmap compatibility** for all core functions

When adding a new module, add corresponding tests that cover at least the
mathematical properties and JIT compatibility.

## 5. Documentation

Documentation is built with Sphinx and hosted on Read the Docs.

Build locally:

```bash
cd docs
make html
```

The output appears in `docs/_build/html/`.

- API docs are generated from docstrings via `autodoc`. Add a new module's
  RST stub in `docs/api/` and include it in `docs/index.rst`.
- Tutorials are `literalinclude` wrappers around the scripts in `examples/`.
  If you add a new example, create a matching RST file in `docs/tutorials/`
  and add it to the toctree.

## 6. AI-Assisted Development

This project was developed with AI assistance (Claude). When using AI tools:

- **Verify all mathematical formulations** against the cited references.
- **Run the cross-validation tests** after any change to core numerics.
- **Do not commit AI-generated code without review** and testing.
- Credit AI contributions with `Co-Authored-By` in the commit message.

## 7. Scientific Standards

dot-jax is a scientific computing library. Contributions that touch the
physics or numerics must meet these standards:

- **Cite your sources**: If you implement a formula, cite the paper (see
  the references in `docs/research.rst` and `README.md`).
- **FEM validation**: Any change to the assembly or forward modules must
  pass cross-validation against redbirdpy at the tolerance specified in
  `tests/conftest.py`.
- **DOT/fNIRS conventions**: Follow the conventions of the biomedical optics
  community. Absorption is `mua` (cm^-1), reduced scattering is `musp`
  (cm^-1), fluence is `Phi` (W/cm^2).
- **Reproducibility**: Examples should be self-contained and produce
  deterministic output (use fixed seeds where applicable).
- **Key references**:
  - Arridge (1999), "Optical tomography in medical imaging"
  - Dehghani et al. (2009), NIRFAST
  - Fang (2010), MCX
  - Farrell, Patterson & Wilson (1992), diffuse reflectance
  - Haskell et al. (1994), boundary conditions
