Installation
============

From source (development)
-------------------------

.. code-block:: bash

   git clone https://github.com/m9h/dot-jax.git
   cd dot-jax
   pip install -e ".[test]"

Or with `uv <https://github.com/astral-sh/uv>`_:

.. code-block:: bash

   uv pip install -e ".[test]"

Dependencies
------------

**Required:**

- JAX >= 0.4.30
- Equinox >= 0.11.0
- Lineax >= 0.0.5
- jaxtyping >= 0.2.24
- NumPy >= 1.26

**Optional (for tests):**

- pytest >= 7.0
- scipy >= 1.10
- redbirdpy >= 0.2.0

**Optional (for meshing):**

- iso2mesh >= 0.5.0

Enabling 64-bit precision
-------------------------

JAX defaults to 32-bit. DOT computations require 64-bit precision.
Enable it at the start of your script:

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", True)
