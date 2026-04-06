Installation
============

From PyPI
---------

.. code-block:: bash

   pip install genevector

From Source
-----------

.. code-block:: bash

   git clone https://github.com/nceglia/genevector.git
   cd genevector
   pip install -e .

With Numba Acceleration
-----------------------

.. code-block:: bash

   pip install genevector[fast]

With Rust Backend
-----------------

Requires the `Rust toolchain <https://rustup.rs/>`_:

.. code-block:: bash

   pip install maturin
   maturin develop --release

Dependencies
------------

Required:

- Python >= 3.9
- PyTorch
- Scanpy, AnnData
- NumPy, SciPy, Pandas
- Matplotlib, Seaborn

Optional:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Package
     - Required for
     - Install
   * - ``numba``
     - Fast MI computation
     - ``pip install numba``
   * - ``squidpy``
     - Building spatial neighbor graphs
     - ``pip install squidpy``
   * - ``torch_geometric``
     - PyG-backed aggregations
     - See `PyG docs <https://pytorch-geometric.readthedocs.io>`_
