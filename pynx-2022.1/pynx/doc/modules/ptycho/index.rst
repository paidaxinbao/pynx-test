.. py:module:: pynx.ptycho

.. toctree::
   :maxdepth: 2

:mod:`pynx.ptycho`: 2D Ptychography
===================================

.. include:: ../../../README.rst
   :start-after: pynx.ptycho section
   :end-before: pynx.ptycho end

API Reference
-------------
Note that the Python API is quickly evolving.

For regular data analysis, it is recommended to use the :doc:`scripts <../../scripts/index>` which are stable,
independently of the underlying Python API.

2D Ptychography (operator-based)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This is the Ptychography class, which can be used with operators

.. automodule:: pynx.ptycho.ptycho
   :members:

2D Ptychography Operators
^^^^^^^^^^^^^^^^^^^^^^^^^
This section lists the operators, which can be imported automatically using `from pynx.ptycho import *`
or `from pynx.ptycho.operator import *`. When this import is done, either CUDA (preferably) or OpenCL operators
will be imported. The documentation below corresponds to OpenCL operators, but this should be identical to CUDA
operators.

.. automodule:: pynx.ptycho.cl_operator
   :members:

Ptychography Runner class
^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pynx.ptycho.runner.runner
   :members:

Ptychography Analysis
^^^^^^^^^^^^^^^^^^^^^
.. automodule:: pynx.ptycho.analysis
   :members:


Examples
--------

Operator-based API, far field
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../pynx/ptycho/examples/ptycho_operators.py

Operator-based API, near field
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../pynx/ptycho/examples/ptycho_operators_near_field.py

