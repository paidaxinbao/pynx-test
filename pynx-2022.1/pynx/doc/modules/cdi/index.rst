.. py:module:: pynx.cdi

.. toctree::
   :maxdepth: 2

:mod:`pynx.cdi`: Coherent Diffraction Imaging
=============================================

.. include:: ../../../README.rst
   :start-after: pynx.cdi section
   :end-before: pynx.cdi end

API Reference
-------------
Note that the Python API uses an 'operator' approach, to enable writing
complex operations in a more mathematical and natural way.

CDI base classes
^^^^^^^^^^^^^^^^
This is the CDI base classes, which can be used with operators

.. automodule:: pynx.cdi.cdi
   :members:

CDI Operators
^^^^^^^^^^^^^
This section lists the operators, which can be imported automatically using `from pynx.cdi import *`.
When this import is done, either CUDA (preferably) or OpenCL operators
will be imported. The documentation below corresponds to OpenCL operators, but this should be identical to CUDA
operators.

.. automodule:: pynx.cdi.cl_operator
   :members:

CDI Runner class
^^^^^^^^^^^^^^^^
The 'runner' class is used for command-line analysis scripts.

.. automodule:: pynx.cdi.runner.runner
   :members:

Regrid module reference
^^^^^^^^^^^^^^^^^^^^^^^
This module is used to transform far-field projections into a 3D interpolated dataset.

.. automodule:: pynx.cdi.runner.regrid
   :members: Regrid3D, parse_bliss_file

Examples
--------
.. literalinclude:: ../../../pynx/cdi/examples/simul_siemens.py

.. literalinclude:: ../../../pynx/cdi/examples/esrf_logo_id10.py

