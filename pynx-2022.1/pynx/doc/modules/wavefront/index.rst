.. py:module:: pynx.wavefront

.. toctree::
   :maxdepth: 2

:mod:`pynx.wavefront`: Basic wavefront propagation (near and far field). Fresnel zone plate simulations
=======================================================================================================

.. include:: ../../../README.rst
   :start-after: pynx.wavefront section
   :end-before: pynx.wavefront end

API Reference
-------------
Note that the Python API uses an 'operator' approach, to enable writing
complex operations in a more mathematical and natural way.

Wavefront
^^^^^^^^^
.. automodule:: pynx.wavefront.wavefront
   :members:

Wavefront operators
^^^^^^^^^^^^^^^^^^^
This section lists the OpenCL operators, but identical operators are available for CUDA. The best set of
operators (determined by querying available languages and devices) is imported automatically when performing
`from pynx.wavefront import *` or `from pynx.cdi.wavefront import *`

.. automodule:: pynx.wavefront.cl_operator
   :members:

Fresnel propagation
^^^^^^^^^^^^^^^^^^^
Warning: this code is old and was mostly written as a proof-of-concept, but is no longer tested/developed.
It is recommended to use the Wavefront operators.

.. autofunction:: pynx.wavefront.fresnel.Fresnel_thread

Illumination from a Fresnel Zone Place
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Warning: this code is old and was mostly written as a proof-of-concept, but is no longer tested/developed.
It is recommended to use the Wavefront operators.

.. autofunction:: pynx.wavefront.fzp.FZP_thread

Examples
--------
Propagation operators
^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../pynx/wavefront/examples/propagation_operators.py

Paganin operator
^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../pynx/wavefront/examples/paganin.py

Illumination from a Fresnel Zone Place
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../../../pynx/wavefront/examples/fzp.py
