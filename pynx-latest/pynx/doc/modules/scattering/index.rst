.. py:module:: pynx.scattering

.. toctree::
   :maxdepth: 2

:mod:`pynx.scattering`: scattering calculations from vectors of xyz atomic positions and hkl values
===================================================================================================

.. include:: ../../../README.rst
   :start-after: pynx.scattering section
   :end-before: pynx.scattering end

API Reference
-------------
This is the reference for scattering calculations. Note that this part of the code is older than the
cdi, ptycho and wavefront modules, so that the API is much more basic.

Structure factor calculations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:module:: pynx.scattering.fhkl

.. autofunction:: pynx.scattering.fhkl.Fhkl_thread

.. autofunction:: pynx.scattering.fhkl.Fhkl_gold

Grazing incidence diffraction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This module requires the cctbx library.

.. automodule:: pynx.scattering.gid
   :members:

Thomson scattering factor
^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:module:: pynx.scattering.fthomson

.. autofunction:: pynx.scattering.fthomson.f_thomson


