.. toctree::
   :maxdepth: 2

.. _scripts:

Scripts Reference
=================

Command-line scripts are available, both `Ptychography`_ and `CDI`_ scripts.

.. _ptycho_scripts:

Ptychography
------------
These are instructions to run the command-line scripts such as `pynx-cxipty.py`, `pynx-id01pty.py`, `pynx-id13pty.py`
and `pynx-id16apty.py`.

This help text can be simply obtained by typing the script without any parameter.

Generic instructions
++++++++++++++++++++

Here are generic instructions applying to all the scripts:

.. literalinclude:: ../../pynx/ptycho/runner/runner.py
   :start-after: helptext_generic
   :end-before: """
   :language: rst

Specific instructions
+++++++++++++++++++++

Instructions for beamline-dedicated scripts can be fully obtained by executing the script without any parameter.

`pynx-cxipty.py`:
^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../pynx/ptycho/runner/cxi.py
   :start-after: helptext_beamline
   :end-before: """
   :language: text

`pynx-id01pty.py`:
^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../pynx/ptycho/runner/id01.py
   :start-after: helptext_beamline
   :end-before: """
   :language: text

`pynx-id13pty.py`:
^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../pynx/ptycho/runner/id13.py
   :start-after: helptext_beamline
   :end-before: """
   :language: text

`pynx-id16apty.py`:
^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../pynx/ptycho/runner/id16a.py
   :start-after: helptext_beamline
   :end-before: """
   :language: text

.. _cdi_scripts:

CDI
---
These are instructions to run the command-line scripts: `pynx-id10cdi.py` and `pynx-id01cdi.py`. They
can also be used for generic data not from these beamlines.

This help text can be simply obtained by typing the script without any parameter.

Generic instructions
++++++++++++++++++++

Here are generic instructions applying to all the scripts:

.. literalinclude:: ../../pynx/cdi/runner/runner.py
   :start-after: helptext_generic
   :end-before: """
   :language: rst

Specific instructions
+++++++++++++++++++++

Instructions for beamline-dedicated scripts can be fully obtained by executing the script without any parameter.

`pynx-id10cdi.py`:
^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../pynx/cdi/runner/id10.py
   :start-after: helptext_beamline
   :end-before: """
   :language: text

`pynx-id01cdi.py`:
^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../pynx/cdi/runner/id01.py
   :start-after: helptext_beamline
   :end-before: """
   :language: text

CDI-regrid
----------
These are instructions to run the: `pynx-cdi-regrid.py` command-line script to prepare
a 3D cdi CXI file from multiple projections.

.. argparse::
   :module: pynx.cdi.runner.regrid
   :func: make_parser
   :prog: pynx-cdi-regrid.py
