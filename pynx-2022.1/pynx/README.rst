===================================================
PyNX: Python tools for Nano-structures Xtallography
===================================================

NEWS
====

* 2021/11/XX: :ref:`PyNX 2021.1<latest_pynx>` is available. See the full :doc:`Changelog<changelog>`
* 2020/12/12: PyNX 2020.2.2 is available with a few fixes.
* 2020/10/23: PyNX 2020.2 is available, notably introducing distributed computing on multiple GPU
* 2020/09: the **PyNX article**: *PyNX: high-performance computing toolkit for coherent
  X-ray imaging based on operators* is out:
  `J. Appl. Cryst. 53 (2020), 1404 <http://dx.doi.org/10.1107/S1600576720010985>`_,
  also available as `arXiv:2008.11511 <https://arxiv.org/abs/2008.11511>`_
* 2020/02/02: version 2020.1 is out !
* 2019/06/19: version 2019.2.6 is out. Note that 2019.2.x (x<6) versions had incorrect Ptycho CUDA scaling, preventing
  correct ML minimisation
* 2018/06: all coherent imaging modules have now been converted to use an operator-based API
* 2016/09: overhaul of the library structure - ``pynx.gpu`` has now become ``pynx.scattering``, with ``pynx.scattering.gid`` as a sub-module.

Introduction
============
PyNX stands for *Python tools for Nano-structures Xtallography*. It can be used for:

* Coherent X-ray imaging simulation and analysis: coherent diffraction imaging (CDI), Ptychography,
  Wavefront propagation, near field and far field techniques...

* Fast scattering calculations from large number of atoms and reciprocal space positions.

PyNX is fully optimised to use Graphical Processing Units, using either CUDA or OpenCL, to provide fast calculations
with 1 to 3 orders of magnitude speedup compared to standard processor calculations.

PyNX scripts
------------
PyNX can be used simply with :doc:`command-line scripts <scripts/index>` for some applications (2D/3D CDI and 2D
Ptychography). These can take generic files as input, such as CXI files (http://cxidb.org), or can analyse data
directly from beamlines.

PyNX as a python toolkit
------------------------
PyNX can be used as a python library with the following main modules:

1) :mod:`pynx.scattering`:  *X-ray scattering computing using graphical processing units*, allowing up to 2.5x10^11 reflections/atoms/seconds
   (single nVidia Titan X). The sub-module``pynx.scattering.gid`` can be used for *Grazing Incidence Diffraction* calculations, using
   the Distorted Wave Born Approximation

2) :mod:`pynx.ptycho` : simulation and analysis of experiments using the *ptychography* technique, using GPU (OpenCL).
   Examples are available in the pynx/Examples directory. Scripts for analysis of raw data from beamlines are also available, as well as using
   or producing ptychography data sets in CXI (Coherent X-ray Imaging) format.

3) :mod:`pynx.wavefront`: *X-ray wavefront propagation* in the near, far field, or continuous (examples available at the end of ``wavefront.py``).
   Also provided are sub-modules for Fresnel propagation and simulation of the illumination from a Fresnel Zone Plate, both using OpenCL for
   high performance computing.

4) :mod:`pynx.cdi`: *Coherent Diffraction Imaging* reconstruction algorithms using GPU for Coherent Diffraction Imaging,
   in 2D or 3D, for small-angle or Bragg diffraction data. This uses either CUDA or OpenCL, but CUDA is strongly
   recommended for 3D data (significant speedup).

Download
========
PyNX is available from:
 * http://ftp.esrf.fr/pub/scisoft/PyNX/
 * http://gitlab.esrf.fr/favre/PyNX (login required, site registration is open & free)

Installation
============

The simplest way to install PyNX (usually in an existing virtual or conda environment)
is to use (in this case with both cuda and OpenCL backends):

.. code-block:: bash

  curl -O http://ftp.esrf.fr/pub/scisoft/PyNX/pynx-latest.tar.bz2
  pip install pynx-latest.tar.bz2[cuda]

For more installation options, see the :doc:`detailed installation instructions
(preferably using a python virtual environment)<install>`


Changelog
=========
See the full :doc:`Changelog<changelog>`:

.. _latest_pynx:

.. include:: ../CHANGELOG.rst
   :end-before: Minor changes

Mailing list, git & issue tracker
=================================
There is a mailing list for PyNX: to subscribe, send an email to pynx-subscribe@esrf.fr

To access the git repository and the issue tracker, you can create an account on
http://gitlab.esrf and ask the developers for access to the PyNX project
at http://gitlab.esrf.fr/favre/PyNX


Citation & Bibliography
=======================
If you use PyNX for scientific work, please consider including a citation:

* If you use PyNX for coherent X-ray Imaging including CDI and ptychography:

 * Cite the 2020 PyNX article:
   `J. Appl. Cryst. 53 (2020), 1404–1413 <http://dx.doi.org/10.1107/S1600576720010985>`_

 * Give a link to the project: http://ftp.esrf.fr/pub/scisoft/PyNX/

* If you use PyNX for GPU scattering calculations:

 * Cite the first PyNX article:
   J. Appl. Cryst. 44(2011), 635-640. A preprint version is also available on ArXiv:1010.2641
 * Give a link to the project: http://ftp.esrf.fr/pub/scisoft/PyNX/

PyNX re-uses or was inspired by features described in the following articles and open-source software packages:
 * PtyPy: 1. B. Enders and P. Thibault, "A computational framework for ptychographic reconstructions",
   Proc Math Phys Eng Sci 472(2196), (2016).
 * M. Odstrčil, A. Menzel, and M. Guizar-Sicairos, "Iterative least-squares solver for
   generalized maximum-likelihood ptychography," Optics Express 26(3), 3108 (2018).
 * S. Marchesini, A. Schirotzek, C. Yang, H. Wu, and F. Maia, "Augmented projections for ptychographic imaging,"
   Inverse Problems 29(11), 115009 (2013).
 * P. Thibault and A. Menzel, "Reconstructing state mixtures from diffraction measurements," Nature 494(7435), 68–71 (2013).
 * P. Thibault and M. Guizar-Sicairos, "Maximum-likelihood refinement for coherent diffractive imaging," New J. Phys. 14(6), 063004 (2012).
 * J. N. Clark, X. Huang, R. Harder, and I. K. Robinson, "High-resolution three-dimensional partially coherent
   diffraction imaging," Nat Commun 3, 993 (2012).
 * S. Marchesini, "A unified evaluation of iterative projection algorithms for phase retrieval,"
   Review of Scientific Instruments 78(1), 011301 (2007).

License
=======
The PyNX library is distributed with a CeCILL-B license (an open-source license similar to the FreeBSD one).
See http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html

Note that CPU computing of the ``pynx.scattering`` module uses the ``sse_mathfun.h`` header, which is distributed
under the zlib license. See http://gruntthepeon.free.fr/ssemath/

See http://ftp.esrf.fr/pub/scisoft/PyNX/README.txt for more details about the license, copyright, as well as
other possible issues regarding ptychography.

Related software packages
=========================
If you are using PyNX for CDI, you should also consider the following:

* J. Carnis's `BCDI: tools for pre(post)-processing Bragg and forward coherent X-ray
  diffraction imaging data <https://github.com/carnisj/bcdi>`_
* D. Simonne's `Gwaihir: GUI for Bragg Coherent Coherent Diffraction (BCDI)
  data analysis <https://github.com/DSimonne/gwaihir>`_

Pynx.cdi: coherent diffraction imaging
======================================
.. pynx.cdi section

Description
-----------
This modules provides algorithm for 2D and 3D reconstruction of single objects using several algorithms:

* hybrid input-output (HIO)
* error reduction (ER)
* relaxed averaged alternating reflections (RAAR),
* maximum likelihood conjugate gradient minimization...
* partial coherence
* etc...

The calculations use an 'operator' approach, where operations on a given cdi dataset can be simply written
by multiplying it by the corresponding operator (FT, projection,..) or by a series of operators.

.. pynx.cdi end

Pynx.ptycho: Ptychography simulation and analysis
=================================================
.. pynx.ptycho section

Description
-----------
This modules allows the simulation and analysis of ptychography experiments, with the following features:

* 2D ptychography using a variety of algorithms: alternating projections, difference map (Thibault et al.), maximum likelihood conjugate gradient
* Works with any type of illumination (sharp or not)
* Object and/or Probe reconstruction
* Probe analysis (modes and focus)
* Incoherent background optimization
* GPU implementation using CUDA/OpenCL is available (and recommended), and is the main focus of current development:

  * example speed on single V1OO GPU card as of 2019/06: 13 ms/cycle for 1000 frames of 256x256 pixels and a simple
    alternating projection (17 ms/cycle for DM and 34 ms/cycle for ML)
  * GPU implementation allows using modes for probe and object
  * Maximum likelihood minimization (Poisson noise, regularisation)
* simple usage scripts are provided to analyse data from CXI files, ESRF beamlines (id01, id13, id16a), and ptypy files.

.. pynx.ptycho end

Pynx.scattering: X-ray scattering GPU computing
===============================================
.. pynx.scattering section

Description
-----------
This module aims to help computing scattering (X-ray or neutrons) for atomic structures, especially if they are distorted or disordered.

The library uses GPU computing (although parallel CPU computing is also available as a fall-back), with the following platforms:

* nVidia's CUDA toolkit and the pyCUDA library
* OpenCL language, along with pyOpenCL library.

Using GPU computing, PyNX provides fast parallel computation of scattering from large assemblies of atoms (>>1000 atoms) and 1D, 2D or 3D coordinates
(>>1000) in reciprocal space.

Typical computing speeds on GPUs more than 10^11 reflections.atoms/s on nVidia cards (3.5x10^11 on 2xTitan, 2x10^11 on a GTX 690, 5x10^10 on a
GTX295, 2.5x10^10 on a GTX465), more than 2 orders of magnitude faster than on a CPU.

Note that the main interest of *pynx.scattering* is the ability to compute scattering from *any* assembly of atoms (not regularly-spaced) to *any* set
of points in reciprocal space. While a FFT will always be faster, it is much more restrictive since the FFT imposes a strict relation between the
sampling in real (atomic positions) and reciprocal (hkl coordinates) space.

.. pynx.scattering end

Pynx.wavefront: Wavefront propagation
=====================================
.. pynx.wavefront section

Description
-----------
This module allows to propagate 2D wavefront using either:

* near field propagation
* far field propagation
* continuous propagation using the fractional Fourier Transform approach
* Sub-module ``pynx.wavefront.fzp`` can be used to calculate the coherent illumination from a Fresnel Zone Plate
* Sub-module ``pynx.wavefront.fresnel`` can be used to simulate

Calculations can be done using a GPU (OpenCL or CUDA), and use an 'operator' approach, where operations on a
given wavefront can be simply written by multiplying that wavefront by the corresponding operator or by a series
of operators.

.. pynx.wavefront end
