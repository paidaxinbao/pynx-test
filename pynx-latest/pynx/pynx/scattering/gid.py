# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2008-2015 : Univ. Joseph Fourier (Grenoble 1), CEA/INAC/SP2M
#   (c) 2016-present : Univ. Grenoble Alpes, CEA/INAC/SP2M
#   (c) 2016-present : ESRF-European Synchrotron Radiation Facility
#       author: Vincent Favre-Nicolin vincent.favre-nicolin@univ-grenoble-alpes.fr, favre@esrf.fr

import warnings
import scipy
from scipy import pi, take


def cctbx_version():
    # Code copied from: cctbx_sources/cctbx/command_line/version.py
    version = None
    for tag_file in ["TAG", "cctbx_bundle_TAG"]:
        tag_path = libtbx.env.under_dist("libtbx", os.path.join("..", tag_file))
        if (os.path.isfile(tag_path)):
            try:
                version = open(tag_path).read().strip()
            except KeyboardInterrupt:
                raise
            except:
                pass
            else:
                break
    if (version is None):
        version = libtbx.env.command_version_suffix
    return version


try:
    import cctbx
    from cctbx import crystal
    from cctbx import xray
    from cctbx import miller
    from cctbx import uctbx
    from cctbx import sgtbx
    from cctbx.array_family import flex
    from cctbx.eltbx import sasaki
    from cctbx.eltbx import tiny_pse

    USE_CCTBX = True
    xray_scatterer = xray.scatterer
except:
    USE_CCTBX = False
    # Dummy class to avoid error when generating documentation
    class xray_scatterer(object):
        pass

# Electron classical radius (Angstroems)
re = 2.814e-15

try:
    from . import fhkl

    has_pynx_gpu = True
except:
    has_pynx_gpu = False


def W2E(x):
    """ nrj->wavelength or wavelength->nrj , nrj in eV, wavelength in Angstroems"""
    return 12398.4 / x

class Scatterer(xray_scatterer):
    """
    Class used to store scatterer properties, including f' and f'' values.
    """
    def __init__(self, label, site, occup, u, energy, fp=None, fs=None):
        """
        
        :param label: name of the scatterer
        :param site: the site to occupy
        :param occup: occupancy
        :param u: isotropic displacement parameter
        :param energy: energy in keV
        :param fp: f'
        :param fs: f''
        """
        xray.scatterer.__init__(self, label=label, site=site, occupancy=occup, u=u)
        self.nrj = energy
        if fp is None or fs is None:
            fp_fdp_sasaki = sasaki.table(self.element_symbol()).at_ev(self.nrj)
            self.fp0 = fp_fdp_sasaki.fp()
            self.fs0 = fp_fdp_sasaki.fdp()
        else:
            self.fp0 = fp0
            self.fs0 = fs

    def GetF0(self, stol):
        """
        Get the thomson scattering factor for this scatterer
        
        :param stol: sin(theta)/lambda value, either a scalar or a numpy array
        :return: 
        """
        scatt = flex.xray_scatterer((self,))
        scatt_dict = xray.ext.scattering_dictionary(scatt)
        scatt_dict.assign_from_table("WK1995")
        return scatt_dict.dict()[self.label].gaussian.at_stol(stol)


class Crystal(object):
    """
    Description of a material content.
    """
    def __init__(self, unitcell, spacegroup, scatterers):
        self.sg = sgtbx.space_group_info(spacegroup)
        self.uc = uctbx.unit_cell(unitcell)
        self.scatt = scatterers
        self.cctbx_scatterers = flex.xray_scatterer()
        for s in self.scatt:
            self.cctbx_scatterers.append(xray.scatterer(label=s.label, site=s.site, occupancy=s.occupancy, u=s.u_iso))
        try:
            # old cctbx version
            xray.add_scatterers_ext(unit_cell=self.uc,
                                    space_group=self.sg.group(),
                                    scatterers=self.cctbx_scatterers,
                                    site_symmetry_table=sgtbx.site_symmetry_table(),
                                    site_symmetry_table_for_new=sgtbx.site_symmetry_table(),
                                    min_distance_sym_equiv=0.5,
                                    u_star_tolerance=0,
                                    assert_min_distance_sym_equiv=True)
        except:
            # cctbx version >= 2011_04_06_0217
            # print "Whoops, cctbx version 2011"
            xray.add_scatterers_ext(unit_cell=self.uc,
                                    space_group=self.sg.group(),
                                    scatterers=self.cctbx_scatterers,
                                    site_symmetry_table=sgtbx.site_symmetry_table(),
                                    site_symmetry_table_for_new=sgtbx.site_symmetry_table(),
                                    min_distance_sym_equiv=0.5,
                                    u_star_tolerance=0,
                                    assert_min_distance_sym_equiv=True,
                                    non_unit_occupancy_implies_min_distance_sym_equiv_zero=False)
        cs = crystal.symmetry(self.uc, spacegroup)
        sp = crystal.special_position_settings(cs)
        self.structure = xray.structure(sp, self.cctbx_scatterers)
        self.structure_as_P1 = self.structure.expand_to_p1()

    def GetF0(self, scatterer, h, k, l):
        """
        Get the thomson scattering factor for a scatterer at given h,k,l reciprocal unit coordinates

        :param scatterer: the scatterer
        :param h: reciprocal lattice unit
        :param k: reciprocal lattice unit
        :param l: reciprocal lattice unit
        :return: a numpy array of the thomson scattering factor with the same shape as h, k, l.
        """
        scatt = flex.xray_scatterer((scatterer,))
        scatt_dict = xray.ext.scattering_dictionary(scatt)
        scatt_dict.assign_from_table("WK1995")
        return scatt_dict.dict()[scatt.label].gaussian.at_stol(self.uc.stol((h, k, l)))

    def CalcMultiplicity(self):
        for i in self.scatt:
            ss = i.apply_symmetry(self.uc, self.sg.group())

    def GetAtomDensity(self):
        """ Number of atoms per cubic meter"""
        vol = self.uc.volume() * 1e-30
        self.CalcMultiplicity()
        density = dict()
        for s in self.scatt:
            # site symmetry      #TODO : calculate real multiplicity for special positions !
            # ssym=s.apply_symmetry(self.uc,self.sg.group())
            density[s.label] = s.occupancy * s.multiplicity() / vol
        return density

    def GetDensity(self):
        """ kg/m^3 """
        vol = self.uc.volume() * 1e-30
        self.CalcMultiplicity()
        density = 0
        for s in self.scatt:
            # site symmetry      #TODO : calculate real multiplicity for special positions !
            # ssym=s.apply_symmetry(self.uc,self.spg.group())
            tpse = tiny_pse.table(s.element_symbol())
            density += s.occupancy * s.multiplicity() * tpse.weight() / 1000 / (vol * 6.0221353e23)
        return density

    def GetRefractionIndexDeltaBeta(self):
        """ Refraction index, n=1-delta-i*beta. returns a tuple delta,beta"""
        delta = 0.0
        beta = 0.0
        density = self.GetAtomDensity()
        wav = W2E(self.scatt[0].nrj) * 1e-10  # assume all scatterers have same list of nrjs
        density = self.GetAtomDensity()
        for s in self.scatt:
            z = sasaki.table(s.label).atomic_number()
            fp_fdp_sasaki = sasaki.table(s.element_symbol()).at_ev(self.scatt[0].nrj)
            delta += wav ** 2 * density[s.label] * re / (2 * pi) * (z - s.fp0)
            beta += wav ** 2 * density[s.label] * re / (2 * pi) * s.fs0
        return delta, beta

    def GetCriticalAngle(self):
        """
        Calculate the critical angle for this material
        :return: the critical angle, in radians
        """
        delta, beta = self.GetRefractionIndexDeltaBeta()
        return scipy.sqrt(2 * delta)

    def GetLinearAbsorptionCoeff(self):
        """
        Calculate the linear absorption coefficient for this material
        :return: 
        """
        mu = 0.0
        density = self.GetAtomDensity()
        for s in self.scatt:
            mu += 5639.9e-28 * W2E(s.nrj) * s.fs0 * density[s.label]
        return mu


class Wave(object):
    def __init__(self, alphai, eparall, eperp, nrj):
        """
            Incident wave above a surface.
            Coordinates:
              - z is perpendicular to the surface, >0 going UP (different from H Dosch's convention)
              - x is the projection of the wavevector on the surface
              - y is parallel to the surface

            alphai: incident angle, with respect to the surface
            eparallel: component of the electric field parallel to the incident plane (vertical plane)
            eperp: component of the electric field perpendicular to the incident plane (along y)
            nrj: values of the energy of the incident wave, in eV

            alphai *or* nrj can be arrays, but not together
        """
        self.alphai = alphai
        self.eparall = eparall
        self.eperp = eperp
        self.ex = scipy.sin(alphai) * eparall
        self.ey = eperp
        self.ez = scipy.cos(alphai) * eparall
        self.kx = 2 * pi / W2E(nrj) * scipy.cos(alphai)
        self.ky = 2 * pi / W2E(nrj) * 0
        self.kz = -2 * pi / W2E(nrj) * scipy.sin(alphai)
        self.nrj = nrj


class DistortedWave:
    """
      Reflected and refracted wave at an interface.

      This will compute the reflection and refraction coefficients,
      as well as the transmitted (complex) wavevector).

      Calculation can be done by supplying the Wave object, and either:

      * the materials used as pynx.gid.Crystal objects:
        - material0: upper (incoming) layer material, can be None (air or vacuum)
        - material1: lower layer material

      * the delta, beta values for the difference of the refraction index
        between the lower and the upper layer (delta: real part, beta: imaginary part)

      If delta and beta are supplied, then material0 and material1 are ignored, otherwise
      the refraction index delta and beta will be calculated using cctbx from
      material1 and material0.
    """

    def __init__(self, material0, material1, wave, delta=None, beta=None):
        if delta is not None and beta is not None:
            self.delta = delta
            self.beta = beta
        else:
            # use cctbx to determine the reazl and imaginary part of the refraction index
            if material0 is None:
                self.delta, self.beta = material1.GetRefractionIndexDeltaBeta()
            elif material1 is None:
                self.delta, self.beta = material0.GetRefractionIndexDeltaBeta()
                self.delta, self.beta = -self.delta, -self.beta
            else:
                delta0, beta0 = material0.GetRefractionIndexDeltaBeta()
                delta1, beta1 = material1.GetRefractionIndexDeltaBeta()
                self.delta = delta1 - delta0
                self.beta = beta1 - beta0
        n = 1 - self.delta - self.beta * 1j
        self.sinr_alphai = (scipy.sin(wave.alphai) ** 2 - 2 * self.delta - self.beta * 2j) ** 0.5
        self.krx = wave.kx * 1.0
        self.kry = wave.ky * 0
        self.krz = -wave.kz
        self.ktx = wave.kx
        self.kty = 0.0
        self.ktz = -2 * pi / W2E(wave.nrj) * self.sinr_alphai
        self.Tix = 2 * self.sinr_alphai / (n ** 2 * scipy.sin(wave.alphai) + self.sinr_alphai)
        self.Tiy = 2 * scipy.sin(wave.alphai) / (scipy.sin(wave.alphai) + self.sinr_alphai)
        self.Tiz = 2 * scipy.sin(wave.alphai) / (n ** 2 * scipy.sin(wave.alphai) + self.sinr_alphai)
        self.Rix = -(n ** 2 * scipy.sin(wave.alphai) - self.sinr_alphai) / (
        n ** 2 * scipy.sin(wave.alphai) + self.sinr_alphai)
        self.Riy = (scipy.sin(wave.alphai) - self.sinr_alphai) / (scipy.sin(wave.alphai) + self.sinr_alphai)
        self.Riz = (n ** 2 * scipy.sin(wave.alphai) - self.sinr_alphai) / (
        n ** 2 * scipy.sin(wave.alphai) + self.sinr_alphai)
        self.PenetrationDepth = 1 / self.ktz.imag  # in Angstroems
        self.erx = self.Rix * wave.ex
        self.ery = self.Riy * wave.ey
        self.erz = self.Riz * wave.ez
        self.etx = self.Tix * wave.ex
        self.ety = self.Tiy * wave.ey
        self.etz = self.Tiz * wave.ez
        ne2 = abs(wave.ex) ** 2 + abs(wave.ey) ** 2 + abs(wave.ez) ** 2
        net2 = abs(self.etx) ** 2 + abs(self.ety) ** 2 + abs(self.etz) ** 2
        ner2 = abs(self.erx) ** 2 + abs(self.ery) ** 2 + abs(self.erz) ** 2
        self.Ti2 = net2 / ne2
        self.Ri2 = ner2 / ne2


def fhkl_dwba4(x, y, z, h, k, l=None, occ=None, alphai=0.2, alphaf=None, substrate=None, wavelength=1.0, e_par=0.,
              e_perp=1.0, gpu_name="CPU", use_fractional=True, language="OpenCL", cl_platform="",
              separate_paths=False, **kwargs):
    """
    Calculate the grazing-incidence X-ray scattered intensity taking into account
    4 scattering paths, for a nanostructure object located above a given substrate.
    The 5th path is the scattering from the substrate, assumed to be below the
    interface at z=0.

    :param x,y,z: coordinates of the atoms in fractionnal coordinates (relative to the
                  substrate unit cell)- if use_fractional==False, these should be given in Angstroems
    :param h,k,l: reciprocal space coordinates. If use_fractional==False, these should be given
                  in inverse Angstroems (multiplied by 2pi - physicist 'k' convention, abs(k)=4pisin(theta)/lambda,
                  i.e. these correspond to k_x,k_y,k_z).
    :param alphai, alphaf: incident and outgoing angles, in radians
    :param substrate: the substrate material, as a pynx.gid.Crystal object - this will be used
                      to calculate the material refraction index.
    :param wavelength: in Angstroems
    :param e_par,e_perp: percentage of polarisation parallel and perpendicular to the incident plane
    :param use_fractional: if True (the default), then coordinates for atoms and reciprocal
                           space are given relatively to the unit cell, otherwise in Angstroems
                           and 2pi*inverse Angstroems.

    Note: Either l or alphaf must be supplied - it is assumed that the lattice
    coordinates are such that the [001] direction is perpendicular to the surface.
    """
    nrj = W2E(wavelength)
    if 'use_fractionnal' in kwargs:
        use_fractional = kwargs['use_fractionnal']
    if use_fractional:
        c = substrate.uc.parameters()[2]
        s_fact = 1.0
    else:
        c = 2 * pi
        s_fact = 1 / c
    if alphaf is None:
        # alphaf, computed from l: l.c* = (sin(alpha_f) + sin(alpha_i))/wavelength
        alphaf = scipy.arcsin(l / c * wavelength - scipy.sin(alphai))

    # Incident wave
    w = Wave(alphai, e_par, e_perp, nrj)
    dw = DistortedWave(None, substrate, w)
    # Reflected wave after the dot
    w1 = Wave(alphaf, e_par, e_perp, nrj)
    dw1 = DistortedWave(None, substrate, w1)

    # First path, direct diffraction
    l = c * scipy.sin(alphaf + alphai) / wavelength
    f1 = fhkl.Fhkl_thread(h * s_fact, k * s_fact, l * s_fact, x, y, z, occ=occ, gpu_name=gpu_name, language=language,
                         cl_platform=cl_platform)[0]

    # Second path, reflection before
    l = c * scipy.sin(alphaf - alphai) / wavelength
    f2 = fhkl.Fhkl_thread(h * s_fact, k * s_fact, l * s_fact, x, y, z, occ=occ, gpu_name=gpu_name, language=language,
                         cl_platform=cl_platform)[0] * dw.Riy

    # Third path, reflection after dot
    l = c * scipy.sin(-alphaf + alphai) / wavelength
    f3 = fhkl.Fhkl_thread(h * s_fact, k * s_fact, l * s_fact, x, y, z, occ=occ, gpu_name=gpu_name, language=language,
                         cl_platform=cl_platform)[0] * dw1.Riy

    # Fourth path, reflection before and after dot
    l = c * scipy.sin(-alphaf - alphai) / wavelength
    f4 = fhkl.Fhkl_thread(h * s_fact, k * s_fact, l * s_fact, x, y, z, occ=occ, gpu_name=gpu_name, language=language,
                         cl_platform=cl_platform)[0] * dw.Riy * dw1.Riy
    if separate_paths: return f1, f2, f3, f4
    return f1 + f2 + f3 + f4


def FhklDWBA4(x, y, z, h, k, l=None, occ=None, alphai=0.2, alphaf=None, substrate=None, wavelength=1.0, e_par=0.,
              e_perp=1.0, gpu_name="CPU", use_fractional=True, language="OpenCL", cl_platform="",
              separate_paths=False, **kwargs):
    if 'use_fractionnal' in kwargs:
        use_fractional = kwargs['use_fractionnal']
    warnings.warn("FhklDWBA4() is deprecated. Use fhkl_dwba4() instead.", DeprecationWarning)
    return fhkl_dwba4(x, y, z, h, k, l, occ, alphai, alphaf, substrate, wavelength, e_par,
              e_perp, gpu_name, use_fractional, language, cl_platform, separate_paths)

def fhkl_dwba5(x, y, z, h, k, l=None, occ=None, alphai=0.2, alphaf=None, substrate=None, wavelength=1.0, e_par=0.,
              e_perp=1.0, gpu_name="CPU", use_fractional=True, verbose=False, language="OpenCL", cl_platform="",
              separate_paths=False):
    """
    WARNING: this code is still in development, and needs to be checked !

    Calculate the grazing-incidence X-ray scattered intensity taking into account
    5 scattering paths, for a nanostructure object located above a given substrate.
    All atoms with z>0 are assumed to be above the surface, and their
    scattering is computed using the 4 DWBA paths. Atoms with z<=0 are below
    the surface, and their scattering is computed using a single path,
    taking into account the refraction and the attenuation length.

    :param x,y,z: coordinates of the atoms in fractionnal coordinates (relative to the
                  substrate unit cell)
    :param h,k,l: reciprocal space coordinates
    :param alphai, alphaf: incident and outgoing angles, in radians
    :param substrate: the substrate material, as a pynx.gid.Crystal object - this will be used
                      to calculate the material refraction index.
    :param wavelength: in Angstroems
    :param e_par,e_perp: percentage of polarisation parallel and perpendicular to the incident plane

    Note: Either l *OR* alphaf must be supplied - it is assumed that the lattice
    coordinates are such that the [001] direction is perpendicular to the surface.
    """
    nrj = W2E(wavelength)

    # Atoms above the surface #
    tmpx = (x + (z + y) * 0).ravel()
    tmpy = (y + (x + z) * 0).ravel()
    tmpz = (z + (x + y) * 0).ravel()
    idx = scipy.nonzero(tmpz > 0)
    if len(idx[0]) > 0:
        if type(occ) != type(None):
            tmpocc = take(occ, idx)
        else:
            tmpocc = None
        f1234 = FhklDWBA4(take(tmpx, idx), take(tmpy, idx), take(tmpz, idx), h, k, l=l, occ=tmpocc, alphai=alphai,
                          alphaf=alphaf, substrate=substrate, wavelength=wavelength, e_par=e_par, e_perp=e_perp,
                          gpu_name=gpu_name, use_fractional=use_fractional, language=language,
                          cl_platform=cl_platform, separate_paths=separate_paths)
    else:
        f1234 = 0
    # Atoms below the surface
    idx = scipy.nonzero(tmpz <= 0)
    if len(idx[0]) > 0:
        if use_fractional:
            c = substrate.uc.parameters()[2]
            s_fact = 1.0
        else:
            c = 2 * pi
            s_fact = 1 / c
        if alphaf is None:
            # alphaf, computed from l and alphai
            alphaf = scipy.arcsin(l / c * wavelength - scipy.sin(alphai))
        else:
            tmpl = (scipy.sin(alphaf) + scipy.sin(alphai)) / wavelength
            if verbose: print("From alphaf: l=%4.2f -> %4.2f" % (tmpl.min(), tmpl.max()))

        wi = Wave(alphai, e_par, e_perp, nrj)
        dwi = DistortedWave(None, substrate, wi)
        # TODO For outgoing beam: check e_par and e_perp, signs for k real and imag...
        wf = Wave(alphaf, e_par, e_perp, nrj)
        dwf = DistortedWave(None, substrate, wf)
        # kz, transmitted
        kz_real, kz_imag = (-dwf.ktz - dwi.ktz).real, (dwf.ktz + dwi.ktz).imag
        if verbose:
            print("wi.kz, dwi.ktz:", wi.kz, dwi.ktz)
            print("kz_below real:", kz_real)
            print("kz_below imag:", kz_imag)
            print("kz_below mean:", kz_real.mean(), kz_imag.mean())
            # print dwf.ktz-dwi.ktz
        # print dwi.Tiy,dwf.Tiy
        # print kz_real,kz_imag
        # Compute scattering
        if type(occ) != type(None):
            tmpocc = take(occ, idx)
        else:
            tmpocc = None
        if use_fractional:
            l_real = c * kz_real / (2 * pi)
            l_imag = c * kz_imag / (2 * pi)
        else:
            l_real = kz_real / (2 * pi)
            l_imag = kz_imag / (2 * pi)
        f5 = fhkl.Fhkl_thread(h * s_fact, k * s_fact, l_real, take(tmpx, idx), take(tmpy, idx), take(tmpz, idx), occ=tmpocc,
                              gpu_name=gpu_name, sz_imag=l_imag, language=language, cl_platform=cl_platform)[0] * dwi.Tiy * (
        -dwf.Tiy)
    else:
        f5 = 0
    if separate_paths: return f1234[0], f1234[1], f1234[2], f1234[3], f5
    return f1234 + f5


def FhklDWBA5(x, y, z, h, k, l=None, occ=None, alphai=0.2, alphaf=None, substrate=None, wavelength=1.0, e_par=0.,
              e_perp=1.0, gpu_name="CPU", use_fractional=True, verbose=False, language="OpenCL", cl_platform="",
              separate_paths=False, **kwargs):
    if 'use_fractionnal' in kwargs:
        use_fractional = kwargs['use_fractionnal']
    warnings.warn("FhklDWBA5() is deprecated. Use fhkl_dwba5() instead.", DeprecationWarning)
    return fhkl_dwba5(x, y, z, h, k, l, occ, alphai, alphaf, substrate, wavelength, e_par,
              e_perp, gpu_name, use_fractional, verbose, language, cl_platform, separate_paths)