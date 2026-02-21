#!/usr/bin/env python3
import argparse
import glob
import math
import numpy as np
import os
import pickle
import re
import scipy.integrate
from types import SimpleNamespace

_a2r = np.pi / 180.0 / 60.0  # Arcmin to radians.
_sky = 41252.96              # Degrees on the sky.
PICO_NOMINAL_APERTURE = 1.4  # Baseline aperture in metres.

timescale_latex = {"minute":    "1\\,min",
                   "10minutes": "10\\,min",
                   "hour":      "1\\,hr",
                   "10hours":   "10\\,hr",
                   "year":      "1\\,yr",
                  }

# The following data are taken from Table 3.2 of the PICO Study Report.
# Columns are:
# - Band center [GHz]
# - Beam FWHM [arcmin]
# - CBE bolo NET [µK_CMB s¹ᐟ²]
# - N_bolo
# - CBE array NET [µK_CMB s¹ᐟ²]
# - Baseline array NET [µK_CMB s¹ᐟ²]
# - Baseline polarization map depth [µK_CMB arcmin]
# - Baseline polarization map depth [Jy sr⁻¹]
_pico_report_band_data = \
"""21 38.4   112  120  12.0  17.0   23.9    8.3
   25 32.0   103  200   8.4  11.9   18.4   10.9
   30 28.3  59.4  120   5.7   8.0   12.4   11.8
   36 23.6  54.4  200   4.0   5.7    7.9   12.9
   43 22.2  41.7  120   4.0   5.6    7.9   19.5
   52 18.4  38.4  200   2.8   4.0    5.7   23.8
   62 12.8  69.2  732   2.7   3.8    5.4   45.4
   75 10.7  65.4 1020   2.1   3.0    4.2   58.3
   90  9.5  37.7  732   1.4   2.0    2.8   59.3
  108  7.9  36.2 1020   1.1   1.6    2.3   77.3
  129  7.4  27.8  732   1.1   1.5    2.1   96.0
  155  6.2  27.5 1020   0.9   1.3    1.8  119.0
  186  4.3  70.8  960   2.0   2.8    4.0  433.0
  223  3.6  84.2  900   2.3   3.3    4.5  604.0
  268  3.2  54.8  960   1.5   2.2    3.1  433.0
  321  2.6  77.6  900   2.1   3.0    4.2  578.0
  385  2.5  69.1  960   2.3   3.2    4.5  429.0
  462  2.1   133  900   4.5   6.4    9.1  551.0
  555  1.5   658  440  23.0  32.5   45.8 1580.0
  666  1.3  2210  400  89.0 126.0  177.0 2080.0
  799  1.1 10400  360 526.0 744.0 1050.0 2880.0"""

def dplanck(f):
    """Copied from pixell, just to avoid importing for a single function.
	The derivative of the planck spectrum with respect to temperature, evaluated
	at frequencies f and temperature T, in units of Jy/sr/K."""
    # A blackbody has intensity I = 2hf**3/c**2/(exp(hf/kT)-1) = V/(exp(x)-1)
    # with V = 2hf**3/c**2, x = hf/kT.
    # dI/dx = -V/(exp(x)-1)**2 * exp(x)
    # dI/dT = dI/dx * dx/dT
    #       = 2hf**3/c**2/(exp(x)-1)**2*exp(x) * hf/k / T**2
    #       = 2*h**2*f**4/c**2/k/T**2 * exp(x)/(exp(x)-1)**2
    #       = 2*x**4 * k**3*T**2/(h**2*c**2) * exp(x)/(exp(x)-1)**2
    #       = .... /(4*sinh(x/2)**2)
    T = 2.72548  # Temperature of the CMB
    k = 1.3806488e-23
    h = 6.62606957e-34
    c = 299792458.0
    x     = h*f/(k*T)
    dIdT  = 2*x**4 * k**3*T**2/(h**2*c**2) / (4*np.sinh(x/2)**2) * 1e26
    return dIdT

class BandInfo(object):
    """Container for information on a PICO frequency band.
    Variables should be self-descriptive."""
    def __init__(self, nu, fwhm, bolo_net, n_bolo, array_net_cbe,
                 array_net_base, depth_uk_base, depth_jy_base):
        self.nu = nu
        self.fwhm = fwhm
        self.bolo_net = bolo_net
        self.n_bolo = n_bolo
        self.array_net_cbe = array_net_cbe
        self.array_net_base = array_net_base
        self.depth_uk_base = depth_uk_base
        self.depth_jy_base = depth_jy_base

        # Compute the solid angle in steradians from the FWHM in arcmin.
        sig2 = (self.fwhm * _a2r)**2 / 8 / np.log(2)
        self.sa = 2 * np.pi * sig2  # Assuming a Gaussian beam.

        # Compute the conversion from uK-arcmin to point-source sensitivity.
        # Multiply by 1e3 / 1e6 to convert from J/sr/K to mJy/sr/uK.
        sa_sq = 0.5 * self.sa   # Assuming a Gaussian beam
        self.fconv = dplanck(self.nu * 1e9) * 1e-3 * self.sa / np.sqrt(sa_sq)

class PICOForecast(object):
    """Class for doing simple forecasts of PICO map depths and point source
    sensitivities.

    Parameters
    ----------
    depth_table_path : str
        Path to pickle file made by Reijo with all the depth calculations.
    aperture : float, default=PICO_NOMINAL_APERTURE
        The primary aperture. The FWHM of the bands will be scaled accordingly
        from the PICO nominal aperture of 1.4 m.
    focal_plane_scaling : float, default=2.0
        If the aperture size is changed, in principle one could scale the number
        of detectors to fit the changed size of the focal plane. Parameterise
        the scaling as (A/A_0)**s, where A is the desired aperture size, A_0 is
        the PICO nominal aperture size (1.4 m), and s == `focal_plane_scaling`.
        The default value of 2.0 is the (optimistic) scenario where number of
        detectors scales with area.
    """
    def __init__(self, depth_table_path="depth_table.pck", 
                 aperture=PICO_NOMINAL_APERTURE, ndet_scaling=2.0):
        self.aperture = aperture
        self.ndet_scaling = ndet_scaling
        self.scale_ndet = (self.aperture / PICO_NOMINAL_APERTURE)**ndet_scaling

        self.band_info = {}
        for l in _pico_report_band_data.split("\n"):
            # N.B. IMPORTANT: currently the only thing being rescaled in the
            # pico band information is the FWHM. The NET and depths are
            # currently not being altered. (They are actually not used for
            # anything currently; just there since the band information is
            # copied directly from the Study Report.)
            if l[0] == "#":
                continue
            x = [float(z) for z in l.split()]
            # Second column is FWMH. Scale accordingly.
            x[1] *= PICO_NOMINAL_APERTURE / aperture
            self.band_info[int(x[0])] = BandInfo(*x)

        self._all_depth = pickle.load(open(depth_table_path, "rb"))
        self.timescale = [x for x in self._all_depth.keys()]
        self.band = [x for x in sorted(self.band_info.keys())]
        self.depth_quantile = [x for x in
                                 sorted(self._all_depth["minute"].keys())]

    def depth(self, band, timescale, quantile):
        """Returns a 1σ map depth/sensitivity in uK-arcmin."""
        depth_nominal = self._all_depth[timescale][quantile][band][0]
        return depth_nominal / np.sqrt(self.scale_ndet)

    def f_sky(self, band, timescale, quantile):
        """Returns the fraction of the sky covered."""
        return self._all_depth[timescale][quantile][band][1]

    def ps_sensitivity(self, band, timescale, quantile, n_sigma=1):
        """Returns a 1σ sensitivity to a point source in mJy."""
        return self.depth(band, timescale, quantile) * _a2r *\
               self.band_info[band].fconv

#so_test = [ArrayInfo( 93, 2.2,   0,    0,     0,     0,      0,      0),
#           ArrayInfo(145, 1.4,   0,    0,     0,     0,      0,      0)]
