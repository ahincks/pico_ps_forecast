#!/usr/bin/env python3

import pico_ps_forecast as pps

# Initialise model.
pico = pps.PICOForecast("depth_table.pck")

print("Available frequencies (GHz):", end="")
for i, nu in enumerate(pico.band):
    if i % 3 == 0:
        print("\n   ", end="")
    print(" %3d" % nu, end="")
print("\n") 
print("Available integration timescales:")
for ts in pico.timescale:
    print("    %s" % ts)
print()
print("Available depth quantiles:", end="")
for i, q in enumerate(pico.depth_quantile):
    if i % 5 == 0:
        print("\n   ", end="")
    print(" %.2f" % q, end="")
print("\n")

# Map depth forecasts are given as 1σ noise levels as a function of depth
# quantile, in units of uK-arcmin. For instance, the depth for the 0.5
# quantile means that the map is at least that deep over 50% of the sky area
# observed by the detector array during a particular timescale. The f_sky for
# each percentile can also be accessed.
nu = 108
timescale = "minute"
quantile = 0.5
print("Depth at %d GHz is < %.0f μK-arcmin over f_sky = %.3f on a %s "\
      "timescale." % (nu, pico.depth(nu, timescale, quantile),
                      pico.f_sky(nu, timescale, quantile), timescale))

# You can also get the 1σ sensitivity to a point source, in units of mJy.
print("PS sensitivity for same parameters is %.0f mJy" %\
      pico.ps_sensitivity(nu, timescale, quantile))

# Now get the sensitivity for a 2 metre probe, assuming we can increase the
# number of detectors as the area of the aperture (ndet_scaling=2.0).
pico2m = pps.PICOForecast("depth_table.pck", aperture=2.0, ndet_scaling=2.0)
print("PS sensitivity for same parameters with 2-metre dish: %.0f mJy" %\
      pico2m.ps_sensitivity(nu, timescale, quantile))
