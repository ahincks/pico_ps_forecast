This is a lightweight tool for accessing and manipulating pre-computed map depth forecasts for a PICO-like satellite.

The depth forecasts are stored in a pickle file (`depth_table.pck`). These were computed by Reijo Keskitalo using the array properties and forecasted sensitivities from the [2019 PICO Study Report](https://science.nasa.gov/wp-content/uploads/2023/04/PICO_Study_Rpt.pdf). The forecast is an array with the following dimensions:
- _Timescale_ — Map depths are computed on various time scales, including 1 minutes (the spacecraft spin period), 10 hours (the spacecraft spin axis precession period) and one year.
- _Quantile_ — The depth achieved during a given timescale is not uniform over the coverage area. Quantiles of the achieved depth are provided.
- _Band_ — Forecasts are given for all 21 of the PICO frequency bands.

Each array element contains two numbers:
- The map depth, reported as the 1σ error level, in μK-arcmin.
- The fraction of the full sky in which this map depth is achieved.

The `pico_ps_forecast.py` module provides a convience method to access these array elements (`PICOForecast.depth()`), as well as a method for returning the 1σ sensitivity to a point source (`PICOForecast.ps_sensitivity()`). For the latter, it is assumed that the noise is not background dominated. (A background-dominated forecast is perhaps to be added.)
