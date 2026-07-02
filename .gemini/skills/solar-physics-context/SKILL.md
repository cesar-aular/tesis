---
name: solar-physics-context
description: Domain knowledge for solar photovoltaic generation forecasting. Use this to validate feature engineering, ensure physical consistency in forecasts, and interpret meteorological impacts (GHI, temperature, zenith angle).
---

# Solar Physics Context

This skill provides the physical foundations for solar power forecasting.

## Core Physical Drivers
- **Global Horizontal Irradiance (GHI)**: The primary driver. Ensure a near-linear correlation with power output during daylight hours.
- **Air Temperature**: High temperatures generally reduce photovoltaic efficiency (Negative coefficient).
- **Zenith Angle**: Models should implicitly or explicitly handle the solar trajectory. 

## Physical Consistency Checks
1. **Nighttime Zero**: Solar generation MUST be zero when GHI is zero or during nighttime hours (calculated via Lat/Lon and timestamp).
2. **Clear Sky Index**: Forecasts exceeding the theoretical Clear Sky GHI for a given location/time should be flagged as physical anomalies.

## Advanced Feature Engineering
- **Solar Position**: Use `hora_sin`/`hora_cos` and `mes_sin`/`mes_cos` to capture seasonality.
- **Atmospheric Attenuation**: Humidity and cloud cover are non-linear attenuators of GHI.
- **Spatio-temporal Correlation**: Nearby plants (e.g., within 50km) should show high error correlation due to shared weather fronts.
