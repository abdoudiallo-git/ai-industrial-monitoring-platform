# Heat Dissipation Failure (HDF)

## Description
Heat Dissipation Failure happens when the machine cannot adequately dissipate
heat generated during the milling process, causing thermal buildup that
degrades the workpiece and stresses internal components.

## Detection thresholds
HDF occurs when both conditions are met simultaneously:
- The temperature difference between process temperature and air temperature
  (`process_temp - air_temp`) is below 8.6 K.
- Rotational speed is below 1380 rpm.

Low rotational speed reduces airflow around the tool, and a small
temperature differential means heat is not being carried away efficiently.

## Recommended action
Monitor `delta_temp` closely when rotational speed drops below 1400 rpm.
Consider forced cooling or reducing load if delta_temp approaches 8.6 K
under these conditions.

## Related features
Engineered feature `delta_temp` (process_temp − air_temp) is the primary
indicator, combined with `rotational_speed`.