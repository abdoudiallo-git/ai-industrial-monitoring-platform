# Random Failures (RNF)

## Description
Random Failures represent machine breakdowns that occur independently of
any sensor reading — mechanical failures with no detectable precursor in
the monitored variables (e.g., unforeseen material defects, electrical
faults).

## Detection thresholds
RNF is modeled as a small constant probability (~0.1%) applied uniformly
regardless of sensor values. It exists in the dataset to represent
real-world failure modes that predictive models cannot capture from
sensor data alone.

## Recommended action
RNF is a reminder that no predictive model reaches 100% recall. A residual
baseline failure rate should always be assumed in maintenance planning,
regardless of model confidence.

## Related features
None — by design, RNF is independent of all sensor features.