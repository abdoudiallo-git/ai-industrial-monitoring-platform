# Power Failure (PWF)

## Description
Power Failure occurs when the mechanical power delivered to the cutting
process falls outside the machine's safe operating envelope — either too
low to perform the cut effectively, or too high, causing mechanical stress.

## Detection thresholds
Power is computed as: power = torque × rotational_speed × (2π / 60)

PWF occurs when:
- Power drops below 3500 W (insufficient cutting power), or
- Power exceeds 9000 W (excessive mechanical stress).

## Recommended action
Track the `power` feature continuously. Sustained operation near either
boundary (3500 W or 9000 W) should trigger a maintenance review even
before an actual failure is recorded.

## Related features
Engineered feature `power` (torque × rotational_speed × 2π/60) directly
encodes this failure mode.