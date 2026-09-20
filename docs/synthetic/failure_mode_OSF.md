# Overstrain Failure (OSF)

## Description
Overstrain Failure occurs when the combination of tool wear and torque
exceeds the mechanical tolerance of the tool for a given machine type,
leading to structural failure of the cutting tool.

## Detection thresholds
OSF occurs when `tool_wear × torque` exceeds a type-dependent threshold:
- Type L (low quality): threshold ≈ 11,000 minNm
- Type M (medium quality): threshold ≈ 12,000 minNm
- Type H (high quality): threshold ≈ 13,000 minNm

Higher-quality machine variants tolerate more combined stress before
overstrain occurs.

## Recommended action
Monitor `tool_wear_torque` relative to the machine's quality type. A
worn tool under high torque is significantly more dangerous on Type L
machines than on Type H machines.

## Related features
Engineered feature `tool_wear_torque` (tool_wear × torque), cross-referenced
with `machine_type`.