# Tool Wear Failure (TWF)

## Description
Tool Wear Failure occurs when the cutting tool has been used beyond its safe
operational lifespan. As the tool degrades, cutting precision drops and the
risk of catastrophic tool breakage increases sharply.

## Detection thresholds
- Tool wear between 200 and 240 minutes: elevated failure risk window.
- Beyond this window, the tool is either replaced or fails, introducing a
  random element into the failure timing within that range.

## Recommended action
Schedule preventive tool replacement before tool_wear reaches 200 minutes.
Machines operating near this threshold should be flagged for inspection
regardless of other sensor readings.

## Related features
`tool_wear` (raw sensor value). Engineering feature `tool_wear_torque`
(tool_wear × torque) can help distinguish TWF from OSF when wear is high
but torque is abnormally elevated too.