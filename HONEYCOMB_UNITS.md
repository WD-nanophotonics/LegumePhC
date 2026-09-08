# Honeycomb and frequency units

Double-click Motif or choose Edit to open **Edit sites**. Each row has its own
shape, radius, rotation and material. Use Add site / Remove in this window.
For example, choose Triangle in both rows, radii 0.2 and 0.18, rotations 0 and
60 degrees. No comma arrays are entered in the UI. Adding a second site to a
standard centred triangular model initializes honeycomb A/B positions. Existing
custom positions stay intact; other new sites require explicit Position values.
Parameters are defined before affine deformation. Apply commits all rows together;
Cancel leaves the model unchanged. There is no separate shortcut panel.

Direct Python scripts use the same constructor:

```python
from studies.defaults import triangular_motif_parameters
motifs = triangular_motif_parameters(
    (0.2, 0.18), (0, 60), kind="polygon", sides=3, epsilon=1.0
)
```

Materials default to n; internally epsilon=n². Lengths are normalized to a=1.
**Actual lattice constant** sets the physical reference (new projects: 400 nm).
Advanced geometry scale is dimensionless and is preserved for old projects.
Affine deformation leaves the reference length unchanged.

In Plot Style select **Normalized / Hz / THz**, then **Apply to Current Plot**.
Normalized frequency is f*a/c. Physical frequency is f_normalized*c/a_actual.
At 400 nm, 0.3 equals 224.8443435 THz. Each result uses its saved reference
length; old results without one remain available in normalized units. To obtain
physical display for those cases, run again with the actual length set.

EFS levels use the selected units and convert when the unit selection changes.
Raw record arrays retain normalized frequencies. Berry curvature is unaffected
by the frequency display choice.
