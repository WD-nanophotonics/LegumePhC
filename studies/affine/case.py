from studies.defaults import AffineParameters, CaseParameters, MaterialParameters, MotifParameters

CASE = CaseParameters(
    lattice="square",
    material=MaterialParameters(),
    motifs=(MotifParameters("circle", "circle", 0.2, center=(0.5, 0.5)),),
    affine=AffineParameters(linear=((1.0, 0.18), (0.0, 0.92)), translation=(0.07, -0.03)),
    name="AffineSquareCircle",
)
