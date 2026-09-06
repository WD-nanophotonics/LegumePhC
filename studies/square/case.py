from studies.defaults import AffineParameters, CaseParameters, MaterialParameters, MotifParameters

CASE = CaseParameters(lattice="square", material=MaterialParameters(), motifs=(MotifParameters("circle", "circle", 0.2, center=(0.5, 0.5)),), affine=AffineParameters(), name="SquareCircle")
