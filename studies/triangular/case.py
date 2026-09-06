from studies.defaults import AffineParameters, CaseParameters, MaterialParameters, MotifParameters

CASE = CaseParameters(
    lattice="triangular",
    material=MaterialParameters(),
    motifs=(
        MotifParameters("G15-A", "polygon", 80.14335684352235 / 400.0, 15, 0.0, (0.0, 0.0)),
        MotifParameters("G15-B", "polygon", 75.13439704080221 / 400.0, 15, 60.0, (0.5, 1.0 / (2.0 * 3.0 ** 0.5))),
    ),
    name="G15",
    affine=AffineParameters(),
)
