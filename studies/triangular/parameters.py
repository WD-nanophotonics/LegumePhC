from dataclasses import replace
from studies.defaults import BCDParameters, BandParameters, BerryParameters, EFSParameters, FieldsParameters, FrequencyParameters, SolverParameters

SOLVER = SolverParameters(gmax=2.0, numeig=4, polarization="te")
FREQUENCY = replace(FrequencyParameters(), solver=replace(SOLVER), qpoint=(0.2, 0.07), band_one_based=2)
BAND = replace(BandParameters(), solver=replace(SOLVER), path="identity", samples_per_segment=16)
FIELDS = replace(FieldsParameters(), solver=replace(SOLVER), qpoint=(0.2, 0.07), bands_one_based=(2, 3), grid_size=8)
EFS = replace(EFSParameters(), solver=replace(SOLVER), bands_one_based=(2, 3), grid_size=5)
BERRY = replace(BerryParameters(), solver=replace(SOLVER), sampling_mode="first_bz_grid", grid_size=3, step=0.01, bands_one_based=(2, 3), rank=2)
BCD = replace(BCDParameters())
