"""Physical display units; solver arrays remain dimensionless."""
import numpy as np

C0 = 299792458.0
LENGTH_UNITS = {"nm": 1e-9, "μm": 1e-6, "m": 1.0}


def reference_length(value, unit="nm"):
    if value is None or str(value).strip() == "":
        return None
    length = float(value) * LENGTH_UNITS[unit]
    if not np.isfinite(length) or length <= 0:
        raise ValueError("Actual lattice constant must be positive and finite")
    return length


def frequency_factor(unit="Normalized", actual_lattice_constant_m=None):
    aliases = {"normalized": "Normalized", "Normalized frequency": "Normalized", "Hz": "GHz", "ghz": "GHz", "thz": "THz"}
    unit = aliases.get(unit, unit)
    if unit == "Normalized":
        return 1.0
    if unit not in {"GHz", "THz"}:
        raise ValueError("Choose Normalized, GHz or THz")
    if actual_lattice_constant_m is None:
        raise ValueError("Actual lattice constant is not set in this result; physical frequency is unavailable")
    length = reference_length(actual_lattice_constant_m, "m")
    return C0 / length / (1e12 if unit == "THz" else 1e9)


def frequency_label(unit):
    unit = {"normalized": "Normalized", "Normalized frequency": "Normalized", "Hz": "GHz"}.get(unit, unit)
    return "Normalized frequency (ωa/2πc)" if unit == "Normalized" else f"Frequency ({unit})"
