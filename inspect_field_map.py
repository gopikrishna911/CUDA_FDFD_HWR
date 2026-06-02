"""
inspect_field_map.py
Reads cavity_field.bin and reports E/B ratios at the locations where the
analytical TEM mode predicts peaks, instead of just the global maxima.

Usage:  python3 inspect_field_map.py cavity_field.bin
"""

import sys
import struct
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python3 inspect_field_map.py cavity_field.bin")
    sys.exit(1)

path = sys.argv[1]
with open(path, "rb") as fp:
    # Header: 120 bytes
    hdr = fp.read(120)
    (magic, version, Nr, Nphi, Nz_cav, _pad0,
     a, b, L, dr, dphi, dz, omega, freq_Hz, peakEr_pre,
     _r1, _r2, _r3) = struct.unpack("<IIiiii dddddddddddd", hdr)

    print(f"magic   = 0x{magic:08x}  (expected 0xCAFE0001)")
    print(f"version = {version}")
    print(f"grid    = {Nr} x {Nphi} x {Nz_cav}")
    print(f"a, b, L = {a:.4f}, {b:.4f}, {L:.4f}")
    print(f"dr, dphi, dz = {dr:.4e}, {dphi:.4e}, {dz:.4e}")
    print(f"omega   = {omega:.6e} rad/s  (f = {freq_Hz/1e6:.4f} MHz)")
    print(f"peak |Er| before norm = {peakEr_pre:.4e}")
    print()

    N = Nr * Nphi * Nz_cav
    def read_arr():
        raw = fp.read(8 * N)
        return np.frombuffer(raw, dtype=np.float64).reshape(Nz_cav, Nphi, Nr)
    Er   = read_arr()
    Ephi = read_arr()
    Ez   = read_arr()
    Br   = read_arr()
    Bphi = read_arr()
    Bz   = read_arr()

# --- where does the analytical TEM mode put peaks? --------------------
# Er ~ (1/r) sin(pi*z/L)  --> peak at r=a (i=0), z=L/2 (k=Nz_cav//2)
# Bphi ~ (1/r) cos(pi*z/L) --> peak at r=a (i=0), z near 0 or L (k=0 or Nz_cav-1)
print(f"Global   |E|max = {max(np.abs(Er).max(), np.abs(Ephi).max(), np.abs(Ez).max()):.4e}")
print(f"Global   |B|max = {max(np.abs(Br).max(), np.abs(Bphi).max(), np.abs(Bz).max()):.4e}")
print()

# Component peaks
for name, F in [("Er", Er), ("Ephi", Ephi), ("Ez", Ez),
                ("Br", Br), ("Bphi", Bphi), ("Bz", Bz)]:
    idx = np.unravel_index(np.argmax(np.abs(F)), F.shape)
    k_max, j_max, i_max = idx
    r_loc = a + (i_max + 0.5)*dr
    z_loc = (k_max + 0.5)*dz
    print(f"  |{name}|max = {abs(F[idx]):.4e}  at (i={i_max}, j={j_max}, k={k_max})  r={r_loc:.3f}m, z={z_loc:.3f}m")

print()
# --- Predicted analytical ratio ---------------------------------------
# For TEM half-wave between PEC plates:
#   Er(r,z)   = E0/r * sin(pi*z/L)
#   Bphi(r,z) = -(E0/(r*omega)) * (pi/L) * cos(pi*z/L)
# so |Er|/|Bphi| = omega*L/pi = 2*f*L
c = 2.998e8
print(f"Analytic |Er|max/|Bphi|max = 2*f*L = {2*freq_Hz*L:.4e}  (≈ c if half-wave)")

# At the SAME (i,j) -- innermost radial cell, all phi -- compare peaks along z
i0 = 0
print()
print(f"At r = a+0.5*dr = {a + 0.5*dr:.4f} m (innermost cell):")
maxEr_along_z = np.abs(Er[:, :, i0]).max()
maxBphi_along_z = np.abs(Bphi[:, :, i0]).max()
print(f"   max|Er|   = {maxEr_along_z:.4e}")
print(f"   max|Bphi| = {maxBphi_along_z:.4e}")
if maxBphi_along_z > 0:
    print(f"   ratio     = {maxEr_along_z/maxBphi_along_z:.4e}  (expect c = {c:.4e})")