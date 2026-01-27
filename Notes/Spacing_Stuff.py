import math

def calculate_max_fit(build_l, build_w, build_h, part_l, part_w, part_h, spacing):
    """
    Calculates the maximum number of rectangular parts that can fit in a single 2D layer.
    """
    # If the part is taller than the build box, it cannot fit in this orientation
    if part_h > build_h:
        return 0
    
    # Formula for N parts with spacing S on all sides (including walls):
    # (N * Part) + ((N + 1) * S) <= BuildDim 
    # N * (Part + S) <= BuildDim - S
    def count_fit(bl, bw, pl, pw):
        num_l = math.floor((bl - spacing) / (pl + spacing))
        num_w = math.floor((bw - spacing) / (pw + spacing))
        return max(0, num_l) * max(0, num_w)

    # Check both horizontal rotations of the part footprint
    fit_rotation_a = count_fit(build_l, build_w, part_l, part_w)
    fit_rotation_b = count_fit(build_l, build_w, part_w, part_l)
    
    return max(fit_rotation_a, fit_rotation_b)

# InnoventX Build Box (mm)
BL, BW, BH = 160, 65, 65
S = 3 # Spacing

# Part Dimensions [Length, Width, Thickness] (mm)
type_iv = [115, 19, 3.40]
type_v = [63.50, 9.53, 3.40]

specs = {"Type IV": type_iv, "Type V": type_v}

for name, dims in specs.items():
    l, w, t = dims
    print(f"--- {name} Results ---")
    print(f"Orientation 1 (Flat):    {calculate_max_fit(BL, BW, BH, l, w, t, S)} units")
    print(f"Orientation 2 (Side):    {calculate_max_fit(BL, BW, BH, l, t, w, S)} units")
    print(f"Orientation 3 (Upright): {calculate_max_fit(BL, BW, BH, w, t, l, S)} units\n")