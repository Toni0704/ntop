# physics_model.py

import math

def physics_model(velocity_inlet, x_cell_size, yz_cell_size):
    """
    Physics-based estimates for AvgVelocity, Mass, PressureDrop, and SurfaceArea
    given Velocity Inlet and lattice cell sizes.

    Parameters:
    velocity_inlet : float  # m/s
    x_cell_size    : float  # mm
    yz_cell_size   : float  # mm

    Returns:
    dict with AvgVelocity, Mass, PressureDrop, SurfaceArea
    """

    # ---- Constants (adjust to match hackathon fluid/material) ----
    rho = 1.225          # kg/m^3, density of air at 20°C
    mu = 1.81e-5         # Pa·s, dynamic viscosity of air
    core_length = 0.1    # m, assumed lattice length in flow direction
    Nx = 10              # number of cells along x direction
    Ny = 10              # number of cells along y
    Nz = 10              # number of cells along z
    strut_thickness = 0.001  # m

    # Convert mm to meters
    x_cell_m = x_cell_size / 1000.0
    yz_cell_m = yz_cell_size / 1000.0

    # ---- 1. AvgVelocity ----
    # For ideal flow, assume average velocity equals inlet velocity minus minor losses
    loss_factor = 0.05  # guessed pressure/velocity loss fraction
    avg_velocity = velocity_inlet * (1 - loss_factor)

    # ---- 2. Mass ----
    # Mass of the lattice = solid volume × material density
    material_density = 2700  # kg/m^3 for aluminum
    cell_volume = x_cell_m * yz_cell_m * yz_cell_m
    strut_volume_per_cell = (3 * strut_thickness * yz_cell_m * yz_cell_m)  # crude estimate
    solid_fraction = strut_volume_per_cell / cell_volume
    total_volume = Nx * Ny * Nz * cell_volume
    mass = total_volume * solid_fraction * material_density

    # ---- 3. PressureDrop ----
    # Use Ergun equation for porous media
    particle_diameter = min(x_cell_m, yz_cell_m)
    porosity = 1 - solid_fraction
    Re = rho * avg_velocity * particle_diameter / mu
    term1 = (150 * (1 - porosity)**2 / (porosity**3)) * (mu * avg_velocity / particle_diameter)
    term2 = (1.75 * (1 - porosity) / (porosity**3)) * (rho * avg_velocity**2 / particle_diameter)
    pressure_drop = (term1 + term2) * core_length

    # ---- 4. SurfaceArea ----
    # Approximate as strut surface area times total struts
    strut_length_per_cell = 3 * yz_cell_m  # 3 edges per cell cube (flow aligned)
    strut_surface_area_per_cell = strut_length_per_cell * strut_thickness * 4  # 4 faces
    surface_area = Nx * Ny * Nz * strut_surface_area_per_cell

    return {
        "AvgVelocity": avg_velocity,
        "Mass": mass,
        "PressureDrop": pressure_drop,
        "SurfaceArea": surface_area
    }

# Example usage:
if __name__ == "__main__":
    result = physics_model(velocity_inlet=2500, x_cell_size=10, yz_cell_size=10)
    for k, v in result.items():
        print(f"{k}: {v}")
