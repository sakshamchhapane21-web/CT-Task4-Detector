import numpy as np
import xraylib


# ==============================================================
# MATERIAL DEFINITIONS
# ==============================================================

MATERIALS = {

    # ----------------------------------------------------------
    # 0 = Air
    # ----------------------------------------------------------
    0: {
        "name": "Air",
        "nist_name": "Air, Dry (near sea level)",
    },

    # ----------------------------------------------------------
    # 1 = Soft tissue
    # ----------------------------------------------------------
    1: {
        "name": "Soft Tissue",
        "nist_name": "Tissue, Soft (ICRP)",
    },

    # ----------------------------------------------------------
    # 2 = Bone
    # ----------------------------------------------------------
    2: {
        "name": "Cortical Bone",
        "nist_name": "Bone, Cortical (ICRP)",
    },

    # ----------------------------------------------------------
    # 3 = Lung
    # ----------------------------------------------------------
    3: {
        "name": "Lung",
        "nist_name": "Lung (ICRP)",
    },

    # ----------------------------------------------------------
    # 4 = Adipose
    # ----------------------------------------------------------
    4: {
        "name": "Adipose",
        "nist_name": "Adipose Tissue (ICRP)",
    },

    # ----------------------------------------------------------
    # 5 = Water
    # ----------------------------------------------------------
    5: {
        "name": "Water",
        "nist_name": "Water, Liquid",
    },

    # ----------------------------------------------------------
    # 6 = Skeletal muscle
    # ----------------------------------------------------------
    6: {
        "name": "Skeletal Muscle",
        "nist_name": "Muscle, Skeletal",
    },
}


# ==============================================================
# MATERIAL INFORMATION
# ==============================================================

def get_material_info(material_id):
    """
    Returns material information.
    """
    # used in-- 
    #   1. mass_attenutaion_coefficient()
    #   2. material_density()

    if material_id not in MATERIALS:
        raise ValueError(
            f"Unknown material ID: {material_id}"
        )

    return MATERIALS[material_id]


# ==============================================================
# MASS ATTENUATION COEFFICIENT
# ==============================================================

def mass_attenuation_coefficient(
    material_id,
    energy_keV
):
    """
    Returns mass attenuation coefficient.

    Units:
        cm^2 / g

    Uses xraylib NIST compound data.
    """

    material = get_material_info(
        material_id
    )
    # print(f"\nMaterial encountered: {material["name"]}; NIST name: {material["nist_name"]}")

    
    # return mass attenuation coefficient
    ## ????????????????????????????????? checkpoint ???????????????????????????????? 
    
    value = xraylib.CS_Total_CP(
        material["nist_name"],
        float(energy_keV)
    )
    
    # energy(keV)  :   1, 2,  2.5,   3, ... 50
    
    # value(cm^2/g) : 3606, 527,  277 , 162, ... 0.2

    return value


# ==============================================================
# DENSITY
# ==============================================================

def material_density(material_id):
    """
    Returns material density.

    Units:
        g / cm^3
        
    Role: Usefule in calculating linear attenuation coefficient by multiplying with mass attenuation coefficient
    """

    material = get_material_info(
        material_id
    )

    # get density by usnig material_id , units: g/cm^3 @ 20deg C and std Pressure (tested for Air)
    data = xraylib.GetCompoundDataNISTByName(
        material["nist_name"]
    )

    return data["density"]


# ==============================================================
# LINEAR ATTENUATION COEFFICIENT
# ==============================================================

def linear_attenuation_coefficient(
    material_id,
    energy_keV
):
    """
    Returns linear attenuation coefficient.

    μ = (μ/rho) * rho

    xraylib:
        μ/rho -> cm^2/g
        rho   -> g/cm^3

    Therefore:
        μ -> cm^-1

    Converted to:
        mm^-1
    """

    mu_over_rho = mass_attenuation_coefficient(
        material_id,
        energy_keV
    )

    rho = material_density(
        material_id
    )

    # calculating --> linear_att_coef = mass_att_coef * density
    mu_cm = mu_over_rho * rho

    ## cm^-1  --> mm^-1 conversion
    mu_mm = mu_cm / 10.0

    # Verified : got the correct linear_att_coef values, also follows the trend of decreasing vlaues on increasing energies
    ## value for 1keV for air is correct as 0.466 mm^-1 (but got 0.434 mm^-1)
    return mu_mm


# ==============================================================
# DEBUG / INFORMATION
# ==============================================================

def print_material_database():

    print("\nMaterial database")
    print("-" * 60)

    for material_id, material in MATERIALS.items():

        density = material_density(
            material_id
        )

        print(
            f"{material_id}: "
            f"{material['name']:20s} "
            f"density = {density:.4f} g/cm^3"
        )