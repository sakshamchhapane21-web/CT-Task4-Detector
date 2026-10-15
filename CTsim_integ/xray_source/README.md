# Task [1]: X Ray Source

### 1. Module Overview 
X-Ray Source Simulation Module

### 2. Inputs & Outputs
- **Inputs:**
  - ### From the Detector / Geometry Team (Per Projection)
  - source_position: Target center coordinates (x, y, z) in mm.
  - detector_position: Detector center coordinates $(x, y, z)$ in mm.
  - cone_angle: Beam opening angle (radians or degrees) toward the detector.

  - ### Protocol Settings (User-Controlled)
  - kVp: Tube potential (defines max photon energy and spectrum shape).
  - mA: Tube current (scales photon intensity/weight).
  - exposure_time: Exposure duration per view (scales photon intensity/weight).
  - focal_spot_size: Target dimensions $(dx, dy)$ (e.g., $0.6\text{ mm}$ or $1.2\text{ mm}$).
  - N_photons: Number of simulated rays to trace per view (e.g., $10^5$).
  
- **Outputs:**
  - photon_positions: Array of shape (N, 3)  Starting $(x, y, z)$ coordinates jittered across the focal spot area.
  - photon_directions: Array of shape (N, 3)  Unit vectors pointing within the cone toward the detector.
  - photon_energies: Array of shape (N,)  Sampled energy values (keV) matching your filtered spectrum.
  - photon_weight (or total intensity): Scalar value representing physical flux, calculated as $\frac{\text{mA} \times \text{exposureTime}}{N}$.

### 3. Current Progress
- X-ray spectrum calculation: Generates the filtered Kramers Bremsstrahlung energy curve.
- Energy sampling: Draws $N$ discrete photon energies matching the spectrum probabilities.
- Beam direction sampling: Generates unit direction vectors uniformly inside a 3D cone toward the detector.

### 4. How We Will Validate Results
- Energy Spectrum Distribution: Plot a normalized histogram of the sampled photon_energies against the analytical Kramers curve to verify matching peak locations and high-energy cutoffs.
- Beam Hardening Verification: Compute the mean energy $\bar{E}$ before and after filtration to confirm the expected shift toward higher energies due to low-energy absorption.
- Spatial Trajectory & Cone Check: Visualize 3D ray origins and unit direction vectors to ensure origins are bounded by the focal spot dimensions and trajectories stay within cone_angle.


### 5. Code File
Our main python code can be found Codes folder under main 
