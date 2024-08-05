import numpy as np
import qutip as qt

D = 2.87  # Zero-field splitting in GHz
B = 0.01  # Magnetic field in Tesla
g_e = -28.03  # Gyromagnetic ratio of the electron, units are GHz/T
_2PI = 2*np.pi
polar = 2   # Angle wrt z axis in degrees
azimuthal = 0   #Angle around the equator in degrees

# Define the electron spin (S=1) operators
Sz = qt.jmat(1, 'z')
Sx = qt.jmat(1, 'x')
Sy = qt.jmat(1, 'y')

def pulse(t, args):
    return args['amp'] * np.sin(args['w'] * t)

# Function to compute fidelity of initial state after evolution under sin pulsed operator
# t: time of evolution
# initial_state: state to evolve from, defines density matrix used for fidelity computation
# H0: time independent hamiltonian
# evolution_operator: operator under which hamiltonian should be pulse
# w: frequency of applied pulse
# amp: amplitude of applied pulse
def fidelity(t, initial_state, H0, n_evolution, evolution_operator, w, amp):
    density_matrix = initial_state * initial_state.dag()
    H = [H0, [evolution_operator, pulse]]
    times = np.linspace(0, t, n_evolution)
    return qt.expect(density_matrix, qt.mesolve(H, initial_state, times, args={"w": w, 'amp': amp}).states[-1])

def hamiltonian_given_B(D, g_e, B, polar, azimuthal):
    Bz = B * np.cos(polar * np.pi / 180)
    Bx = B * np.sin(polar * np.pi / 180) * np.cos(azimuthal * np.pi / 180)
    By = B * np.sin(polar * np.pi / 180) * np.sin(azimuthal * np.pi / 180)

    # Sanity check to make sure conversion is correct
    assert(np.abs(Bx**2 + By**2 + Bz**2 - B**2) < B/1000)
    H0 = D * Sz**2 + g_e * (Bz * Sz + Bx * Sx + By * Sy)

    return H0

def split_ratio_given_angles(D, g_e, B, polar, azimuthal):
    H0 = hamiltonian_given_B(D, g_e, B, polar, azimuthal)
    eigenenergies, eigenstates = H0.eigenstates()
    fa = eigenenergies[1]-eigenenergies[0]
    fb = eigenenergies[2]-eigenenergies[0]

    return abs((D - fa)/(D - fb))

def angle_between_vectors(v1, v2):
    """
    Compute the angle between two vectors in degrees.
    
    Parameters:
    - v1 (tuple): First vector as (x1, y1, z1)
    - v2 (tuple): Second vector as (x2, y2, z2)
    
    Returns:
    - angle (float): Angle between the vectors in degrees
    """
    # Unpack vectors
    x1, y1, z1 = v1
    x2, y2, z2 = v2
    
    # Compute dot product
    dot_product = x1 * x2 + y1 * y2 + z1 * z2
    
    # Compute magnitudes
    magnitude_v1 = np.sqrt(x1**2 + y1**2 + z1**2)
    magnitude_v2 = np.sqrt(x2**2 + y2**2 + z2**2)
    
    # Compute cosine of the angle
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    
    # Handle numerical issues and compute the angle in radians
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    
    return np.degrees(theta)

# Returns list of fidelities, one for each frequency
# D: zero field splitting (GHz)
# g_e: electron gyromagnetic ratio (GHz/T)
# B: magnitude of external magnetic field (T)
# polar: angle wrt z axis of external magnetic field (deg)
# azimuthal: angle on equatorial plane of externam magnetic field (deg)
# nv_orientation: tuple of form (x,y,z) describing the orientation of NV in space, example (1,1,1), magnitude is irrelevant
# frequencies: list of frequencies for which to compute fidelity
# t: time for evolution under Hamiltonian before fidelity is computed
# n_evolution: datapoints for numerical solution of Schroedinger EQ
# initial_state: state to evolve from, defines density matrix used for fidelity computation
# H0: time independent hamiltonian
# evolution_operator: operator under which hamiltonian should be pulse
# amp: amplitude of applied pulse
def odmr(D, g_e, B, polar, azimuthal, nv_orientation, frequencies, t, n_evolution, initial_state, evolution_operator, amp):
    Bz = B * np.cos(polar * np.pi / 180)
    Bx = B * np.sin(polar * np.pi / 180) * np.cos(azimuthal * np.pi / 180)
    By = B * np.sin(polar * np.pi / 180) * np.sin(azimuthal * np.pi / 180)

    effective_polar = angle_between_vectors(nv_orientation, (Bx, By, Bz))
    
    H0 = hamiltonian_given_B(D, g_e, B, effective_polar, 0)

    fidelities = []
    for frequency in frequencies:
        fidelities.append(fidelity(t, initial_state, H0, n_evolution, evolution_operator, frequency, amp))

    return fidelities