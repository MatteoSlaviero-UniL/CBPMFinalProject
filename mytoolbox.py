import numpy as np
import qutip as qt
import os
import hashlib
import csv

# Constants
D = 2.87  # Zero-field splitting in GHz
g_e = -28.03  # Gyromagnetic ratio of the electron in GHz/T

# Define the electron spin (S=1) operators
Sz = qt.jmat(1, 'z')
Sx = qt.jmat(1, 'x')
Sy = qt.jmat(1, 'y')

def pulse(t, args):
    """Define the pulse function for time-dependent Hamiltonian."""
    return args['amp'] * np.sin(args['w'] * t)

def generate_cache_filename(params, folder='cache_fidelity'):
    """
    Generate a cache filename based on the hash of the input parameters.
    """
    hash_object = hashlib.md5(str(params).encode())
    hash_string = hash_object.hexdigest()
    return os.path.join(folder, f"{hash_string}.csv")

def read_cache(filename):
    """
    Read cached data from a CSV file.
    """
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    return float(data[1][1])

def write_cache(filename, params, fidelity_value):
    """
    Write data to a CSV file.
    """
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Parameter', 'Value'])
        for key, value in params.items():
            writer.writerow([key, value])
        writer.writerow(['Fidelity', fidelity_value])

def fidelity(t, initial_state, H0, n_evolution, evolution_operator, w, amp):
    """
    Compute the fidelity of the initial state after evolution under a sin pulsed operator.

    Parameters:
    - t (float): Time of evolution.
    - initial_state (Qobj): State to evolve from.
    - H0 (Qobj): Time-independent Hamiltonian.
    - n_evolution (int): Number of evolution steps.
    - evolution_operator (Qobj): Operator under which Hamiltonian should be pulsed.
    - w (float): Frequency of applied pulse.
    - amp (float): Amplitude of applied pulse.

    Returns:
    - fidelity (float): Fidelity of the state after evolution.
    """
    # # Generate parameters dictionary for hashing and caching
    # params = {
    #     't': t,
    #     'initial_state': initial_state.full().tolist(),
    #     'H0': H0.full().tolist(),
    #     'n_evolution': n_evolution,
    #     'evolution_operator': evolution_operator.full().tolist(),
    #     'w': w,
    #     'amp': amp
    # }

    # # Generate cache filename
    # cache_filename = generate_cache_filename(params)

    # # Check if cache file exists
    # if os.path.exists(cache_filename):
    #     # Read from cache
    #     return read_cache(cache_filename)

    # Compute fidelity
    density_matrix = initial_state * initial_state.dag()
    H = [H0, [evolution_operator, pulse]]
    times = np.linspace(0, t, n_evolution)
    result = qt.mesolve(H, initial_state, times, args={"w": w, 'amp': amp})
    fidelity_value = qt.expect(density_matrix, result.states[-1])

    # # Write to cache
    # write_cache(cache_filename, params, fidelity_value)

    return fidelity_value

def hamiltonian_given_B(D, g_e, B, polar, azimuthal):
    """
    Construct the Hamiltonian for given magnetic field parameters.

    Parameters:
    - D (float): Zero-field splitting in GHz.
    - g_e (float): Gyromagnetic ratio of the electron in GHz/T.
    - B (float): Magnetic field magnitude in Tesla.
    - polar (float): Polar angle in degrees.
    - azimuthal (float): Azimuthal angle in degrees.

    Returns:
    - H0 (Qobj): Hamiltonian.
    """
    Bz = B * np.cos(np.radians(polar))
    Bx = B * np.sin(np.radians(polar)) * np.cos(np.radians(azimuthal))
    By = B * np.sin(np.radians(polar)) * np.sin(np.radians(azimuthal))

    if not B == 0:
        # Sanity check to ensure the magnetic field is correctly calculated
        assert np.abs(Bx**2 + By**2 + Bz**2 - B**2) <= B / 1000

    H0 = D * Sz**2 + g_e * (Bz * Sz + Bx * Sx + By * Sy)
    return H0

def split_ratio_given_angles(D, g_e, B, polar, azimuthal):
    """
    Calculate the split ratio given the angles.

    Parameters:
    - D (float): Zero-field splitting in GHz.
    - g_e (float): Gyromagnetic ratio of the electron in GHz/T.
    - B (float): Magnetic field magnitude in Tesla.
    - polar (float): Polar angle in degrees.
    - azimuthal (float): Azimuthal angle in degrees.

    Returns:
    - split_ratio (float): Split ratio.
    """
    H0 = hamiltonian_given_B(D, g_e, B, polar, azimuthal)
    eigenenergies, _ = H0.eigenstates()
    fa = eigenenergies[1] - eigenenergies[0]
    fb = eigenenergies[2] - eigenenergies[0]

    return abs((D - fa) / (D - fb))

def angle_between_vectors(v1, v2):
    """
    Compute the angle between two vectors in degrees.

    Parameters:
    - v1 (tuple): First vector as (x1, y1, z1).
    - v2 (tuple): Second vector as (x2, y2, z2).

    Returns:
    - angle (float): Angle between the vectors in degrees.
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

def parameter_hash(params):
    """
    Generate a hash based on the parameters.

    Parameters:
    - params (tuple): Parameters to hash.

    Returns:
    - hash_str (str): SHA256 hash of the parameters.
    """
    params_str = "_".join(map(str, params))
    return hashlib.sha256(params_str.encode()).hexdigest()

def save_fidelities_to_csv(file_path, params, fidelities):
    """
    Save fidelities to a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file.
    - params (dict): Dictionary of parameters.
    - fidelities (list): List of fidelities.
    """
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the parameters in the preamble
        for key, value in params.items():
            writer.writerow([key, value])
        # Write a blank row for separation
        writer.writerow([])
        # Write the fidelities
        writer.writerow(['Frequencies', 'Fidelities'])
        for frequency, fidelity in zip(params['frequencies'], fidelities):
            writer.writerow([frequency, fidelity])

def odmr_cartesian(D, g_e, B_field, orientation, frequencies, t, n_evolution, initial_state, evolution_operator, amp):
    """
    Compute ODMR signal for given parameters in Cartesian coordinates.

    Parameters:
    - D (float): Zero-field splitting in GHz.
    - g_e (float): Gyromagnetic ratio of the electron in GHz/T.
    - B_field (tuple): Magnetic field vector in Cartesian coordinates (Bx, By, Bz) in Tesla.
    - orientation (tuple): NV orientation vector in Cartesian coordinates (nx, ny, nz).
    - frequencies (list): List of frequencies for which to compute fidelity.
    - t (float): Time for evolution under Hamiltonian before fidelity computation.
    - n_evolution (int): Number of data points for numerical solution of Schrödinger equation.
    - initial_state (Qobj): Initial state for evolution.
    - evolution_operator (Qobj): Operator under which Hamiltonian is pulsed.
    - amp (float): Amplitude of applied pulse.

    Returns:
    - fidelities (list): List of fidelities for each frequency.
    """
    # Create cache directory if it doesn't exist
    cache_dir = 'cache_odmr_cartesian'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Generate a hash for the parameters
    params = (D, g_e, B_field, orientation, tuple(frequencies), t, n_evolution, amp)
    hash_str = parameter_hash(params)
    cache_file = os.path.join(cache_dir, f'{hash_str}.csv')

    # Check if the cache file exists
    if os.path.exists(cache_file):
        return load_fidelities_from_csv(cache_file)

    # Compute the Hamiltonian using the Cartesian B field and orientation
    H0 = effective_hamiltonian(D, g_e, orientation, B_field)

    # Calculate fidelities
    fidelities = []
    for frequency in frequencies:
        fidelities.append(fidelity(t, initial_state, H0, n_evolution, evolution_operator, frequency, amp))

    # Save the fidelities to a cache file
    params_dict = {
        'D': D,
        'g_e': g_e,
        'B_field': B_field,
        'orientation': orientation,
        'frequencies': frequencies,
        't': t,
        'n_evolution': n_evolution,
        'amp': amp
    }
    save_fidelities_to_csv(cache_file, params_dict, fidelities)

    return fidelities

def load_fidelities_from_csv(file_path):
    """
    Load fidelities from a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - fidelities (list): List of fidelities.
    """
    fidelities = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row and row[0] == 'Frequencies':
                break  # Skip preamble
        for row in reader:
            if row:
                fidelities.append(float(row[1]))
    return fidelities

def odmr(D, g_e, B, polar, azimuthal, nv_orientation, frequencies, t, n_evolution, initial_state, evolution_operator, amp):
    """
    Compute ODMR signal for given parameters, using a cache to avoid redundant calculations.

    Parameters:
    - D (float): Zero-field splitting in GHz.
    - g_e (float): Gyromagnetic ratio of the electron in GHz/T.
    - B (float): Magnetic field magnitude in Tesla.
    - polar (float): Polar angle in degrees.
    - azimuthal (float): Azimuthal angle in degrees.
    - nv_orientation (tuple): NV orientation vector as (x, y, z).
    - frequencies (list): List of frequencies for which to compute fidelity.
    - t (float): Time for evolution under Hamiltonian before fidelity computation.
    - n_evolution (int): Number of data points for numerical solution of Schrödinger equation.
    - initial_state (Qobj): Initial state for evolution.
    - evolution_operator (Qobj): Operator under which Hamiltonian is pulsed.
    - amp (float): Amplitude of applied pulse.

    Returns:
    - fidelities (list): List of fidelities for each frequency.
    """
    # Create cache directory if it doesn't exist
    cache_dir = 'cache_odmr'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Generate a hash for the parameters
    params = (D, g_e, B, polar, azimuthal, nv_orientation, tuple(frequencies), t, n_evolution, amp)
    hash_str = parameter_hash(params)
    cache_file = os.path.join(cache_dir, f'{hash_str}.csv')

    # Check if the cache file exists
    if os.path.exists(cache_file):
        return load_fidelities_from_csv(cache_file)

    # Compute the Hamiltonian
    Bz = B * np.cos(np.radians(polar))
    Bx = B * np.sin(np.radians(polar)) * np.cos(np.radians(azimuthal))
    By = B * np.sin(np.radians(polar)) * np.sin(np.radians(azimuthal))
    effective_polar = angle_between_vectors(nv_orientation, (Bx, By, Bz))
    H0 = hamiltonian_given_B(D, g_e, B, effective_polar, 0)

    # Calculate fidelities
    fidelities = []
    for frequency in frequencies:
        fidelities.append(fidelity(t, initial_state, H0, n_evolution, evolution_operator, frequency, amp))

    # Save the fidelities to a cache file
    params_dict = {
        'D': D,
        'g_e': g_e,
        'B': B,
        'polar': polar,
        'azimuthal': azimuthal,
        'nv_orientation': nv_orientation,
        'frequencies': frequencies,
        't': t,
        'n_evolution': n_evolution,
        'amp': amp
    }
    save_fidelities_to_csv(cache_file, params_dict, fidelities)

    return fidelities

def compute_eigenvalues(D, g_e, orientation, B_field):
    eigenvalues = effective_hamiltonian(D, g_e, orientation, B_field).eigenenergies()
    
    return eigenvalues

def effective_hamiltonian(D, g_e, orientation, B_field):
    # Normalize the orientation vector
    n_x, n_y, n_z = orientation / np.linalg.norm(orientation)
    
    # Define spin-1 operators
    Sx = qt.jmat(1, 'x')
    Sy = qt.jmat(1, 'y')
    Sz = qt.jmat(1, 'z')
    
    B_parallel, B_perpendicular = decompose_vector(orientation, B_field)
    
    # Hamiltonian H = 2π(D * (Sz^2) + g_e * (B_parallel * Sz))
    return D * Sz**2 + g_e * (B_parallel * Sz + B_perpendicular * Sx)

def decompose_vector(orientation, B_field):
    # Normalize the orientation vector
    orientation = np.array(orientation)
    B_field = np.array(B_field)
    
    orientation_norm = np.linalg.norm(orientation)
    
    if orientation_norm == 0:
        raise ValueError("Orientation vector must be non-zero.")
    
    # Unit vector along orientation
    unit_orientation = orientation / orientation_norm
    
    # Projection of B_field onto orientation (parallel component)
    B_parallel_magnitude = np.dot(B_field, unit_orientation)
    B_parallel = B_parallel_magnitude * unit_orientation
    
    # Perpendicular component of B_field
    B_perpendicular = B_field - B_parallel
    
    return np.linalg.norm(B_parallel), np.linalg.norm(B_perpendicular)
