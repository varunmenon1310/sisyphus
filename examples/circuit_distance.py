"""Circuit-level distance from a Stim circuit (requires `pip install stim`).

The example builds a distance-5 rotated surface-code memory experiment with
5 rounds of noisy syndrome extraction, converts its detector error model into
check matrices, and searches for the smallest undetectable fault set that
flips the logical observable. Expected result: 5.

    python examples/circuit_distance.py
"""
import stim

from sisyphus_qec import dem_to_matrices, lift_columns, side_distance, verify_fault_set

if __name__ == "__main__":
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=5, rounds=5,
                                     after_clifford_depolarization=0.001,
                                     before_measure_flip_probability=0.001,
                                     after_reset_flip_probability=0.001)
    dem = circuit.detector_error_model(decompose_errors=False)
    H, L, columns = dem_to_matrices(dem)
    print(f"detector error model: {H.shape[0]} detectors x {H.shape[1]} distinct faults, "
          f"{L.shape[0]} observable(s)")

    res = side_distance(H, None, pair_rows=L, num_trials=100_000, num_workers=4, cutoff=5)
    faults = lift_columns(res.witness, columns)
    print(f"circuit distance <= {res.distance}  ({res.shots} iterations, {res.wall_time:.1f} s)")
    print("fault set (error indices in dem.flattened()):", faults)
    print("independent check on the DEM:", verify_fault_set(dem, faults))
