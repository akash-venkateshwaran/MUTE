import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.repair import Repair


class AccelerationRepairVectorized(Repair):
    def __init__(self, max_iter: int = 7):
        super().__init__()
        self.max_iter = max_iter


    
    def _do(self, problem, X, **kwargs):
        # X has shape (population_size, num_variables)
        # problem is an instance of SpeedOptimizationProblem, giving access to its attributes

        # Get problem-specific parameters
        acc_min = problem.acc_min
        acc_max = problem.acc_max
        timestep_min = problem.timestep_min # This is your 1/60 hr
        
        # Get lower and upper bounds for all variables, ensuring they are broadcastable
        # problem.xl and problem.xu are typically 1D arrays of shape (n_var,)
        xl_broadcast = problem.xl[None, :] # Reshape to (1, n_var) for broadcasting
        xu_broadcast = problem.xu[None, :] # Reshape to (1, n_var) for broadcasting

        # Apply repair iteratively for a few passes to handle cascading effects
        # For vectorized operations, it's often more efficient to do a fixed small number of iterations
        # rather than checking for a 'repaired_in_this_pass' flag (which would require a loop anyway).
        max_repair_iterations = 5 # You can tune this; often 2-5 is sufficient

        repaired_X = X.copy() # Work on a copy to avoid modifying X in place prematurely

        for _ in range(max_repair_iterations):
            # Calculate accelerations for all individuals, for all segments
            # acc will have shape (population_size, n_var - 1)
            acc = np.diff(repaired_X, axis=1) / timestep_min

            # Find where upper acceleration bound is violated
            violated_upper = acc > acc_max
            
            # Find where lower acceleration bound is violated
            violated_lower = acc < acc_min
            
            # Create a mask for any acceleration violation
            any_violation = violated_upper | violated_lower

            # If no violations, we can break early from the repair iterations
            if not np.any(any_violation):
                break

            # Calculate the target speeds for the next step based on limits
            # These calculations are vectorized across all individuals and segments
            target_s_next_upper = repaired_X[:, :-1] + acc_max * timestep_min
            target_s_next_lower = repaired_X[:, :-1] + acc_min * timestep_min

            # Create an array to store the adjustments for the next speed (X[:, j+1])
            adjustments = np.zeros_like(acc) # Same shape as acc (pop_size, n_var-1)

            # Apply adjustments for upper bound violations
            # Only update where violated_upper is True
            adjustments[violated_upper] = target_s_next_upper[violated_upper]

            # Apply adjustments for lower bound violations
            # Only update where violated_lower is True
            adjustments[violated_lower] = target_s_next_lower[violated_lower]

            # Apply the adjustments to the 'next' speeds (X[:, j+1])
            # Only apply if there was a violation, otherwise keep original
            # This is an efficient way to update only the problematic spots
            repaired_X[:, 1:] = np.where(any_violation, adjustments, repaired_X[:, 1:])

            # Crucially, after adjusting for acceleration,
            # ensure all speeds are within their overall problem bounds (xl, xu).
            # This is also vectorized.
            repaired_X = np.clip(repaired_X, xl_broadcast, xu_broadcast)

        return repaired_X




# Dummy problem class
class SpeedOptimizationProblem:
    def __init__(self):
        self.n_var = 100
        self.timestep_min = 0.15  # hrs
        self.acc_min = -2.0       # knots/hr
        self.acc_max = 2.0        # knots/hr
        self.xl = np.full(self.n_var, 4.0)
        self.xu = np.full(self.n_var, 11.0)


# Generate test speed profiles
n_segments = 100
dt = 0.15
time = np.linspace(0, n_segments * dt, n_segments)
time_acc = time[1:]  # for acceleration plots

n0, n1, n2, n3 = 10, 25, 20, 45
part0 = np.full(n0, 10.0)
part1 = np.full(n1, 6.0)
part2 = np.full(n2, 8.0)
part3 = np.full(n3, 10.0)

# stitch together
profile_1 = np.concatenate([part0, part1, part2, part3])

# profile_1 = np.tile([5.0, 10.0], n_segments // 2)
profile_2 = np.linspace(5.0, 10.0, n_segments)
profile_3 = 5 + 5 * (np.arange(n_segments) % 10 < 5)
X = np.stack([profile_1, profile_2, profile_3], axis=0)

# Plot original speed + acceleration profiles
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
colors = ['tab:blue', 'tab:orange', 'tab:green']

for i in range(3):
    axs[0].plot(time, X[i], label=f"Profile {i+1}", color=colors[i])
    acc = np.diff(X[i]) / 0.15
    axs[1].plot(time_acc, acc, label=f"Accel {i+1}", color=colors[i])

axs[0].set_title("Original Speed Profiles")
axs[0].set_xlabel("Time (hr)")
axs[0].set_ylabel("Speed (knots)")
axs[0].grid(True)
axs[0].legend()

axs[1].set_title("Original Acceleration Profiles")
axs[1].set_xlabel("Time (hr)")
axs[1].set_ylabel("Acceleration (knots/hr)")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()


# Apply repair
problem = SpeedOptimizationProblem()
repair = AccelerationRepairVectorized(max_iter=7)
X_repaired = repair._do(problem, X.copy())

# Plot repaired speed + acceleration profiles
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

for i in range(3):
    axs[0].plot(time, X_repaired[i], label=f"Profile {i+1}", color=colors[i])
    acc = np.diff(X_repaired[i]) / 0.15
    axs[1].plot(time_acc, acc, label=f"Accel {i+1}", color=colors[i])

axs[0].set_title("Repaired Speed Profiles")
axs[0].set_xlabel("Time (hr)")
axs[0].set_ylabel("Speed (knots)")
axs[0].grid(True)
axs[0].legend()

axs[1].set_title("Repaired Acceleration Profiles")
axs[1].set_xlabel("Time (hr)")
axs[1].set_ylabel("Acceleration (knots/hr)")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()

dist_before = np.sum(X * dt, axis=1)     # shape (3,)

# distance after repair
dist_after  = np.sum(X_repaired * dt, axis=1)

# print results
for i, (db, da) in enumerate(zip(dist_before, dist_after), start=1):
    print(f"Profile {i}: before = {db:.3f} NM, after = {da:.3f} NM")
