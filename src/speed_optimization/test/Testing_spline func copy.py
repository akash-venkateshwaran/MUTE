import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, splrep, splev

def speed_profile_smoothener(x: np.ndarray,
                             t_low: np.ndarray,
                             t_high: np.ndarray,
                             spline_s: float = 0.2) -> np.ndarray:
    """
    Vectorized upsampling + B-spline smoothing.

    Parameters
    ----------
    x        : (n_gen, n_low) coarse speeds
    t_low    : (n_low,) low-resolution time vector
    t_high   : (n_high,) high-resolution time vector
    spline_s : smoothing factor for `splrep`

    Returns
    -------
    speed_hi : (n_gen, n_high) smoothed profile
    """
    n_gen, _ = x.shape

    # 1) Linear interpolation from t_low → t_high
    speed_hi = interp1d(t_low,
                        x,
                        kind='linear',
                        axis=1,
                        bounds_error=False,
                        fill_value='extrapolate')(t_high)

    # 2) Row‐wise B‐spline smoothing
    for i in range(n_gen):
        coeff = splrep(t_high, speed_hi[i], s=spline_s)
        speed_hi[i] = splev(t_high, coeff)

    return speed_hi


def main():
    # ------------------------------------------------------------
    # 1) Create a synthetic “coarse” profile (1 × 10 points)
    # ------------------------------------------------------------
    t_low = np.linspace(0, 10, 50)    # 10 coarse time steps between 0 and 10
    t_high = np.linspace(0, 10, 500)  # 500 “minute‐resolution” steps

    # Example: base sine wave + noise at coarse points
    np.random.seed(0)
    speeds_low = np.sin(t_low)[None, :] + 0.3 * np.random.randn(1, t_low.size)

    # ------------------------------------------------------------
    # 2) Apply smoothing with several spline_s values
    # ------------------------------------------------------------
    spline_s_values = [0.0, 0.1, 5.0, 10.0, 1000.0]
    smoothed_profiles = {}

    for s in spline_s_values:
        smoothed = speed_profile_smoothener(speeds_low, t_low, t_high, spline_s=s)
        smoothed_profiles[s] = smoothed[0]  # only one generation (row 0)

    # ------------------------------------------------------------
    # 3) Plot raw coarse data + all smoothed curves
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(t_low, speeds_low.flatten(), 'o', label='Coarse Data', markersize=6)
    for s, prof in smoothed_profiles.items():
        plt.plot(t_high, prof, label=f's = {s}')
    plt.title('Speed Profile: Different Spline‐Smoothing Factors')
    plt.xlabel('Time')
    plt.ylabel('Speed')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # 4) Compute and plot curvature (abs(second derivative)) of each
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    for s, prof in smoothed_profiles.items():
        # 2nd derivative via two successive gradients
        first_deriv = np.gradient(prof, t_high)
        second_deriv = np.gradient(first_deriv, t_high)
        curvature = np.abs(second_deriv)

        plt.plot(t_high, curvature, label=f's = {s}')
    plt.title('Curvature (|second derivative|) of Smoothed Profiles')
    plt.xlabel('Time')
    plt.ylabel('Curvature')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
