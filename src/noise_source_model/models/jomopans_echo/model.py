import numpy as np
import matplotlib.pyplot as plt


def compute_L_s_f_JE(f, V, l, C, V_C, l_0):
    L_s_f_0 = compute_baseline(f, C, V_C)
    term_velocity = 60 * np.log10(V / V_C)
    term_length = 20 * np.log10(l / l_0)
    L_s_f_JE = L_s_f_0 + term_velocity + term_length
    return L_s_f_JE


def compute_L_s_f_JE_matrix(f, V, l, C, V_C, l_0):

    L_s_f_0 = compute_baseline(f, C, V_C)  # Shape: (1000,)
    term_velocity = 60 * np.log10(V / V_C)  # Shape: (100, 15)
    term_length = 20 * np.log10(l / l_0)  # Scalar
    
    L_s_f_0_expanded = L_s_f_0.reshape(1, 1, -1)  # Shape: (1, 1, 1000)
    term_length_expanded = np.array(term_length).reshape(1, 1, 1)  # Shape: (1, 1, 1)
    
    # Compute L_s_f_JE with shape (100, 15, 1000)
    L_s_f_JE = L_s_f_0_expanded + term_velocity[:, :, np.newaxis] + term_length_expanded
    
    return L_s_f_JE

def compute_baseline(f, C, V_C):
    f_ref = 1  # Reference frequency in Hz
    V_ref = 1  # Reference speed in knots
    f_hat = f / f_ref
    L_s_f_0 = np.zeros_like(f)

    idx_below_100 = f_hat < 100
    idx_above_100 = f_hat >= 100
    f_hat_below_100 = f_hat[idx_below_100]
    f_hat_above_100 = f_hat[idx_above_100]

    if f_hat_below_100.size > 0 and C in ['Containership', 'Vehicle Carrier', 'Tanker', 'Bulker']:
        K_LF = 208  # Constant in dB
        f1_LF = 600 * (V_ref / V_C)
        D_LF = 0.8 if C in ['Containership', 'Bulker'] else 1.0

        term1_below = K_LF - 40 * np.log10(f1_LF)
        term2_below = 10 * np.log10(f_hat_below_100)
        term3_below = -10 * np.log10((1 - (f_hat_below_100 / f1_LF)**2)**2 + D_LF**2)
        L_s_f_0[idx_below_100] = term1_below + term2_below + term3_below

    if f_hat_above_100.size > 0:
        K = 191  # Constant in dB
        f1 = 480 * (V_ref / V_C)
        D = 4 if C == 'Cruise' else 3

        term1_above = K - 20 * np.log10(f1)
        if C in ['Containership', 'Vehicle Carrier', 'Tanker', 'Bulker']:
            term2_above = -10 * np.log10((1 - (f_hat_above_100 / f1))**2 + D**2)
            L_s_f_0[idx_above_100] = term1_above + term2_above
        else:
            term2_above = -10 * np.log10((1 - (f_hat / f1))**2 + D**2)
            L_s_f_0 = term1_above + term2_above

    return L_s_f_0

def get_vessel_ref_speed(vessel_class):
    vessel_dict = {
        'Fishing': 6.4,
        'Tug': 3.7,
        'Naval': 11.1,
        'Recreational': 10.6,
        'Government/Research': 8.0,
        'Cruise': 17.1,
        'Passenger': 9.7,
        'Bulker': 13.9,
        'Containership': 18.0,
        'Vehicle Carrier': 15.8,
        'Tanker': 12.4,
        'Other': 7.4,
        'Dredger': 9.5
    }

    if vessel_class in vessel_dict:
        return vessel_dict[vessel_class]
    else:
        raise ValueError(f'The vessel class "{vessel_class}" is not recognized.')

def convert_to_decidecade(f, L_s_f_JE):

    f_decidecade_full = np.array([12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000])
    f_decidecade = f_decidecade_full[(f_decidecade_full >= f.min()) & (f_decidecade_full <= f.max())]
    L_s_JE_decidecade = np.zeros_like(f_decidecade)

    for j in range(len(f_decidecade)):
        idx = np.argmin(np.abs(f - f_decidecade[j]))
        L_s_JE_decidecade[j] = L_s_f_JE[idx] + 10 * np.log10(0.231 * f_decidecade[j])

    return L_s_JE_decidecade, f_decidecade

def convert_to_decidecade_matrix(f, L_s_f_JE):

    f_decidecade_full = np.array([12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000])
    f_decidecade = f_decidecade_full[(f_decidecade_full >= f.min()) & (f_decidecade_full <= f.max())]
    
    L_s_JE_decidecade = np.zeros((L_s_f_JE.shape[0], L_s_f_JE.shape[1], len(f_decidecade))) # Shape: (100, 15, len(f_decidecade))
    f_decidecade_broadcast = f_decidecade.reshape(1, 1, -1)  # Shape: (1, 1, len(f_decidecade))
    f_broadcast = f.reshape(1, 1, -1)  # Shape: (1, 1, len(f))

    for j in range(len(f_decidecade)):
        idx = np.argmin(np.abs(f_broadcast - f_decidecade_broadcast[:,:,j]), axis=-1)
        L_s_JE_decidecade[:, :, j] = L_s_f_JE[np.arange(L_s_f_JE.shape[0])[:, None], np.arange(L_s_f_JE.shape[1]), idx[:]] + 10 * np.log10(0.231 * f_decidecade[j])

    return L_s_JE_decidecade, f_decidecade

def plot_L_s_JE_decidecade(L_s_JE_decidecade, f_decidecade):
    # # Compute area under curve of each ele of L_s_JE_decidecade of shape (100,15,1000)
    # area = np.zeros((L_s_JE_decidecade.shape[0], L_s_JE_decidecade.shape[1]))
    # for i in range(area.shape[0]):  
    #     for j in range(area.shape[1]):
    #         area[i, j] = simps(L_s_JE_decidecade[i, j, :].reshape(-1), f_decidecade)


    # Reshape L_s_JE_decidecade from (100,15, len(deciband)) to (100*15, len(deciband))
    reshaped_L_s_JE = L_s_JE_decidecade.reshape((-1, f_decidecade.shape[0]))
    
    plt.figure(figsize=(10, 6))
    for i in range(reshaped_L_s_JE.shape[0]):
        plt.plot(f_decidecade, reshaped_L_s_JE[i, :], color='black')
    
    plt.xscale('log')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('L_{s,JE} (dB)')
    plt.title('Overlay of L_s_JE_decidecade across Different Velocity Matrices')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

def jomopans_model(f: np.ndarray, C: str, V: float, l: float):
        """
        f : Freq range
        C : Ship type
        V : Velocity in kn
        l : Ship length in feet
        """
        # Retrieve V_C for the given vessel class
        try:
                V_C = get_vessel_ref_speed(C)
        except ValueError as e:
                raise ValueError(str(e))

        l_0 = 300  # Reference length in feet

        # Compute L_s_f_JE
        L_s_f_JE = compute_L_s_f_JE(f, V, l, C, V_C, l_0)

        # Convert L_s_f_JE to decidecade bands
        L_s_JE_decidecade, f_decidecade = convert_to_decidecade(f, L_s_f_JE)

        if np.isneginf(L_s_JE_decidecade).any():
                L_s_JE_decidecade[:] = 0.0
        return L_s_JE_decidecade, f_decidecade
        

def jomopans_model_matrix(f: np.ndarray, C: str, V: np.ndarray, l: float):
        """
        f : Freq range
        C : Ship type
        V : Velocity Matrix in kn
        l : Ship length in feet
        """
        # Retrieve V_C for the given vessel class
        try:
                V_C = get_vessel_ref_speed(C)
        except ValueError as e:
                raise ValueError(str(e))

        l_0 = 300  # Reference length in feet

        # Compute L_s_f_JE
        L_s_f_JE = compute_L_s_f_JE_matrix(f, V, l, C, V_C, l_0)

        # Convert L_s_f_JE to decidecade bands
        L_s_JE_decidecade, f_decidecade = convert_to_decidecade_matrix(f, L_s_f_JE)

        if np.isneginf(L_s_JE_decidecade).any():
                L_s_JE_decidecade[:] = 0.0
        return L_s_JE_decidecade, f_decidecade

# def main():
#         f = np.logspace(1, 4, 1000)  # Frequency range from 10^1 to 10^4
#         C = 'Bulker'  # Class type

#         V = 11*np.ones((100,15))  # Actual vessel speed in knots
#         # V = np.random.uniform(11, 14, size=(100,15))

#         l = 692.257  # Length of the vessel in feet

#         L_s_JE_decidecade, f_decidecade = jomopans_model_matrix(f,C,V,l)
#         plot_L_s_JE_decidecade(L_s_JE_decidecade,f_decidecade)
# if __name__ == '__main__':
#     main()
