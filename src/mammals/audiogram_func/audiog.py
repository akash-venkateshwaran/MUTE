import numpy as np

def audiogram_function(freq_Hz):
    """
    Computes the threshold (th) for an array of frequencies in Hz
    using the audiogram function from  Killer whale (Orcinus orca) behavioral audiograms JASA (2017).

    Parameters:
    freq_Hz (numpy array): Frequency in Hz

    Returns:
    numpy array: Threshold (th) in dB
    """
    # Parameters for the audiogram function
    T0 = 44.13
    F1 = 12.55  # in kHz
    F2 = 26.50  # in kHz
    A = 40.58
    B = 2.74

    # Convert frequency from Hz to kHz
    freq_kHz = freq_Hz / 1000

    # Calculate threshold
    th = T0 + A * np.log10(1 + (F1 / freq_kHz)) + (freq_kHz / F2) ** B

    return th
