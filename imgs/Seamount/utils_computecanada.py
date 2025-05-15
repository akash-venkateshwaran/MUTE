import numpy as np
from scipy.io import loadmat  
import os


def compute_tl_limit(tl_slice):
    """
    Function to compute the minimum and maximum TL limits based on the
    standard deviation of the data.

    Parameters:
        tl_slice - The 2D slice of TL data for a specific depth level

    Returns:
        tl_min - The computed minimum limit for TL
        tl_max - The computed maximum limit for TL
    """
    tl_data = tl_slice.flatten()
    tl_data = tl_data[tl_data <= 700]
    
    tl_mean = np.mean(tl_data)
    tl_std = np.std(tl_data)
    
    tl_min = tl_mean - 2 * tl_std
    tl_max = tl_mean + 2 * tl_std
    
    return tl_min, tl_max


def extract_tl(filename, units='km'):
    """
    Extract the transmission loss (TL) as a 3D matrix from the SHD file
    and convert cylindrical coordinates to Cartesian coordinates.

    Parameters:
        filename - The SHD file to read.
        units    - The units for distance ('m' or 'km').

    Returns:
        tlt_output - The 3D matrix of transmission loss (z * x * y).
        x          - The x-coordinates in Cartesian coordinates.
        y          - The y-coordinates in Cartesian coordinates.
        z          - The z-coordinates (depth).
        freq       - Frequency of the SHD file.
        Pos        - Position information from SHD file.
    """

    if units not in ['m', 'km']:
        raise ValueError("Invalid units. Must be 'm' or 'km'.")

    _, _, freq, _, _, Pos, pressure = read_shd(filename)

    nrz = Pos['r.z'].shape[0]
    tlt = np.abs(pressure[:, 0, :, :])
    tlt = np.transpose(tlt, (1, 0, 2))

    tlt = np.reshape(tlt, (nrz, -1, tlt.shape[2]))
    tlt[np.isnan(tlt)] = 1e-6
    tlt[np.isinf(tlt)] = 1e-6
    tlt[tlt < 1e-37] = 1e-37

    tlt = -20.0 * np.log10(tlt)

    ntheta = Pos['theta'].shape[0]
    d_theta = (Pos['theta'][-1] - Pos['theta'][0]) / (ntheta - 1)

    tlt_output = []
    for i in range(tlt.shape[0]):
        tl = np.squeeze(tlt[i, :, :])
        if (np.mod(Pos['theta'][-1] + d_theta - Pos['theta'][0] + .001, 360.0) < .002):
            if i == tlt.shape[0] - 1:
                Pos['theta'] = np.append(Pos['theta'], Pos['theta'][-1] + d_theta)
            tl = np.vstack((tl, tl[0, :]))
        tl = tl.T
        tlt_output.append(tl)
    tlt_output = np.array(tlt_output)

    th, r = np.meshgrid(Pos['theta'], Pos['r.r'])
    th = (2 * np.pi / 360.0) * th
    x, y = pol2cart(th, r)

    if units == 'km':
        x /= 1000
        y /= 1000

    z = Pos['r.z']

    return tlt_output, x, y, z, freq, Pos



def read_shd(*args):
    if len(args) < 1 or len(args) > 3:
        raise ValueError("Invalid number of arguments")

    filename = args[0]
    freq = args[1] if len(args) >= 2 else np.nan
    xs = args[1] if len(args) == 3 else np.nan
    ys = args[2] if len(args) == 3 else np.nan

    PlotType = None

    path, ext = os.path.splitext(filename)
    name = os.path.basename(path)

    if ext == '.mat':
        _, name2, ext2 = os.path.splitext(name)
        if ext2 == '.shd':
            FileType = 'shdmat'
        elif ext2 == '.grn':
            FileType = 'grnmat'
    else:
        if filename == 'ASCFIL':
            FileType = 'asc'
        elif filename == 'SHDFIL':
            FileType = 'shd'
        elif filename == 'tl.grid':
            FileType = 'RAM'
        else:
            endchar = len(filename)
            if endchar >= 4:
                FileType = filename[-3:].lower()

    if FileType in ['shd', 'grn']:
        if len(args) == 1:
            PlotTitle, PlotType, freqVec, freq0, atten, Pos, pressure = read_shd_bin(filename)
        elif len(args) == 2:
            PlotTitle, PlotType, freqVec, freq0, atten, Pos, pressure = read_shd_bin(filename, freq)
        elif len(args) == 3:
            PlotTitle, PlotType, freqVec, freq0, atten, Pos, pressure = read_shd_bin(filename, xs, ys)
    elif FileType == 'shdmat':
        data = loadmat(filename)
        PlotTitle = data.get('PlotTitle')
        PlotType = data.get('PlotType')
        freqVec = data.get('freqVec')
        freq0 = data.get('freq0')
        atten = data.get('atten')
        Pos = data.get('Pos')
        pressure = data.get('pressure')

        if not np.isnan(xs):
            xdiff = np.abs(Pos['s']['x'][0] - xs * 1000)
            idxX = np.argmin(xdiff)
            ydiff = np.abs(Pos['s']['y'][0] - ys * 1000)
            idxY = np.argmin(ydiff)
            pressureT = pressure[idxX, idxY, :, :, :, :]
            pressure = np.reshape(pressureT, [Pos['Nsz'], Pos['Nrz'], Pos['Ntheta'], Pos['Nrr']])

        if not np.isnan(freq):
            freqdiff = np.abs(freqVec - freq)
            ifreq = np.argmin(freqdiff)
            pressure = pressure[ifreq, 0, :, :]
    elif FileType == 'asc':
        PlotTitle, PlotType, freqVec, freq0, atten, Pos, pressure = read_shd_asc(filename)
    elif FileType == 'grnmat':
        data = loadmat(filename)
        PlotTitle = data.get('PlotTitle')
        PlotType = data.get('PlotType')
        freqVec = data.get('freqVec')
        freq0 = data.get('freq0')
        atten = data.get('atten')
        Pos = data.get('Pos')
        pressure = data.get('pressure')
        Pos['r']['range'] = Pos['r']['range'].T

        if not np.isnan(freq):
            freqdiff = np.abs(freqVec - freq)
            ifreq = np.argmin(freqdiff)
            pressure = pressure[ifreq, 0, :, :]
    elif FileType == 'RAM':
        PlotTitle, PlotType, freqVec, freq0, atten, Pos, pressure = read_ram_tlgrid()
    else:
        raise ValueError('Unrecognized file extension')

    return PlotTitle, PlotType, freqVec, freq0, atten, Pos, pressure

def read_shd_asc(filename):
    with open(filename, 'r') as f:
        PlotTitle = f.readline().strip()
        PlotType = f.readline().strip()
        Nfreq = int(f.readline().strip())
        Ntheta = int(f.readline().strip())
        Nsd = int(f.readline().strip())
        Nrd = int(f.readline().strip())
        Nrr = int(f.readline().strip())
        freq0 = float(f.readline().strip())
        atten = float(f.readline().strip())

        freqVec = np.array([float(f.readline().strip()) for _ in range(Nfreq)])
        Pos = {
            'theta': np.array([float(f.readline().strip()) for _ in range(Ntheta)]),
            's.depth': np.array([float(f.readline().strip()) for _ in range(Nsd)]),
            'r.depth': np.array([float(f.readline().strip()) for _ in range(Nrd)]),
            'r.range': np.array([float(f.readline().strip()) for _ in range(Nrr)])
        }

        temp1 = np.array([float(f.readline().strip()) for _ in range(2 * Nrr * Nrd)])
        temp1 = temp1.reshape((2 * Nrr, Nrd))

    pressure = temp1[::2, :].T + 1j * temp1[1::2, :].T

    return PlotTitle, PlotType, freqVec, freq0, atten, Pos, pressure




def read_shd_bin(filename, *args):
    freq = args[0] if len(args) == 2 else np.nan
    xs = args[0] if len(args) == 3 else np.nan
    ys = args[1] if len(args) == 3 else np.nan

    with open(filename, 'rb') as f:
        recl = np.fromfile(f, dtype=np.int32, count=1)[0]
        title = f.read(80).decode('utf-8').strip()

        f.seek(4 * recl, 0)
        PlotType = f.read(10).decode('utf-8').strip()

        f.seek(2 * 4 * recl, 0)
        Nfreq = np.fromfile(f, dtype=np.int32, count=1)[0]
        Ntheta = np.fromfile(f, dtype=np.int32, count=1)[0]
        Nsx = np.fromfile(f, dtype=np.int32, count=1)[0]
        Nsy = np.fromfile(f, dtype=np.int32, count=1)[0]
        Nsz = np.fromfile(f, dtype=np.int32, count=1)[0]
        Nrz = np.fromfile(f, dtype=np.int32, count=1)[0]
        Nrr = np.fromfile(f, dtype=np.int32, count=1)[0]
        freq0 = np.fromfile(f, dtype=np.float32, count=1)[0]
        atten = np.fromfile(f, dtype=np.float32, count=1)[0]

        f.seek(3 * 4 * recl, 0)
        freqVec = np.fromfile(f, dtype=np.float64, count=Nfreq)

        f.seek(4 * 4 * recl, 0)
        Pos = {'theta': np.fromfile(f, dtype=np.float32, count=Ntheta)}

        if PlotType[:2] != 'TL':
            f.seek(5 * 4 * recl, 0)
            Pos['s.x'] = np.fromfile(f, dtype=np.float32, count=Nsx)

            f.seek(6 * 4 * recl, 0)
            Pos['s.y'] = np.fromfile(f, dtype=np.float32, count=Nsy)
        else:
            f.seek(5 * 4 * recl, 0)
            Pos['s.x'] = np.linspace(*np.fromfile(f, dtype=np.float32, count=2), Nsx)

            f.seek(6 * 4 * recl, 0)
            Pos['s.y'] = np.linspace(*np.fromfile(f, dtype=np.float32, count=2), Nsy)

        f.seek(7 * 4 * recl, 0)
        Pos['s.z'] = np.fromfile(f, dtype=np.float32, count=Nsz)

        f.seek(8 * 4 * recl, 0)
        Pos['r.z'] = np.fromfile(f, dtype=np.float32, count=Nrz)

        f.seek(9 * 4 * recl, 0)
        Pos['r.r'] = np.fromfile(f, dtype=np.float32, count=Nrr)

        if PlotType == 'rectilin  ':
            pressure = np.zeros((Ntheta, Nsz, Nrz, Nrr), dtype=np.complex64)
            Nrcvrs_per_range = Nrz
        elif PlotType == 'irregular ':
            pressure = np.zeros((Ntheta, Nsz, 1, Nrr), dtype=np.complex64)
            Nrcvrs_per_range = 1
        else:
            pressure = np.zeros((Ntheta, Nsz, Nrz, Nrr), dtype=np.complex64)
            Nrcvrs_per_range = Nrz

        if np.isnan(xs):
            ifreq = 1
            if not np.isnan(freq):
                freqdiff = np.abs(freqVec - freq)
                ifreq = np.argmin(freqdiff) + 1

            for itheta in range(Ntheta):
                for isz in range(Nsz):
                    for irz in range(Nrcvrs_per_range):
                        recnum = (10 + (ifreq - 1) * Ntheta * Nsz * Nrcvrs_per_range
                                  + itheta * Nsz * Nrcvrs_per_range + isz * Nrcvrs_per_range + irz)

                        f.seek(recnum * 4 * recl, 0)
                        temp = np.fromfile(f, dtype=np.float32, count=2 * Nrr)
                        pressure[itheta, isz, irz, :] = temp[::2] + 1j * temp[1::2]
        else:
            xdiff = np.abs(Pos['s.x'] - xs * 1000)
            idxX = np.argmin(xdiff)
            ydiff = np.abs(Pos['s.y'] - ys * 1000)
            idxY = np.argmin(ydiff)

            for itheta in range(Ntheta):
                for isz in range(Nsz):
                    for irz in range(Nrcvrs_per_range):
                        recnum = (10 + (idxX - 1) * Nsy * Ntheta * Nsz * Nrcvrs_per_range
                                  + (idxY - 1) * Ntheta * Nsz * Nrcvrs_per_range
                                  + itheta * Nsz * Nrcvrs_per_range + isz * Nrcvrs_per_range + irz)

                        f.seek(recnum * 4 * recl, 0)
                        temp = np.fromfile(f, dtype=np.float32, count=2 * Nrr)
                        pressure[itheta, isz, irz, :] = temp[::2] + 1j * temp[1::2]

    return title, PlotType, freqVec, freq0, atten, Pos, pressure


import numpy as np

def read_ram_tlgrid():
    filename = 'tl.grid'

    try:
        fid = open(filename, 'rb')
    except FileNotFoundError:
        raise FileNotFoundError(f"Unable to open {filename}")

    # First guess: unsigned 8-byte integer
    rlen = np.fromfile(fid, dtype=np.uint64, count=1)[0]

    # Check the value read against the known value
    if rlen == 60:
        rm_datatype = np.uint64
    else:
        # Try unsigned 4-byte integer
        fid.seek(0)
        rlen = np.fromfile(fid, dtype=np.uint32, count=1)[0]
        if rlen == 60:
            rm_datatype = np.uint32
        else:
            fid.close()
            raise ValueError("Error while determining datatype of record markers")

    # Continue reading the header containing the problem description
    freq = np.fromfile(fid, dtype=np.float32, count=1)[0]
    zs = np.fromfile(fid, dtype=np.float32, count=1)[0]
    zr = np.fromfile(fid, dtype=np.float32, count=1)[0]
    rmax = np.fromfile(fid, dtype=np.float32, count=1)[0]
    dr = np.fromfile(fid, dtype=np.float32, count=1)[0]
    ndr = np.fromfile(fid, dtype=np.int32, count=1)[0]
    zmax = np.fromfile(fid, dtype=np.float32, count=1)[0]
    dz = np.fromfile(fid, dtype=np.float32, count=1)[0]
    ndz = np.fromfile(fid, dtype=np.int32, count=1)[0]
    zmplt = np.fromfile(fid, dtype=np.float32, count=1)[0]
    c0 = np.fromfile(fid, dtype=np.float32, count=1)[0]
    np_val = np.fromfile(fid, dtype=np.int32, count=1)[0]
    ns = np.fromfile(fid, dtype=np.int32, count=1)[0]
    rs = np.fromfile(fid, dtype=np.float32, count=1)[0]
    lz = np.fromfile(fid, dtype=np.int32, count=1)[0]
    rlen = np.fromfile(fid, dtype=rm_datatype, count=1)[0]  # Record markers

    # Calculate `lr`
    lr = int(np.floor(rmax / (dr * ndr)))

    # Read the TL matrix
    TL = np.zeros((lz, lr))

    for j in range(lr):
        rlen = np.fromfile(fid, dtype=rm_datatype, count=1)[0]  # Record markers
        TL[:, j] = np.fromfile(fid, dtype=np.float32, count=lz)
        rlen = np.fromfile(fid, dtype=rm_datatype, count=1)[0]  # Record markers

    fid.close()

    # Convert to standard Acoustics Toolbox variable names
    PlotTitle = 'RAM'
    PlotType = 'rectilin'
    atten = 0

    Pos = {
        's.z': zs,
        'r.z': np.arange(1, lz + 1) * ndz * dz,
        'r.range': np.arange(1, lr + 1) * ndr * dr
    }

    pressure = np.zeros((1, 1, lz, lr))
    pressure[0, 0, :, :] = 10 ** (-TL / 20)

    return PlotTitle, PlotType, freq, atten, Pos, pressure



def pol2cart(th, r, z=None):
    x = r * np.cos(th)
    y = r * np.sin(th)
    return (x, y, z) if z is not None else (x, y)