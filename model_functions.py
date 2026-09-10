#! /bin/env/ python

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
import time
import sys

# before defining functions, instantiate plotting font structure.
font = FontProperties()
font.set_name("serif")
font.set_size(15)

# also, define some constants here.
T_sun = 4.925490947e-6  # --> in units of seconds

# also, set data type for spectrum.
dtype = np.float16

def compute_A1_from_mass_function(period_orbit: float, mass_pulsar: float = 1.4, 
    mass_companion: float = 1.4, angle_incl: float = 90.):

    """
    Computes the value of projected semi-major axis for the pulsar's orbit.

    Parameters
    ----------
    period_orbit : float
        Orbital period of binary orbit, in units of days.

    mass_pulsar : float, optional
        Gravitational mass of pulsar, in units of solar masses.

    mass_companion : float, optional
        Gravitational mass of companion star, in units of solar masses.

    angle_incl : float, optional
        Orbital inclination angle, in units of degrees.

    Returns
    -------
    axis_semi_pulsar : float
        Projected semimajor axis of the pulsar's orbit, in units of (light-)seconds.
    """

    # first, compute Keplerian mass function for pulsar's orbit.
    sini = np.sin(angle_incl * np.pi / 180)
    mass_total = mass_pulsar + mass_companion
    mass_function = (mass_companion * sini) ** 3 / mass_total ** 2

    # now use it to compute the value of A1 given an orbital period.
    pb = period_orbit * 86400
    axis_semi_pulsar = (mass_function * T_sun * pb ** 2 / 4 / np.pi ** 2) ** (1./3)
    
    return axis_semi_pulsar

def compute_apparent_frequency(period_spin: float, period_orbit: float, 
    time_length: float, mass_pulsar: float = 1.4, mass_companion: float = 1.4, 
    angle_incl: float = 90., time_orbit_asc: float = 57000.):
    """
    Computes the expected/apparent value of spin frequency for a binary pulsar 
    evaluated at the center of a timeseries of length equal to 'time_length.'
    """
    
    # compute quantities related to orbit.
    nb = 2 * np.pi / (period_orbit * 86400.)
    axis_semi_pulsar = compute_A1_from_mass_function(
        period_orbit, mass_pulsar=mass_pulsar, mass_companion=mass_companion, angle_incl=angle_incl
    )
    delta = (nb * 86400) * (time_length / 86400 - time_orbit_asc) % (2 * np.pi)
    sind = np.sin(delta)
    cosd = np.cos(delta)

    # now compute factors that adjusts the intrinsic spin frequency.
    term1 = nb * axis_semi_pulsar * cosd
    term2 = -nb ** 2 * axis_semi_pulsar * sind * (time_length / 2)
    
    # finally, compute apparent spin frequency and return.
    freq_int = 1 / period_spin
    freq_app = freq_int * (1 + term1 + term2)

    return freq_app

def compute_apparent_acceleration(period_spin: float, period_orbit: float, 
    time_length: float, mass_pulsar: float = 1.4, mass_companion: float = 1.4, 
    angle_incl: float = 90., time_orbit_asc: float = 57000.):
    """
    Computes the expected/apparent value of spin frequency for a binary pulsar 
    evaluated at the center of a timeseries of length equal to 'time_length.'
    """
    
    # compute quantities related to orbit.
    nb = 2 * np.pi / (period_orbit * 86400.)
    axis_semi_pulsar = compute_A1_from_mass_function(
        period_orbit, mass_pulsar=mass_pulsar, mass_companion=mass_companion, angle_incl=angle_incl
    )
    delta = (nb * 86400) * (time_length / 86400 - time_orbit_asc) % (2 * np.pi)
    sind = np.sin(delta)
    cosd = np.cos(delta)

    # now compute the acceleration term.
    freq_int = 1 / period_spin
    accel = -2 * nb ** 2 * axis_semi_pulsar * freq_int * sind

    return accel

def compute_phase_mean(phase_ref: float, period_spin: float, period_orbit: float,
    time_length: float, mass_pulsar: float = 1.4, mass_companion: float = 1.4,
    angle_incl: float = 90., time_orbit_asc: float = 57000.):
    
    
    # now compute the acceleration term.
    freq_int = 1 / period_spin
    freq_app = compute_apparent_frequency(
        period_spin, period_orbit, time_length, mass_pulsar=mass_pulsar, 
        mass_companion=mass_companion, angle_incl=angle_incl, time_orbit_asc=time_orbit_asc
    )
    accel_app = -0.5 * compute_apparent_acceleration(
        period_spin, period_orbit, time_length, mass_pulsar=mass_pulsar, 
        mass_companion=mass_companion, angle_incl=angle_incl, time_orbit_asc=time_orbit_asc
    )
    term1 = freq_app * (time_length / 2)
    term2 = accel_app * (time_length / 2) ** 2
    phase_mean = phase_ref + term1 + term2

    return phase_mean

def compute_phase_pulse_expected(time: np.ndarray, phase_ref: float, period_spin: float, 
    period_orbit: float, mass_pulsar: float = 1.4, mass_companion: float = 1.4, 
    angle_incl: float = 90., time_orbit_asc: float = 57000.):
    """
    Computes the pulse phase of the pulsar (eq. 17 in K. Smith [2016]).

    Parameters
    ----------
    time : array_like
        An array of time values at which to evaluate pulse phase.

    phase_ref : float
        'Intrinsic' value of reference phase for pulsar.

    period_spin: float
        Spin period of pulsar, in units of seconds.

    period_orbit : float
        Orbital period of binary orbit, in units of days.

    mass_pulsar : float, optional
        Gravitational mass of pulsar, in units of solar masses.

    mass_companion : float, optional
        Gravitational mass of companion star, in units of solar masses.

    angle_incl : float, optional
        Orbital inclination angle, in units of degrees.

    time_orbit_asc : float, optional.
        Time of passage by pulsar through ascending-node longitude, in units of MJD.

    Returns
    -------
    phase_pulse : array_like
        Pulse phases for a binary pulsar, normalized to unity.
    """

    # compute the total length of timeseries.
    time_length = time[-1] - time[0]
    dt = time - (time_length / 2)

    # compute the polynomial  terms.
    term1 = compute_phase_mean(
        phase_ref, period_spin, period_orbit, time_length, mass_pulsar=mass_pulsar,
        mass_companion=mass_companion, angle_incl=angle_incl, time_orbit_asc=time_orbit_asc
    )
    term2 = compute_apparent_frequency(
        period_spin, period_orbit, time_length, mass_pulsar=mass_pulsar, 
        mass_companion=mass_companion, angle_incl=angle_incl, time_orbit_asc=time_orbit_asc
    ) * dt
    term3 = 0.5 * compute_apparent_acceleration(
        period_spin, period_orbit, time_length, mass_pulsar=mass_pulsar,
        mass_companion=mass_companion, angle_incl=angle_incl, time_orbit_asc=time_orbit_asc
    ) * (dt ** 2 - time_length ** 2 / 12)

    print("term1:", term1)
    print("term2:", term2)
    print("term3:", term3)

    # compute the phase and return.
    phase_pulse = (term1 + term2 + term3) % 1.

    return phase_pulse

def compute_phase_pulse_generic(time: float, phase_ref: float, frequency: float, 
    acceleration: float = 0., time_ref: float = 0.):
    """
    Computes the pulse phase of the pulsar. 
    """

    # compute the polynomial  terms.
    dt = time - time_ref
    phase_pulse = phase_ref
    phase_pulse += frequency * dt
    phase_pulse += 0.5 * acceleration * dt**2

    # return the value(s).
    phase_pulse %= 1.
    return phase_pulse

def compute_profile_vonMises(phase_pulse: float, duty_cycle: float = 0.1):
    """
    Computes a von Mises profile, as defined in Equation 2 of 
    K. Smith (2016, arXiv:1610.06831).
    """

    # compute concentration parameter.
    kappa = np.log(2) / 2 / np.sin(np.pi * duty_cycle / 2, dtype=dtype)**2

    # now compute profile, noting that pulse phase is defined differently.
    # (it runs from 0 to 1 here).
    profile = np.exp(-2 * kappa * np.sin(np.pi * phase_pulse)**2, dtype=dtype)

    # and return.
    return profile

def compute_statistic_overlap(times: float, phase_ref: float, frequency: float, 
    profile_data: dtype, duty_cycle: float = 0.1):
    """
    Computes the overlap integral described on pg. 2 of K. Smith 
    (2016, arXiv:1610.06831).
    """

    phases_model = compute_phase_pulse(times, phase_ref, frequency)
    profile_model = compute_profile_vonMises(phases_model, duty_cycle=duty_cycle)
    overlap = np.sum(profile_model * profile_data)

    return overlap

def compute_statistic_snr_peak(profile_data: float, region_off: float = [0.3, 0.7]):
    """
    Computes the peak S/N of a pulse.
    """

    num = len(profile_data)
    peak = profile_data.max()
    idx_off_start = int(num * region_off[0])
    idx_off_stop = int(num * region_off[1])
    rms = np.sqrt(np.mean(profile_data[idx_off_start : idx_off_stop] ** 2))
    snr = peak / rms

    return snr

def plot_time_phase(times: float, phase_ref: float, frequency: float,
    profile_data: float, acceleration: float = 0., num_bins: int = 32):

    # compute pulse phases and bin edges for downsampled profile.
    phase_pulse = compute_phase_pulse(times, phase_ref, frequency, acceleration=acceleration)
    phase_pulse_bins = np.linspace(0., 1., num=num_bins+1)    
    phase_pulse_idxs = np.digitize(phase_pulse, phase_pulse_bins)

    # determine the index of the beginning of the first full cycle, 
    # as well as the index of the end of the last full cycle. 
    idx_min = phase_pulse_idxs.argmin()
    idx_max = (len(phase_pulse) - 1) - np.flip(phase_pulse_idxs).argmax()

    # loop over all full cycles of the profile data.
    bin_counts = 0 
    profile_binned_all = []
    profile_binned = np.zeros(num_bins) 

    for current_idx in range(idx_min, idx_max + 1):
        phase_pulse_current = phase_pulse[current_idx]
        profile_binned_idx_current = phase_pulse_idxs[current_idx] - 1
        profile_binned[profile_binned_idx_current] += profile_data[current_idx]
        
        try:

            # first, check if next iteration will be in different bin.
            # if so, then divide current bin value by number of counts 
            # (to obtain an averaged profile value in the bin).
            profile_binned_idx_next = phase_pulse_idxs[current_idx + 1]
            diff_bin_idx = profile_binned_idx_next - profile_binned_idx_current  

            if diff_bin_idx != 0:
                profile_binned[profile_binned_idx_current] /= (bin_counts + 1)
                bin_counts = 0

            # next, if the next phase wraps to the beginning of the pulse 
            # phase, then break out of this loop.
            phase_pulse_next = phase_pulse[current_idx + 1]
            diff_phase_pulse = phase_pulse_next - phase_pulse_current

            if diff_phase_pulse < 0.:
                profile_binned_all.append(profile_binned.tolist())
                profile_binned[:] = 0.

        except Exception as exc:
            print(exc)

        bin_counts += 1

    return np.array(profile_binned_all)

if __name__ == "__main__":

    # try out some test values and ranges.
    # (physical values are for B1534+12 from the ATNF catalog.)
    m1 = 2.08
    m2 = 0.253
    pb = 4.7669446191
    f0 = 346.5319964932129
    incl = 87.56
    tasc = 57552.08324415
    phase_ref = 0.32
    times = np.linspace(0., 3600., num=2 ** 18)

    # now try out some of the above functions.
    a1 = compute_A1_from_mass_function(pb, m1, m2, incl)
    print("value of A1 from assumed values:", a1)
    print("... this should be close to 3.9775 lt-s")

    phases = compute_phase_pulse_expected(times, phase_ref, 1/f0, pb, m1, m2, incl, tasc)
    print("expected phases:", phases)
    profile_data = compute_profile_vonMises(phases)
    plt.plot(times, profile_data)
    plt.show()
