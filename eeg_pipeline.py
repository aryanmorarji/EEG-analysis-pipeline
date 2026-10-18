from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import mne


# Default channel names for the 16-channel EEG cap
DEFAULT_CH_NAMES = [
    "FP1", "FP2",
    "FC5", "FC6",
    "FZ",
    "T7", "T8",
    "CZ",
    "P7", "P3", "PZ", "P4", "P8",
    "O1", "OZ", "O2"
]

DEFAULT_SFREQ = 512.0  # Hz

def load_subject(
    subject_id: str,
    data_dir: Optional[str | Path] = None,
    ch_names: Optional[list[str]] = None,
    sfreq: Optional[float] = None,
    signal_key: str = "SIGNAL",
) -> tuple[mne.io.Raw, np.ndarray, np.ndarray]:
    """
    Load EEG data and condition triggers for a single subject.

    This function assumes the .mat file contains a 'SIGNAL' array of shape
    (n_samples, 19) with the following columns:
        - column 0: time vector
        - columns 1-16: 16 EEG channels
        - column 17: eyes-closed trigger
        - column 18: eyes-open trigger

    Returns
    -------
    raw : mne.io.Raw
        MNE Raw object containing the 16 EEG channels.
    eyes_closed_trigger : np.ndarray
        1D trigger array marking eyes-closed blocks.
    eyes_open_trigger : np.ndarray
        1D trigger array marking eyes-open blocks.
    """
    if ch_names is None:
        ch_names = DEFAULT_CH_NAMES
    if sfreq is None:
        sfreq = DEFAULT_SFREQ

    if len(ch_names) != 16:
        raise ValueError(f"Expected 16 channel names, got {len(ch_names)}.")

    if data_dir is None:
        data_dir = Path(__file__).parent / "data"
    else:
        data_dir = Path(data_dir)

    mat_path = data_dir / f"{subject_id}.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Data file not found: {mat_path}")

    mat = sio.loadmat(str(mat_path))

    if signal_key not in mat:
        raise KeyError(
            f"Signal key '{signal_key}' not found in {mat_path}. "
            f"Available keys: {list(mat.keys())}"
        )

    signal = mat[signal_key]  # expected shape: [n_samples, 19]

    signal = np.asarray(signal)
    if signal.ndim != 2 or signal.shape[1] != 19:
        raise ValueError(
            f"Expected SIGNAL with shape (n_samples, 19), got {signal.shape}."
        )

    
    eeg_data = signal[:, 1:17]  # shape: [n_samples, 16]
    data = eeg_data.T  # shape: [16, n_samples]

    eyes_closed_trigger = np.asarray(signal[:, 17]).squeeze()
    eyes_open_trigger = np.asarray(signal[:, 18]).squeeze()

    if data.shape[0] != len(ch_names):
        raise ValueError(
            f"Number of channels in data ({data.shape[0]}) does not match "
            f"provided ch_names ({len(ch_names)})."
        )

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types="eeg",
    )

    raw = mne.io.RawArray(data, info, first_samp=0, copy="auto")

    return raw, eyes_closed_trigger, eyes_open_trigger

def inspect_and_mark_bad_channels(
    raw: mne.io.Raw,
    bad_channels: list[str] | None = None,
    show_plot: bool = True,
) -> mne.io.Raw:
    """Inspect EEG channels and optionally mark bad channels.

    Parameters
    ----------
    raw : mne.io.Raw
        EEG recording to inspect.
    bad_channels : list[str] | None
        Channel names to mark as bad. If None, keeps existing bad channels.
    show_plot : bool
        Whether to display an interactive raw-data plot for inspection.

    Returns
    -------
    mne.io.Raw
        Copy of the input with the selected channels marked bad.
    """
    raw = raw.copy()

    if bad_channels is not None:
        raw.info["bads"] = bad_channels

    if show_plot:
        raw.plot(
            n_channels=len(raw.ch_names),
            scalings="auto",
            block=True,
            title="Inspect EEG channels: click a channel name to mark/unmark bad",
        )

    print("Bad channels:", raw.info["bads"])
    return raw

def preprocess_raw(
    raw: mne.io.Raw,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    notch_freq: float | None = 50.0,
) -> mne.io.Raw:
    """Apply bandpass and optional mains-noise filtering to EEG data."""
    raw_filtered = raw.copy()
    raw_filtered.filter(l_freq=l_freq, h_freq=h_freq, picks="eeg", verbose=False)

    if notch_freq is not None:
        raw_filtered.notch_filter(freqs=notch_freq, picks="eeg", verbose=False)

    return raw_filtered

def extract_events(
    eyes_closed_trigger: np.ndarray,
    eyes_open_trigger: np.ndarray,
) -> np.ndarray:
    """
    Create an MNE-compatible events array from condition trigger arrays.

    Parameters
    ----------
    eyes_closed_trigger : np.ndarray
        1D array where non-zero samples mark the onset of an eyes-closed block.
    eyes_open_trigger : np.ndarray
        1D array where non-zero samples mark the onset of an eyes-open block.

    Returns
    -------
    events : np.ndarray, shape (n_events, 3)
        MNE events array. Columns are:
            0: onset sample index
            1: previous trigger value
            2: event ID (1 = eyes closed, 2 = eyes open)
    """
    closed_onsets = np.flatnonzero(eyes_closed_trigger)
    open_onsets = np.flatnonzero(eyes_open_trigger)

    if len(closed_onsets) == 0 or len(open_onsets) == 0:
        raise ValueError("No event onsets found in one or both trigger arrays.")

    closed_events = np.column_stack((
        closed_onsets,
        np.zeros(len(closed_onsets), dtype=int),
        np.ones(len(closed_onsets), dtype=int),
    ))

    open_events = np.column_stack((
        open_onsets,
        np.zeros(len(open_onsets), dtype=int),
        np.full(len(open_onsets), 2, dtype=int),
    ))

    events = np.vstack((closed_events, open_events))

    # MNE expects events ordered by sample index.
    return events[np.argsort(events[:, 0])]

def make_epochs(
    raw: mne.io.Raw,
    events: np.ndarray,
    event_id: dict[str, int],
    epoch_length: float = 10.0,
) -> mne.Epochs:
    """Create equal-length EEG epochs from event onsets."""
    return mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=0.0,
        tmax=epoch_length,
        baseline=None,
        preload=True,
        verbose=False,
    )

def compute_alpha_power(
    epochs: mne.Epochs,
    alpha_low: float = 8.0,
    alpha_high: float = 12.0,
    picks: str | list[str] = "eeg",
) -> dict[str, float]:
    """
    Compute mean alpha-band power for each condition.

    Returns
    -------
    dict[str, float]
        Mean alpha power for each condition.
    """
    alpha_power = {}

    for condition in epochs.event_id:
        condition_epochs = epochs[condition]

        spectrum = condition_epochs.compute_psd(
            method="welch",
            fmin=alpha_low,
            fmax=alpha_high,
            picks=picks,
            verbose=False,
        )

        psds, freqs = spectrum.get_data(return_freqs=True)

        # psds shape: (n_epochs, n_channels, n_freqs)
        alpha_power[condition] = float(np.mean(psds))

    return alpha_power

def plot_alpha_power_comparison(
    alpha_power: dict[str, float],
    title: str = "Mean alpha-band power",
) -> plt.Figure:
    """Plot mean alpha power for each experimental condition."""
    conditions = list(alpha_power.keys())
    values = list(alpha_power.values())

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(conditions, values, color=["#4C72B0", "#DD8452"])

    ax.set_title(title)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Mean alpha power (V²/Hz)")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    plt.show()

    return fig

def plot_condition_psds(
    epochs: mne.Epochs,
    picks: str | list[str] = "eeg",
    fmin: float = 1.0,
    fmax: float = 40.0,
    alpha_low: float = 8.0,
    alpha_high: float = 12.0,
) -> plt.Figure:
    """Plot mean PSDs for eyes-closed and eyes-open on shared axes."""
    fig, ax = plt.subplots(figsize=(9, 5))

    colours = {
        "eyes_closed": "#4C72B0",
        "eyes_open": "#DD8452",
    }

    for condition in epochs.event_id:
        spectrum = epochs[condition].compute_psd(
            method="welch",
            fmin=fmin,
            fmax=fmax,
            picks=picks,
            verbose=False,
        )

        psds, freqs = spectrum.get_data(return_freqs=True)

        # PSD shape: (n_epochs, n_channels, n_frequency_bins).
        mean_psd = psds.mean(axis=(0, 1))

        # Convert V²/Hz to dB for an interpretable PSD scale.
        mean_psd_db = 10 * np.log10(mean_psd)

        ax.plot(
            freqs,
            mean_psd_db,
            label=condition.replace("_", " ").title(),
            color=colours.get(condition),
            linewidth=2,
        )

    ax.axvspan(
        alpha_low,
        alpha_high,
        color="gold",
        alpha=0.18,
        label="Alpha band (8–12 Hz)",
    )

    ax.set_title("Average power spectral density by condition")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB re 1 V²/Hz)")
    ax.set_xlim(fmin, fmax)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    return fig

def run_full_pipeline(
    subject_id: str,
    *,
    data_dir: str | Path | None = None,
    bad_channels: list[str] | None = None,
    inspect_channels: bool = False,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
    notch_freq: float | None = 50.0,
    epoch_length: float = 10.0,
    alpha_low: float = 8.0,
    alpha_high: float = 12.0,
    posterior_channels: list[str] | None = None,
) -> dict:
    """Run the complete EEG preprocessing and alpha-power pipeline.

    Returns the processed data and alpha-power summaries for one subject.
    """
    if posterior_channels is None:
        posterior_channels = ["P7", "P3", "PZ", "P4", "P8", "O1", "OZ", "O2"]

    # Load continuous 16-channel EEG and its two trigger arrays.
    raw, closed_trigger, open_trigger = load_subject(
        subject_id,
        data_dir=data_dir,
    )

    # Optional manual visual review. Any channels selected in the viewer
    # will be stored in raw.info["bads"].
    raw_checked = inspect_and_mark_bad_channels(
        raw,
        bad_channels=bad_channels,
        show_plot=inspect_channels,
    )

    # Filter EEG.
    raw_clean = preprocess_raw(
        raw_checked,
        l_freq=l_freq,
        h_freq=h_freq,
        notch_freq=notch_freq,
    )

    # Create MNE events, then 10-second condition-labelled epochs.
    events = extract_events(closed_trigger, open_trigger)

    event_id = {
        "eyes_closed": 1,
        "eyes_open": 2,
    }

    epochs = make_epochs(
        raw_clean,
        events,
        event_id=event_id,
        epoch_length=epoch_length,
    )

    # Compute condition summaries using all retained EEG channels and
    # the posterior subset.
    alpha_power_all = compute_alpha_power(
        epochs,
        alpha_low=alpha_low,
        alpha_high=alpha_high,
        picks="eeg",
    )

    available_posterior = [
        channel
        for channel in posterior_channels
        if channel in epochs.ch_names
        and channel not in epochs.info["bads"]
    ]

    if not available_posterior:
        raise ValueError("No requested posterior channels are available for analysis.")

    alpha_power_posterior = compute_alpha_power(
        epochs,
        alpha_low=alpha_low,
        alpha_high=alpha_high,
        picks=available_posterior,
    )

    return {
        "raw": raw,
        "raw_checked": raw_checked,
        "raw_clean": raw_clean,
        "events": events,
        "epochs": epochs,
        "alpha_power_all": alpha_power_all,
        "alpha_power_posterior": alpha_power_posterior,
        "posterior_channels": available_posterior,
    }