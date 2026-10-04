from pathlib import Path
from typing import Optional

import numpy as np
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
) -> mne.io.Raw:
    """
    Load EEG data for a single subject and return an MNE Raw object.

    This function assumes the .mat file contains a 'SIGNAL' array of shape
    (n_samples, 19) with the following columns:
        - column 0: time vector
        - columns 1-16: 16 EEG channels
        - columns 17-18: event metadata (eyes open/closed markers)

    Parameters
    ----------
    subject_id : str
        Subject identifier, e.g. "subject_01". The function expects a file
        named "{subject_id}.mat" in `data_dir`.
    data_dir : str | Path | None
        Path to the data directory. If None, uses the `data/` folder in the
        same directory as this module.
    ch_names : list[str] | None
        Optional list of channel names. If None, uses DEFAULT_CH_NAMES.
    sfreq : float | None
        Optional sampling frequency (Hz). If None, uses DEFAULT_SFREQ.
    signal_key : str
        Key in the .mat file that contains the EEG data array.

    Returns
    -------
    raw : mne.io.Raw
        MNE Raw object containing the EEG data (16 channels).
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

    return raw