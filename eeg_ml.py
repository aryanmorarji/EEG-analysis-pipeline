import mne
import pandas as pd


def build_epoch_metadata(
    epochs_by_subject: dict[str, mne.Epochs],
) -> pd.DataFrame:
    """Return one metadata row per retained epoch, grouped by subject."""
    rows = []

    for subject_id in sorted(epochs_by_subject):
        epochs = epochs_by_subject[subject_id]

        label_mapping = {
            epochs.event_id["eyes_open"]: 0,
            epochs.event_id["eyes_closed"]: 1,
        }

        for epoch_index, event in enumerate(epochs.events):
            event_code = int(event[2])

            if event_code not in label_mapping:
                raise ValueError(
                    f"Unexpected event code {event_code} in {subject_id}."
                )

            rows.append({
                "subject_id": subject_id,
                "epoch_index": epoch_index,
                "onset_sample": int(event[0]),
                "label": label_mapping[event_code],
            })

    return pd.DataFrame(
        rows,
        columns=["subject_id", "epoch_index", "onset_sample", "label"],
    )