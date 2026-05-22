"""One-time setup: fetch fsaverage BEM assets and write fsaverage-fwd.fif."""
from pathlib import Path

import mne

PREP_ROOT = Path(__file__).resolve().parent
SUBJECTS_DIR = PREP_ROOT / "subjects_dir_fsaverage"
FORWARD_PATH = SUBJECTS_DIR / "fsaverage" / "fsaverage-fwd.fif"
RAW_EEGLAB = (
    PREP_ROOT
    / "data_epoched"
    / "raw_eeglab_and_block_idents"
    / "Tuebingen"
    / "sub-018"
    / "sub-018_task-tep_all_eeg.set"
)

COMMON_CHANNELS = [
    "AF3", "AF4", "AF7", "AF8", "C1", "C2", "C3", "C4", "C5", "C6",
    "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "CPz", "Cz", "F1", "F2",
    "F3", "F4", "F5", "F6", "F7", "F8", "FC1", "FC2", "FC3", "FC4",
    "FC5", "FC6", "FT7", "FT8", "Fp1", "Fp2", "Fpz", "Fz", "Iz", "O1",
    "O2", "Oz", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "PO3",
    "PO4", "PO7", "PO8", "POz", "Pz", "T7", "T8", "TP7", "TP8",
]


def main():
    if FORWARD_PATH.exists():
        print(f"Forward solution already exists: {FORWARD_PATH}")
        return

    if not RAW_EEGLAB.exists():
        raise FileNotFoundError(
            f"EEGLAB epochs not found at {RAW_EEGLAB}. "
            "Link sub-018 under preprocessing/data_epoched/raw_eeglab_and_block_idents/Tuebingen/ first."
        )

    mne.datasets.fetch_fsaverage(subjects_dir=str(SUBJECTS_DIR), verbose=True)

    bem_dir = SUBJECTS_DIR / "fsaverage" / "bem"
    trans = bem_dir / "fsaverage-trans.fif"
    src = bem_dir / "fsaverage-ico-5-src.fif"
    bem = bem_dir / "fsaverage-5120-5120-5120-bem-sol.fif"
    for path in (trans, src, bem):
        if not path.exists():
            raise FileNotFoundError(f"Missing fsaverage asset: {path}")

    epochs = mne.read_epochs_eeglab(RAW_EEGLAB).crop(-0.1, -0.05)[0]
    epochs.pick(COMMON_CHANNELS)
    epochs.set_montage(mne.channels.make_standard_montage("standard_1005"))

    print("Computing forward solution (may take a few minutes)...")
    forward = mne.make_forward_solution(epochs.info, trans, src, bem, verbose=True)
    mne.write_forward_solution(FORWARD_PATH, forward, overwrite=True)
    print(f"Wrote {FORWARD_PATH}")


if __name__ == "__main__":
    main()
