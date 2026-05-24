from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class FilterOpts:
    cutoff: object  # float or list of floats
    btype: str
    order: int
    pad_time: float


@dataclass
class ChannelRejectPre:
    z_score_threshold_mad: List[float]
    z_score_threshold_power: float
    fmin_fmax: List[float]
    z_score_threshold_autocorr: float


@dataclass
class ChannelRejectPost:
    z_score_threshold_auc: float


@dataclass
class ChannelRejectOpts:
    pre: ChannelRejectPre
    post: ChannelRejectPost


@dataclass
class ICAFiltering:
    order_bandpass: int
    order_bandstop: int
    pad_time_bandpass: float
    cutoff: List[float]


@dataclass
class ICAOpts:
    pc_threshold: float
    bad_component_thresholds: dict
    n_min_comps_to_reject: dict
    threshold_min_components_to_reject: dict
    pre_timerange: List[float]
    filtering: ICAFiltering


@dataclass
class SoundOpts:
    max_iterations: int
    lambda_: float
    convergence_tolerance: float
    fixed_max_iterations: bool


@dataclass
class SSPSIROpts:
    timerange: list
    method: list


@dataclass
class OcularRejectOpts:
    z_threshold: float
    pre_timerange_min: float
    post_timerange: List[Optional[float]]


@dataclass
class PreTrialRejectOpts:
    global_zscore_threshold: List[float]
    local_zscore_threshold: float
    psd_zscore_threshold: object  # float or False
    psd_freq_range: List[float]


@dataclass
class PostTrialRejectOpts:
    global_zscore_threshold: List[float]
    local_zscore_threshold: float


@dataclass
class TrialRejectOpts:
    ocular: OcularRejectOpts
    pre: PreTrialRejectOpts
    post: PostTrialRejectOpts


@dataclass
class PreprocConfig:
    # Time ranges
    pre_range: List[float]
    post_range: List[float]
    baseline: List[float]
    reject_range: List[float]
    artifact_window_1: List[float]
    artifact_window_2: List[Optional[float]]

    # Channel rejection
    freq_range: List[float]
    channel_reject_opts: ChannelRejectOpts

    # ICA
    ica_opts: ICAOpts

    # SOUND & SSP-SIR
    sound_opts: SoundOpts
    ssp_sir_opts: SSPSIROpts

    # Trial rejection
    trial_reject_opts: TrialRejectOpts

    # Filter options
    filter_opts: FilterOpts

    # General
    target_sfreq: float
    use_ica_on_pre: bool

    # Channels
    common_channels: List[str]

    def to_dicts(self):
        """Convert config to the dict format used by processing functions."""
        channel_reject_opts = {
            'pre': {
                'z_score_threshold_mad': self.channel_reject_opts.pre.z_score_threshold_mad,
                'z_score_threshold_power': self.channel_reject_opts.pre.z_score_threshold_power,
                'fmin_fmax': self.channel_reject_opts.pre.fmin_fmax,
                'z_score_threshold_autocorr': self.channel_reject_opts.pre.z_score_threshold_autocorr,
            },
            'post': {
                'z_score_threshold_auc': self.channel_reject_opts.post.z_score_threshold_auc,
            },
        }

        ica_opts = {
            'pc_threshold': self.ica_opts.pc_threshold,
            'bad_component_thresholds': self.ica_opts.bad_component_thresholds,
            'n_min_comps_to_reject': self.ica_opts.n_min_comps_to_reject,
            'threshold_min_components_to_reject': self.ica_opts.threshold_min_components_to_reject,
            'pre_timerange': self.ica_opts.pre_timerange,
            'filtering': {
                'order_bandpass': self.ica_opts.filtering.order_bandpass,
                'order_bandstop': self.ica_opts.filtering.order_bandstop,
                'pad_time_bandpass': self.ica_opts.filtering.pad_time_bandpass,
                'cutoff': self.ica_opts.filtering.cutoff,
            },
        }

        sound_opts = {
            'max_iterations': self.sound_opts.max_iterations,
            'lambda': self.sound_opts.lambda_,
            'convergence_tolerance': self.sound_opts.convergence_tolerance,
            'fixed_max_iterations': self.sound_opts.fixed_max_iterations,
        }

        ssp_sir_opts = {
            'timerange': self.ssp_sir_opts.timerange,
            'method': self.ssp_sir_opts.method,
        }

        trial_reject_opts = {
            'ocular': {
                'z_threshold': self.trial_reject_opts.ocular.z_threshold,
                'pre_timerange_min': self.trial_reject_opts.ocular.pre_timerange_min,
                'post_timerange': self.trial_reject_opts.ocular.post_timerange,
            },
            'pre': {
                'global_zscore_threshold': self.trial_reject_opts.pre.global_zscore_threshold,
                'local_zscore_threshold': self.trial_reject_opts.pre.local_zscore_threshold,
                'psd_zscore_threshold': self.trial_reject_opts.pre.psd_zscore_threshold,
                'psd_freq_range': self.trial_reject_opts.pre.psd_freq_range,
            },
            'post': {
                'global_zscore_threshold': self.trial_reject_opts.post.global_zscore_threshold,
                'local_zscore_threshold': self.trial_reject_opts.post.local_zscore_threshold,
            },
        }

        filter_opts = {
            'cutoff': self.filter_opts.cutoff,
            'btype': self.filter_opts.btype,
            'order': self.filter_opts.order,
            'pad_time': self.filter_opts.pad_time,
        }

        return {
            'channel_reject_opts': channel_reject_opts,
            'ica_opts': ica_opts,
            'sound_opts': sound_opts,
            'ssp_sir_opts': ssp_sir_opts,
            'trial_reject_opts': trial_reject_opts,
            'filter_opts': filter_opts,
        }


def get_default_config() -> PreprocConfig:
    freq_range = [30, 47]

    return PreprocConfig(
        pre_range=[-0.505, -0.005],
        post_range=[-0.03, 0.1],
        baseline=[-0.025, -0.015],
        reject_range=[0.02, 0.06],
        artifact_window_1=[-0.014, 0.014],
        artifact_window_2=[None, 0.015],
        freq_range=freq_range,
        channel_reject_opts=ChannelRejectOpts(
            pre=ChannelRejectPre(
                z_score_threshold_mad=[-3, 3],
                z_score_threshold_power=5,
                fmin_fmax=freq_range,
                z_score_threshold_autocorr=4,
            ),
            post=ChannelRejectPost(z_score_threshold_auc=3),
        ),
        ica_opts=ICAOpts(
            pc_threshold=0.99,
            bad_component_thresholds={'eye blink': 0.9},
            n_min_comps_to_reject={'eye blink': 2},
            threshold_min_components_to_reject={'eye blink': 0.7},
            pre_timerange=[-1.1, -0.005],
            filtering=ICAFiltering(
                order_bandpass=2,
                order_bandstop=2,
                pad_time_bandpass=0.5,
                cutoff=[1, 100],
            ),
        ),
        sound_opts=SoundOpts(
            max_iterations=10,
            lambda_=0.01,
            convergence_tolerance=1e-9,
            fixed_max_iterations=False,
        ),
        ssp_sir_opts=SSPSIROpts(
            timerange=['automatic', 50],
            method=['threshold', 0.99],
        ),
        trial_reject_opts=TrialRejectOpts(
            ocular=OcularRejectOpts(
                z_threshold=2,
                pre_timerange_min=-0.1,
                post_timerange=[0.015, None],
            ),
            pre=PreTrialRejectOpts(
                global_zscore_threshold=[-8, 4],
                local_zscore_threshold=5,
                psd_zscore_threshold=False,
                psd_freq_range=freq_range,
            ),
            post=PostTrialRejectOpts(
                global_zscore_threshold=[-np.inf, 6],
                local_zscore_threshold=5,
            ),
        ),
        filter_opts=FilterOpts(cutoff=[2, 47], btype='bandpass', order=2, pad_time=0.1),
        target_sfreq=1000,
        use_ica_on_pre=False,
        common_channels=[
            'AF3', 'AF4', 'AF7', 'AF8', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
            'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'CPz', 'Cz',
            'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8',
            'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8',
            'Fp1', 'Fp2', 'Fpz', 'Fz', 'Iz', 'O1', 'O2', 'Oz',
            'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8',
            'PO3', 'PO4', 'PO7', 'PO8', 'POz', 'Pz',
            'T7', 'T8', 'TP7', 'TP8',
        ],
    )
