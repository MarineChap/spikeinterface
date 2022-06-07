"""Classes and functions for computing multiple quality metrics."""

from copy import deepcopy

import numpy as np
import pandas as pd

from spikeinterface.core.waveform_extractor import WaveformExtractor, BaseWaveformExtractorExtension
import spikeinterface.toolkit.qualitymetrics.metrics as metrics
from spikeinterface.toolkit.qualitymetrics.metrics.pca_metrics import _possible_pc_metric_names
from inspect import getmembers, isfunction

class QualityMetricCalculator(BaseWaveformExtractorExtension):
    """Class to compute quality metrics of spike sorting output.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor object

    Notes
    -----
    principal_components are loaded automatically if already computed.
    """

    extension_name = 'quality_metrics'

    def __init__(self, waveform_extractor):
        BaseWaveformExtractorExtension.__init__(self, waveform_extractor)

        if waveform_extractor.is_extension('principal_components'):
            self.principal_component = waveform_extractor.load_extension('principal_components')
        else:
            self.principal_component = None

        self.recording = waveform_extractor.recording
        self.sorting = waveform_extractor.sorting

        self._metrics = None

    def _set_params(self, metric_names=None, peak_sign='neg',
                    max_spikes_for_nn = 2000, n_neighbors = 6, seed=None):
        self._metric_func = []
        if metric_names is None:

            metric_names, self._metric_func = getmembers(metrics, isfunction)
            metric_names = [name.split("compute_")[0] for name in metric_names]

            # This is too slow
            # metric_names = list(_metric_name_to_func.keys()) + _possible_pc_metric_names

            # So by default we take all metrics and 3 metrics PCA based only
            # 'nearest_neighbor' is really slow and not taken by default

            if self.principal_component is not None:
                metric_names += ['isolation_distance', 'l_ratio', 'd_prime']
        else:

            for name in metric_names:
                fct = getattr(metrics, "compute_"+ name, None)
                if fct is None and name not in _possible_pc_metric_names:
                        raise ValueError("This metrics name does not exist - " + name) # Way to raise just a warning ?
                self._metric_func .append(fct)

        params = dict(metric_names=metric_names,
                      peak_sign=peak_sign,
                      max_spikes_for_nn=int(max_spikes_for_nn),
                      n_neighbors=int(n_neighbors),
                      seed=int(seed) if seed is not None else None)

        return params

    def _specific_load_from_folder(self):
        self._metrics = pd.read_csv(self.extension_folder / 'metrics.csv', index_col=0)

    def _reset(self):
        self._metrics = None

    def _specific_select_units(self, unit_ids, new_waveforms_folder):
        # filter metrics dataframe
        new_metrics = self._metrics.loc[np.array(unit_ids)]
        new_metrics.to_csv(new_waveforms_folder / self.extension_name / 'metrics.csv')

    def compute_metrics(self):
        """Compute quality metrics.

        Parameters
        ----------
        metric_names: list or None
            List of quality metrics to compute. If None, all metrics are computed
        **kwargs: keyword arguments for quality metrics (TODO)
            max_spikes_for_nn: int
                maximum number of spikes to use per cluster in PCA metrics
            n_neighbors: int
                number of nearest neighbors to check membership of in PCA metrics
            seed: int
                seed for pseudo-random number generator used in PCA metrics (e.g. nn_isolation)

        Returns
        -------
        metrics: pd.DataFrame

        """

        metric_names = self._params['metric_names']

        unit_ids = self.sorting.unit_ids
        metrics = pd.DataFrame(index=unit_ids)

        for func, name in zip(self._metric_func, metric_names):

            if name in _possible_pc_metric_names :   # PC metric based
                if self.principal_component is None:
                    raise ValueError('waveform_principal_component must be provided to compute ' + name + ' metrics')

                kwargs = {k: self._params[k] for k in ('max_spikes_for_nn', 'n_neighbors', 'seed')}
                pc_metrics = calculate_pc_metrics(self.principal_component,
                                                  metric_names=name, **kwargs)
                for col, values in pc_metrics.items():
                    metrics[col] = pd.Series(values)
            else:  # Misc metrics
                # TODO add for params from different functions
                kwargs = {k: self._params[k] for k in ('peak_sign',)}

                res = func(self.waveform_extractor, **kwargs)
                if isinstance(res, dict):
                    # res is a dict convert to series
                    metrics[name] = pd.Series(res)
                else:
                    # res is a namedtuple with several dict
                    # so several columns
                    for i, col in enumerate(res._fields):
                        metrics[col] = pd.Series(res[i])


        self._metrics = metrics

        # save to folder
        metrics.to_csv(self.extension_folder / 'metrics.csv')


    def get_metrics(self):
        """Get the computed metrics."""

        msg = "Quality metrics are not computed. Use the 'compute_metrics()' function."
        assert self._metrics is not None, msg
        return self._metrics


WaveformExtractor.register_extension(QualityMetricCalculator)


def compute_quality_metrics(waveform_extractor, load_if_exists=False,
                            metric_names=None, **params):
    """Compute quality metrics on waveform extractor.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        The waveform extractor to compute metrics on.
    load_if_exists : bool, optional, default: False
        Whether to load precomputed quality metrics, if they already exist.
    metric_names : list or None
        List of quality metrics to compute.
    **params
        Keyword arguments for quality metrics.

    Returns
    -------
    metrics: pandas.DataFrame
        Data frame with the computed metrics
    """

    folder = waveform_extractor.folder
    ext_folder = folder / QualityMetricCalculator.extension_name
    if load_if_exists and ext_folder.is_dir():
        qmc = QualityMetricCalculator.load_from_folder(folder)
    else:
        qmc = QualityMetricCalculator(waveform_extractor)
        qmc.set_params(metric_names=metric_names, **params)
        qmc.compute_metrics()

    metrics = qmc.get_metrics()

    return metrics


def get_quality_metric_list():
    """Get a list of the available quality metrics."""

    return deepcopy(list(_metric_name_to_func.keys()))
