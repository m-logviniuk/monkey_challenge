"""MONKEY: multi-centre inflammatory-cell detection in PAS kidney biopsies.

A compact U-Net density-map regressor detects lymphocytes and monocytes from
point annotations, trained and scored inside annotated ROIs only. The package
provides the data contract, ROI-masked Gaussian density targets, the model
(convolution and Kolmogorov-Arnold decoder heads), leave-one-centre-out
training with checkpoint/resume, peak detection, FROC scoring, metrics, and
figures.
"""

__version__ = "0.1.0"
