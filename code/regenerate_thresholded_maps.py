"""Regenerate thresholded maps for figures."""
import os.path as op
from glob import glob

from nilearn import image

DERIV_DIR = op.abspath("../derivatives/ale")

img_files = sorted(
    glob(op.join(DERIV_DIR, "*_z_level-cluster_corr-FWE_method-montecarlo.nii.gz"))
)
for img_file in img_files:
    # Threshold with a z-statistic threshold corresponding to p<0.05.
    thresh_img = image.threshold_img(img_file, threshold=1.64)
    out_file = img_file.replace(
        "z_level-cluster_corr-FWE_method-montecarlo.nii.gz",
        "z_corr-FWE_thresh-05.nii.gz",
    )
    thresh_img.to_filename(out_file)
