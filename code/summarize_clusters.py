"""Create summary tables of the ALE meta-analyses."""
import os.path as op
from glob import glob

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import masking
from nimare import utils
from nimare.io import convert_sleuth_to_dataset
from nimare.meta.kernel import ALEKernel
from scipy import ndimage
from scipy.spatial.distance import cdist


def summarize_image(dset, img):
    img = nib.load(img)
    ale_kernel = ALEKernel()
    ma_imgs = ale_kernel.transform(dset, return_type="image")

    # Define array for 6-connectivity, aka NN1 or "faces"
    conn_mat = np.zeros((3, 3, 3), int)
    conn_mat[1, 1, :] = 1
    conn_mat[1, :, 1] = 1
    conn_mat[:, 1, 1] = 1

    stat_map = img.get_fdata()

    # Binarize using CDT
    binarized = stat_map > 1.64
    binarized = binarized.astype(int)

    # Now re-label and create table
    label_map = ndimage.measurements.label(binarized, conn_mat)[0]
    clust_ids = sorted(list(np.unique(label_map)[1:]))

    df = pd.DataFrame(index=dset.ids, columns=clust_ids)
    gingerale_df = pd.DataFrame(index=dset.ids, columns=clust_ids)

    for i_cluster, c_val in enumerate(clust_ids):
        cluster_mask = label_map == c_val
        cluster_idx = np.vstack(np.where(cluster_mask))

        cluster_mask = nib.Nifti1Image(
            cluster_mask.astype(int), img.affine, header=img.header
        )
        ma_data = masking.apply_mask(ma_imgs, cluster_mask)
        ma_summary = np.sum(ma_data, axis=1)
        ma_summary /= np.sum(ma_summary)
        df[c_val] = ma_summary

        for j_exp, exp_id in enumerate(dset.ids):
            coords = dset.coordinates.loc[dset.coordinates["id"] == exp_id]
            ijk = utils.mm2vox(coords[["x", "y", "z"]], img.affine)
            distances = cdist(cluster_idx.T, ijk)
            distances = distances < 1
            distances = np.any(distances, axis=0)
            n_included_voxels = np.sum(distances)
            gingerale_df.loc[exp_id, c_val] = n_included_voxels

    return df, gingerale_df


if __name__ == "__main__":
    project_dir = op.abspath("..")
    input_dir = op.join(project_dir, "derivatives/ale")
    output_dir = op.join(project_dir, "derivatives/ale-tables")

    # Combined meta-analysis
    # Import all the text files to create a combined text file dataset
    # All
    text_files = glob(op.join(project_dir, "code/text-files/*.txt"))
    dset = convert_sleuth_to_dataset(text_files, target="mni152_2mm")
    img_file = op.join(
        input_dir,
        "TrauEgtPTSD+HCgtPTSD_z_level-cluster_corr-FWE_method-montecarlo.nii.gz",
    )
    df, df_ga = summarize_image(dset, img_file)
    print(dset)
    df.to_csv(
        op.join(output_dir, "TrauEgtPTSD+HCgtPTSD_taylor.tsv"),
        sep="\t",
        index=True,
        index_label="Experiment",
    )
    df_ga.to_csv(
        op.join(output_dir, "TrauEgtPTSD+HCgtPTSD_gingerale.tsv"),
        sep="\t",
        index=True,
        index_label="Experiment",
    )

    # TrauEgtPTSD
    text_files = glob(op.join(project_dir, "code/text-files/*TrauEgtPTSD*.txt"))
    dset = convert_sleuth_to_dataset(text_files, target="mni152_2mm")
    img_file = op.join(
        input_dir, "TrauEgtPTSD_z_level-cluster_corr-FWE_method-montecarlo.nii.gz"
    )
    df, df_ga = summarize_image(dset, img_file)
    print(dset)
    df.to_csv(
        op.join(output_dir, "TrauEgtPTSD_taylor.tsv"),
        sep="\t",
        index=True,
        index_label="Experiment",
    )
    df_ga.to_csv(
        op.join(output_dir, "TrauEgtPTSD_gingerale.tsv"),
        sep="\t",
        index=True,
        index_label="Experiment",
    )

    # HCgtPTSD
    text_files = glob(op.join(project_dir, "code/text-files/*HCgtPTSD*.txt"))
    dset = convert_sleuth_to_dataset(text_files, target="mni152_2mm")
    img_file = op.join(
        input_dir, "HCgtPTSD_z_level-cluster_corr-FWE_method-montecarlo.nii.gz"
    )
    df, df_ga = summarize_image(dset, img_file)
    print(dset)
    df.to_csv(
        op.join(output_dir, "HCgtPTSD_taylor.tsv"),
        sep="\t",
        index=True,
        index_label="Experiment",
    )
    df_ga.to_csv(
        op.join(output_dir, "HCgtPTSD_gingerale.tsv"),
        sep="\t",
        index=True,
        index_label="Experiment",
    )
