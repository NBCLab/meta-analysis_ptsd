import os
import os.path as op
import argparse
import numpy as np
import nibabel as nib
from nimare.io import convert_sleuth_to_dataset
from glob import glob
from nimare.correct import FWECorrector
from nimare.meta.ale import ALE, ALESubtraction

def thresh_img(logp_img, z_img, p):
    # Natural log was used until NiMARE 0.0.4, and this manuscript used 0.0.3.
    sig_inds = np.where(logp_img.get_fdata() > -np.log(p))
    z_img_data = z_img.get_fdata()
    z_img_thresh_data = np.zeros(z_img.shape)
    z_img_thresh_data[sig_inds] = z_img_data[sig_inds]
    z_img = nib.Nifti1Image(z_img_thresh_data, z_img.affine)
    return z_img

project_dir = '/home/data/nbc/misc-projects/meta-analyses/meta-analysis_ptsd'
output_dir = op.join(project_dir, 'derivatives', 'ale')

text_files_all = glob(op.join(project_dir, 'code', 'text-files', '*.txt'))
text_files_trauma = glob(op.join(project_dir, 'code', 'text-files', '*TrauEgtPTSD*.txt'))
text_files_hc = glob(op.join(project_dir, 'code', 'text-files', '*HCgtPTSD*.txt'))

#Combined meta-analysis
#Import all the text files to create a combined text file dataset
dset = convert_sleuth_to_dataset(text_files_all, target="mni152_2mm")

ale = ALE(kernel__fwhm=None)

results = ale.fit(dset)
corr = FWECorrector(
    method="montecarlo", n_iters=10000, voxel_thresh=0.001, n_cores=4
)
cres = corr.transform(results)

prefix = 'TrauEgtPTSD+HCgtPTSD'

cres.save_maps(output_dir=output_dir, prefix=prefix)

os.makedirs(output_dir, exist_ok=True)

dset.save(op.join(output_dir, prefix + ".pkl.gz"))

z_img_logp = nib.load(op.join(output_dir, '{prefix}_logp_level-cluster_corr-FWE_method-montecarlo.nii.gz'.format(prefix=prefix)))
z_img = nib.load(op.join(output_dir, '{prefix}_z.nii.gz'.format(prefix=prefix)))
z_img_thresh = thresh_img(z_img_logp, z_img, 0.05)
nib.save(z_img_thresh, op.join(output_dir, '{prefix}_z_corr-FWE_thresh-05.nii.gz'.format(prefix=prefix)))

#Individual meta-analyses and contrast
dset1 = convert_sleuth_to_dataset(text_files_trauma, target="mni152_2mm")
dset2 = convert_sleuth_to_dataset(text_files_hc, target="mni152_2mm")

ale1 = ALE(kernel__fwhm=None)
ale2 = ALE(kernel__fwhm=None)

res1 = ale1.fit(dset1)
res2 = ale2.fit(dset2)
corr = FWECorrector(
    method="montecarlo", n_iters=10000, voxel_thresh=0.001, n_cores=4)
cres1 = corr.transform(res1)
cres2 = corr.transform(res2)
sub = ALESubtraction(n_iters=10000)
sres = sub.fit(ale1, ale2)

prefix1 = 'TrauEgtPTSD'
prefix2 = 'HCgtPTSD'
prefix3 = 'TrauEgtPTSD-HCgtPTSD'
cres1.save_maps(output_dir=output_dir, prefix=prefix1)
cres2.save_maps(output_dir=output_dir, prefix=prefix2)
sres.save_maps(output_dir=output_dir, prefix=prefix3)
dset1.save(op.join(output_dir, prefix1 + ".pkl.gz"))
dset2.save(op.join(output_dir, prefix2 + ".pkl.gz"))

#get thresholded individual analyses
z_img_group1_logp  = nib.load(op.join(output_dir, '{prefix1}_logp_level-cluster_corr-FWE_method-montecarlo.nii.gz'.format(prefix1=prefix1)))
z_img_group1 = nib.load(op.join(output_dir, '{prefix1}_z.nii.gz'.format(prefix1=prefix1)))
z_img_group1_thresh = thresh_img(z_img_group1_logp, z_img_group1, 0.05)
nib.save(z_img_group1_thresh, op.join(output_dir, '{prefix1}_z_corr-FWE_thresh-05.nii.gz'.format(prefix1=prefix1)))

z_img_group2_logp  = nib.load(op.join(output_dir, '{prefix2}_logp_level-cluster_corr-FWE_method-montecarlo.nii.gz'.format(prefix2=prefix2)))
z_img_group2 = nib.load(op.join(output_dir, '{prefix2}_z.nii.gz'.format(prefix2=prefix2)))
z_img_group2_thresh = thresh_img(z_img_group2_logp, z_img_group2, 0.05)
nib.save(z_img_group2_thresh, op.join(output_dir, '{prefix2}_z_corr-FWE_thresh-05.nii.gz'.format(prefix2=prefix2)))

#get thresholded contrast analyses
z_img = nib.load(op.join(output_dir, '{prefix}_z_desc-group1MinusGroup2.nii.gz'.format(prefix=prefix3)))

#for first direction
sig_inds = np.where(z_img.get_fdata() > -np.log(0.05))

z_img_group1_data = z_img_group1.get_fdata()
z_img_group2_data = z_img_group2.get_fdata()

#for first direction
zimg1_sub_zimg2 = z_img_group1_data - z_img_group2_data

z_img_group1_thresh_data = np.zeros(z_img_group1.shape)
sig_inds = np.where(z_img.get_fdata() < np.log(0.05))
z_img_group1_thresh_data[sig_inds] = zimg1_sub_zimg2[sig_inds]

z_img_group1_final = nib.Nifti1Image(z_img_group1_thresh_data, z_img_group1.affine)
nib.save(z_img_group1_final, op.join(output_dir, '{prefix1}-{prefix2}_z_thresh-05.nii.gz'.format(prefix1=prefix1, prefix2=prefix2)))

#for second direction
zimg2_sub_zimg1 = z_img_group2_data - z_img_group1_data

z_img_group2_thresh_data = np.zeros(z_img_group2.shape)
sig_inds = np.where(z_img.get_fdata() < np.log(0.05))
z_img_group2_thresh_data[sig_inds] = zimg2_sub_zimg1[sig_inds]

z_img_group2_final = nib.Nifti1Image(z_img_group2_thresh_data, z_img_group2.affine)
nib.save(z_img_group2_final, op.join(output_dir, '{prefix2}-{prefix1}_z_thresh-05.nii.gz'.format(prefix1=prefix1, prefix2=prefix2)))
