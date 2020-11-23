#!/usr/bin/env python
# Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

'''
This preprocessing script loads nrrd files obtained by the data conversion tool: https://github.com/MIC-DKFZ/LIDC-IDRI-processing/tree/v1.0.1
After applying preprocessing, images are saved as numpy arrays and the meta information for the corresponding patient is stored
as a line in the dataframe saved as info_df.pickle.
'''

import os
import SimpleITK as sitk
import numpy as np
from multiprocessing import Pool
import pandas as pd
import numpy.testing as npt
from skimage.transform import resize
import subprocess
import pickle
import json
import configs
cf = configs.configs()

def pp_patient(inputs):

    #read image
    ix, path = inputs
    pid = path.split('/')[-1]
    
    if cf.multiphase:
        with open(os.path.abspath(cf.pp_mp_cf), 'r') as f:
            pp_mp_cf = json.load(f)
    phases = pp_mp_cf[cf.mp_setting]
    
    concat_images = list()
    
    for ii in phases:
        
        img = sitk.ReadImage(os.path.join(path,'{}.nii.gz'.format(ii)))

        img_arr = sitk.GetArrayFromImage(img)
        print('processing {} {}'.format(pid,ii), img.GetSpacing(), img_arr.shape)
        #img_arr = resample_array(img_arr, img.GetSpacing(), cf.target_spacing) #already re-sampled in prior pre-processing
        img_arr = np.clip(img_arr, -1200, 600)
        #img_arr = (1200 + img_arr) / (600 + 1200) * 255  # a+x / (b-a) * (c-d) (c, d = new)
        img_arr = img_arr.astype(np.float16)
        img_arr = (img_arr - np.mean(img_arr)) / np.std(img_arr).astype(np.float16)
        concat_images.append(img_arr)
        
    concat = np.stack(concat_images,axis=3)
    
    print ("Concatenated Img Shape "+str(concat.shape))

    #Open Characteristics File
    df = pd.read_csv(os.path.join(cf.root_dir, 'raw_characteristics.csv'), sep=',',converters={'PatientID': lambda x: str(x)})
    df = df[df.PatientID == pid]

    #Make Masks Array, Grab Mask ID Per Patient
    final_rois = np.zeros_like(img_arr, dtype=np.uint8)
    mal_labels = []
    roi_ids = set([ii.split('.')[0].split('_')[0] for ii in os.listdir(path) if 'mask.nii.gz' in ii])
    rix = 1
    for rid in roi_ids:

        #Grab Mask Paths and Nodule IDs
        roi_id_paths = [ii for ii in os.listdir(path) if 'mask.nii' in ii]
        print ("ROI ID Paths:"+str(roi_id_paths))
        nodule_ids = [ii.split('.')[0].split('_')[0].lstrip("0") for ii in roi_id_paths]
        print ("Nodule ID:"+str(nodule_ids))

        #Grab Severity Value From Characteristics file
        rater_labels = [df[df.ROI_ID == int(ii)].Severity.values[0] for ii in nodule_ids]
        print ("Rater Labels:"+str(rater_labels))

        ##Take Mean Severity Value
        #rater_labels.extend([0] * (4-len(rater_labels)))
        #mal_label = np.mean([ii for ii in rater_labels if ii > -1])
        mal_label = rater_labels
        mal_list = mal_label
        print ("#############Mal Label: "+str(mal_list))
        ##Read Mask Paths
        #roi_rater_list = []
        # for rp in roi_id_paths:
        rp = roi_id_paths[0]
        roi = sitk.ReadImage(os.path.join(cf.raw_data_dir, pid, rp))
        roi_arr = sitk.GetArrayFromImage(roi).astype(np.uint8)
        roi_arr = resample_array(roi_arr, roi.GetSpacing(), cf.target_spacing)
        assert roi_arr.shape == img_arr.shape, [roi_arr.shape, img_arr.shape, pid, roi.GetSpacing()]
        
        for ix in range(len(img_arr.shape)):
            npt.assert_almost_equal(roi.GetSpacing()[ix], img.GetSpacing()[ix])
        #roi_rater_list.append(roi_arr)

        final_rois = roi_arr

        # roi_rater_list.extend([np.zeros_like(roi_rater_list[-1])]*(4-len(roi_id_paths)))
        # roi_raters = np.array(roi_rater_list)
        # roi_raters = np.mean(roi_raters, axis=0)
        # roi_raters[roi_raters < 0.5] = 0

        # if np.sum(roi_raters) > 0:
        #     mal_labels.append(mal_label)
        #     final_rois[roi_raters >= 0.5] = rix
        #     rix += 1
        # else:
        #     # indicate rois suppressed by majority voting of raters
        #     print('suppressed roi!', roi_id_paths)
        #     with open(os.path.join(cf.pp_dir, 'suppressed_rois.txt'), 'a') as handle:
        #         handle.write(" ".join(roi_id_paths))

    #Generate Foreground Slice Indices
    final_rois = np.around(final_rois)
    fg_slices = [ii for ii in np.unique(np.argwhere(final_rois != 0)[:, 0])]
    
    #Make Array From Severity 
    #mal_labels = np.array(mal_label)

    if mal_list[0] == [0]:
        mal_labels_assert_test = []
    else:
        mal_labels_assert_test = mal_list

    print ("Print Malignancy Labels:"+str(mal_list))
    print ("Print Unique Values in ROI Array:"+str(len(np.unique(final_rois))))


    assert len(mal_labels_assert_test) + 1 == len(np.unique(final_rois)), [len(mal_labels), np.unique(final_rois), pid]

    np.save(os.path.join(cf.pp_dir, '{}_rois.npy'.format(pid)), final_rois)
    np.save(os.path.join(cf.pp_dir, '{}_img.npy'.format(pid)), concat)

    with open(os.path.join(cf.pp_dir, 'meta_info_{}.pickle'.format(pid)), 'wb') as handle:
        meta_info_dict = {'pid': pid, 'class_target': mal_list, 'spacing': img.GetSpacing(), 'fg_slices': fg_slices}
        print (meta_info_dict)
        pickle.dump(meta_info_dict, handle)

def aggregate_meta_info(exp_dir):

    files = [os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if 'meta_info' in f]
    df = pd.DataFrame(columns=['pid', 'class_target', 'spacing', 'fg_slices'])
    for f in files:
        with open(f, 'rb') as handle:
            df.loc[len(df)] = pickle.load(handle)

    df.to_pickle(os.path.join(exp_dir, 'info_df.pickle'))
    print ("aggregated meta info to df with length", len(df))

def resample_array(src_imgs, src_spacing, target_spacing):

    src_spacing = np.round(src_spacing, 3)
    target_shape = [int(src_imgs.shape[ix] * src_spacing[::-1][ix] / target_spacing[::-1][ix]) for ix in range(len(src_imgs.shape))]
    for i in range(len(target_shape)):
        try:
            assert target_shape[i] > 0
        except:
            raise AssertionError("AssertionError:", src_imgs.shape, src_spacing, target_spacing)

    img = src_imgs.astype(float)
    resampled_img = resize(img, target_shape, order=1, clip=True, mode='edge').astype('float16')

    return resampled_img

#Convert .nii.gz to .nrrd

if __name__ == "__main__":

    paths = [os.path.join(cf.raw_data_dir, ii) for ii in os.listdir(cf.raw_data_dir) if not ii.startswith('.')]

    if not os.path.exists(cf.pp_dir):
        os.makedirs(cf.pp_dir)

    # pool = Pool(processes=1)
    # p1 = pool.map(pp_patient, enumerate(paths), chunksize=1)
    # pool.close()
    # pool.join()
    for i in enumerate(paths):
        pp_patient(i)

    aggregate_meta_info(cf.pp_dir)
    subprocess.call('cp {} {}'.format(os.path.join(cf.pp_dir, 'info_df.pickle'), os.path.join(cf.pp_dir, 'info_df_bk.pickle')), shell=True)
