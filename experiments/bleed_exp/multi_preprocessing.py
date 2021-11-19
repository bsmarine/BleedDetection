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
import random
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
        print('processing {} {}'.format(pid,ii), img.GetSpacing(), img_arr.shape,np.mean(img_arr),np.std(img_arr))
        #img_arr = resample_array(img_arr, img.GetSpacing(), cf.target_spacing) #already re-sampled in prior pre-processing
        img_arr = np.clip(img_arr, -1200, 600)
        #img_arr = (1200 + img_arr) / (600 + 1200) * 255  # a+x / (b-a) * (c-d) (c, d = new)
        concat_images.append(img_arr)

    mask = sitk.ReadImage(os.path.join(cf.raw_data_dir, pid, '01_mask.nii.gz'))
    mask_arr = sitk.GetArrayFromImage(mask).astype(np.uint8)
    mask_arr[mask_arr>10] = 0

    concat_images.append(mask_arr)

    # Find random patch of patch size with random offset, apply to images

    if cf.pp_patches is not None:
        
        concat_images = generate_positive_patches(mask_arr,concat_images,cf.pp_patches,40)

    # Remove mask_arr from concat

    mask_arr = concat_images.pop()
    
    # Concatenate images into singe img array

    concat = np.stack(concat_images,axis=3)

    # Z normalization of concatenated images as one multi-dimensional array

    concat = concat.astype(np.float32)
    concat = (concat - np.mean(concat)) / np.std(concat).astype(np.float16)

    print ("After concatenation ",np.mean(concat),np.std(concat),concat.dtype)
    
    print ("Concatenated Img Shape "+str(concat.shape))

    #Open Characteristics File
    df = pd.read_csv(os.path.join(cf.root_dir, 'raw_characteristics_gi.csv'), sep=',',converters={'PatientID': lambda x: str(x)})
    df = df[df.PatientID == pid]

    #Make Masks Array, Grab Mask ID Per Patient
    #final_rois = np.zeros_like(img_arr, dtype=np.uint8)
    mal_labels = []
    roi_ids = set([ii.split('.')[0].split('_')[0] for ii in os.listdir(path) if '01_mask.nii.gz' in ii])
    print (roi_ids)
    rix = 1
    for rid in roi_ids:

        #Grab Mask Paths and Nodule IDs
        roi_id_paths = [ii for ii in os.listdir(path) if '01_mask.nii' in ii]
        print ("ROI ID Paths:"+str(roi_id_paths))
        nodule_ids = [ii.split('.')[0].split('_')[0].lstrip("0") for ii in roi_id_paths]
        print ("Nodule ID:"+str(nodule_ids))

        #Grab Severity Value From Characteristics file
        rater_labels = [1] #[df[df.ROI_ID == int(ii)].Severity.values[0] for ii in nodule_ids]
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

        # roi = sitk.ReadImage(os.path.join(cf.raw_data_dir, pid, rp))
        # roi_arr = sitk.GetArrayFromImage(roi).astype(np.uint8)
        # roi_arr[roi_arr>10] = 0
        roi_arr = mask_arr

        if cf.multiphase:

        # Will need to change manually if two-phase ie img_arr = concat[:,:,0]

            img_arr = concat[:,:,:,0]
        else:
            img_arr= concat

        #roi_arr = resample_array(roi_arr, roi.GetSpacing(), cf.target_spacing)
        assert roi_arr.shape == img_arr.shape, [roi_arr.shape, img_arr.shape, pid, mask.GetSpacing()]
        
        for ix in range(len(img_arr.shape)):
            npt.assert_almost_equal(mask.GetSpacing()[ix], img.GetSpacing()[ix])
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

def generate_positive_patches(mask_arr,studies,patch_size,random_center_displacement):

    q = random_center_displacement
    
    pos_patches = list()

    where = np.where(mask_arr==1)
    z_mid = int(where[0].mean())
    y_mid = int(where[1].mean())
    x_mid = int(where[2].mean())

    print (x_mid,y_mid,z_mid)

    repeat = True
    
    while repeat == True:

        z_start = random.randint(-(q),q)+z_mid-(patch_size[0]/2)
        if z_start < 0:
            z_start = 0
        y_start = random.randint(-(q),q)+y_mid-(patch_size[1]/2)
        if y_start < 0:
            y_start = 0
        x_start = random.randint(-(q),q)+x_mid-(patch_size[2]/2)
        if x_start < 0:
            x_start = 0

        z_end = z_start+patch_size[0]
        y_end = y_start+patch_size[1]
        x_end = x_start+patch_size[2]

        numbers = [ int(x) for x in [x_start,x_end,y_start,y_end,z_start,z_end] ]

        pos_patches = [numbers]

        print ("Positive Patches: ",pos_patches)

        Z = pos_patches[0]
        patches = list()
        for arr in studies:
            print (arr.shape,Z)
            data = arr[Z[4]:Z[5],Z[2]:Z[3],Z[0]:Z[1]]

            ## Pad if doesn't fit correct full patch size
            if np.any([data.shape[dim] < ps for dim, ps in enumerate(patch_size)]):
                new_shape = [np.max([data.shape[dim], ps]) for dim, ps in enumerate(patch_size)]
                data = pad_nd_image(data, new_shape, mode='constant')
            
            print ("Patch Shape ",data.shape)

            mean = np.mean(data)

            if len(np.unique(data))==2:
                if mean == 0.0:
                    print ("Mask missing "+str(mean))
                    repeat = True
                    break
            
            if mean > -650:
                print ("Appropriate Mean of Patch "+str(mean))
                repeat = False
            else:
                print ("Inappropriate Mean of Patch "+str(mean))
                repeat = True
                break

            patches.append(data)
    
    return patches

def pad_nd_image(image, new_shape=None, mode="edge", kwargs=None, return_slicer=False, shape_must_be_divisible_by=None):
    """
    one padder to pad them all. Documentation? Well okay. A little bit. by Fabian Isensee

    :param image: nd image. can be anything
    :param new_shape: what shape do you want? new_shape does not have to have the same dimensionality as image. If
    len(new_shape) < len(image.shape) then the last axes of image will be padded. If new_shape < image.shape in any of
    the axes then we will not pad that axis, but also not crop! (interpret new_shape as new_min_shape)
    Example:
    image.shape = (10, 1, 512, 512); new_shape = (768, 768) -> result: (10, 1, 768, 768). Cool, huh?
    image.shape = (10, 1, 512, 512); new_shape = (364, 768) -> result: (10, 1, 512, 768).

    :param mode: see np.pad for documentation
    :param return_slicer: if True then this function will also return what coords you will need to use when cropping back
    to original shape
    :param shape_must_be_divisible_by: for network prediction. After applying new_shape, make sure the new shape is
    divisibly by that number (can also be a list with an entry for each axis). Whatever is missing to match that will
    be padded (so the result may be larger than new_shape if shape_must_be_divisible_by is not None)
    :param kwargs: see np.pad for documentation
    """
    if kwargs is None:
        kwargs = {}

    if new_shape is not None:
        old_shape = np.array(image.shape[-len(new_shape):])
    else:
        assert shape_must_be_divisible_by is not None
        assert isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray))
        new_shape = image.shape[-len(shape_must_be_divisible_by):]
        old_shape = new_shape

    num_axes_nopad = len(image.shape) - len(new_shape)

    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

    if not isinstance(new_shape, np.ndarray):
        new_shape = np.array(new_shape)

    if shape_must_be_divisible_by is not None:
        if not isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray)):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(new_shape)
        else:
            assert len(shape_must_be_divisible_by) == len(new_shape)

        for i in range(len(new_shape)):
            if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                new_shape[i] -= shape_must_be_divisible_by[i]

        new_shape = np.array([new_shape[i] + shape_must_be_divisible_by[i] - new_shape[i] % shape_must_be_divisible_by[i] for i in range(len(new_shape))])

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [[0, 0]]*num_axes_nopad + list([list(i) for i in zip(pad_below, pad_above)])
    res = np.pad(image, pad_list, mode, **kwargs)
    if not return_slicer:
        return res
    else:
        pad_list = np.array(pad_list)
        pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
        slicer = list(slice(*i) for i in pad_list)
        return res, slicer


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
