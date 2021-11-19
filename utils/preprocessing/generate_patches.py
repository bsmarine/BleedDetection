import numpy as np 
import SimpleITK as sitk 
import random

def generate_negative_patches(studies):

    art_arr = sitk.GetArrayFromImage(studies[0])
    ven_arr = sitk.GetArrayFromImage(studies[1])
    noncon_arr = sitk.GetArrayFromImage(studies[2])
    mask_arr = sitk.GetArrayFromImage(studies[3])

    tot_num_neg = 1

    neg_patch = 0
    
    #### Choose where to randomly select a negative patch using
    #### Gaussian dist. with peak and std according to
    #### blocks of 32 voxels. For example, 16 blocks in 512 voxels.
    rand_xy = np.around(np.random.normal(4,2,size=(2,1000)))
    rand_z = np.around(np.random.normal(10,4,1000))

    rand_xyz = np.vstack((rand_xy, rand_z))

    i = 0

    chosen_xyz = list()
    output_patches = list()

    ##Create Random Negative Patches
    while neg_patch<tot_num_neg:
        bad_patch_flag=False
        i = i+1
        print (i)
        
        ## If no patch is found after 80 attempts, abort
        if i == 80:
            return
        
        ## Randomly selected blocks of 32 converted to voxels
        x_start = int(rand_xyz[0][i]*32) 
        y_start = int(rand_xyz[1][i]*32)
        z_start = int(rand_xyz[2][i]*32)
        
        # Choose 256^3 patch
        x_end = x_start+256
        y_end = y_start+256
        z_end = z_start+256

        print ("Random patch: ")
        print (x_start,x_end,y_start,y_end,z_start,z_end)

        patch_noncon = noncon_arr[z_start:z_end,y_start:y_end,x_start:x_end]
        patch_ven = ven_arr[z_start:z_end,y_start:y_end,x_start:x_end]
        patch_art = art_arr[z_start:z_end,y_start:y_end,x_start:x_end]
        patch_mask = mask_arr[z_start:z_end,y_start:y_end,x_start:x_end]

        if [rand_xyz[0][i],rand_xyz[1][i],rand_xyz[2][i]] in chosen_xyz or rand_xyz[0][i]<0 \
            or rand_xyz[0][i]>11 or rand_xyz[1][i]>11 or rand_xyz[1][i]<0 or rand_xyz[2][i]>15 or rand_xyz[2][i]<0:
            print ("Exclude Random Patch --> Outside FOV")
            continue
        
        ##Determine if patch has a low mean value, std or excessive 0 values
        for arr_patch in [patch_noncon,patch_ven,patch_art]:
            if len(np.where(arr_patch==0)[0]) > 0.3*(256**3) or arr_patch.mean()<-450:
                print ("Try new patch, likely off target given stats")
                print (len(np.where(arr_patch==0)[0])/(256**3))
                print (arr_patch.mean())
                print (arr_patch.std())
                bad_patch_flag=True
        
        if bad_patch_flag == True or patch_mask.mean()>0:
            print ("Rejected 'cause of flag or patch mask mean greater than 0")
            print (patch_mask.mean()>0)
            continue

        print ([x_start,x_end,y_start,y_end,z_start,z_end])
        output_patches.append([x_start,x_end,y_start,y_end,z_start,z_end])
        chosen_xyz.append([rand_xyz[0][i],rand_xyz[1][i],rand_xyz[2][i]])
        print ("Identified a neg_patch number : "+str(neg_patch))
        neg_patch = neg_patch + 1

        #sitk.WriteImage(sitk.GetImageFromArray(arr_patch),os.path.join(OUT_DIR,str(i)+".nii.gz"))
                
    return output_patches

def generate_positive_patches(studies,patch_size,random_center_displacement):

    q = random_center_displacement
    
    pos_patch = list()

    mask_arr = sitk.GetArrayFromImage(studies[3])

    where = np.where(mask_arr==1)
    z_mid = int(where[0].mean())
    y_mid = int(where[1].mean())
    x_mid = int(where[2].mean())

    print (z_mid,y_mid,x_mid)

    z_start = random.randint(-(q),q)+z_mid-(patch_size[0]/2)
    y_start = random.randint(-(q),q)+y_mid-(patch_size[1]/2)
    x_start = random.randint(-(q),q)+x_mid-(patch_size[2]/2)

    z_end = z_start+patch_size[0]
    y_end = y_start+patch_size[1]
    x_end = x_start+patch_size[2]

    numbers = [ int(x) for x in [x_start,x_end,y_start,y_end,z_start,z_end] ]

    pos_patch.append(numbers)

    return pos_patch

def test_onbody_patch(load_path):
    try:
        arr = sitk.GetArrayFromImage(sitk.ReadImage(load_path))
        print (load_path)
        print (np.mean(arr))
        # print (np.max(arr))
        # print (np.min(arr))
    except Exception as e:
        print ("Could not convert because "+str(e))
