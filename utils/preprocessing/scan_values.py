import numpy as np
import os


in_dir = "pp_test_full_gi"

for i in os.listdir(in_dir):
    if "img.npy" in i:
        print (i)
        arr = np.load(os.path.join(in_dir,i))
        if arr.shape != (256,256,256,3):
            print (arr.shape)
        print (np.mean(arr),np.amin(arr),np.amax(arr))

    if "rois.npy" in i:
        print (i)
        arr = np.load(os.path.join(in_dir,i))
        if arr.shape != (256,256,256):
            print (arr.shape)


print ("done")
