import numpy as np
import os


in_dir = "pp_groin_full"

for i in os.listdir(in_dir):
    if "rois.npy" in i:
        print (i)
        arr = np.load(os.path.join(in_dir,i))

        arr = arr.astype('int8')

        print ("Dtype now ",arr.dtype)

print ("done")
