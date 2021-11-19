import numpy as np
import os


in_dir = "pp_train_128_ven"

for i in os.listdir(in_dir):
    if "img.npy" in i:
        print (i)
        arr = np.load(os.path.join(in_dir,i))

        print ("shape now ",arr.shape)

        arr = arr[:,:,:,0]

        arr = np.save(os.path.join(in_dir,i),arr)
print ("done")
