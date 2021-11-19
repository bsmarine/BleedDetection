import code
import numpy as np
import os

in_dir = "pp_groin_full_f16"

for i in os.listdir(in_dir):
   if 'img.npy' in os.path.join(in_dir,i):
     arr = np.load(os.path.join(in_dir,i))
     print (i,np.mean(arr),np.std(arr))


#code.interact(local=locals())
