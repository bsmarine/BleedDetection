
import numpy as np
import os
import cProfile, pstats

from batchgenerators.transforms.spatial_transforms import MirrorTransform as Mirror
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading import SingleThreadedAugmenter
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.utility_transforms import ConvertSegToBoundingBoxCoordinates


def augmentation(patient_data):
    my_transforms = []

    mirror_transform = Mirror(axes=np.arange(3))
    my_transforms.append(mirror_transform)

    spatial_transform = SpatialTransform(patch_size=[256, 256, 256], patch_center_dist_from_border= (125.0, 125.0), 
                                do_elastic_deform=False, alpha=(0.0, 1500.0), sigma=(30.0, 50.0), do_rotation=True, 
                                angle_x= (0, 0.0),angle_y=(0, 0.0),angle_z=(0.0, 6.283185307179586), do_scale=True, 
                                scale=(0.8, 1.1),random_crop=False,order_data=2,order_seg=2)

    my_transforms.append(spatial_transform)


    my_transforms.append(ConvertSegToBoundingBoxCoordinates(3, get_rois_from_seg_flag=False, class_specific_seg_flag=False))
    all_transforms = Compose(my_transforms)

    multithreaded_generator = SingleThreadedAugmenter(patient_data, all_transforms)
    #multithreaded_generator = MultiThreadedAugmenter(patient_data, all_transforms, num_processes=cf.n_workers, seeds=range(cf.n_workers))
    
    return multithreaded_generator

##Dummy Data Creation

dumb_img = np.random.random_sample((3,256,256,256))-0.5
dumb_img.astype('float16')
data = list()
for i in range(0,8):
    data.append(dumb_img)

dumb_seg = np.zeros(shape=(1,256,256,256))
dumb_seg[0][120:135,120:135,120:135] = 1
dumb_seg.astype('uint8')
seg = list()
for i in range(0,8):
    seg.append(dumb_seg)

class_target = list()
for i in range(0,8):
    class_target.append([1])
batch_ids = [['1'],['2'],['3'],['4'],['5'],['6'],['7'],['8']]

# pp_dir = "/home/aisinai/data/preprocessed_data/pp_groin_256_f16"
# batch_ids = [['g11'],['g1'],['g4'],['g5'],['g14'],['g17'],['g19'],['g29']]
# data = list()
# seg = list()
# pids = list()

# img_batch = [os.path.join(pp_dir,"{}_img.npy".format(i)) for i in batch_ids]

# seg_batch = [os.path.join(pp_dir,"{}_rois.npy".format(i)) for i in batch_ids]

# for j in img_batch:
#     img = np.load(j)
#     data.append(img)

# for k in seg_batch:
#     roi = np.load(k)
#     seg.append(roi)

data = np.array(data)
seg = np.array(seg)
class_target = np.array(class_target)
print (data.shape,seg.shape,class_target.shape,class_target)

batches = list()

batch_one = {'data':data,'seg':seg,'pid':batch_ids,'class_target':class_target} #Data, Seg, PID dictionary

batches.append(batch_one)

batches_i = iter(batches)

### Run and Profile Standalone Script

profiler = cProfile.Profile()
profiler.enable()

augmented_data = augmentation(batches_i)

result = next(augmented_data)

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats()