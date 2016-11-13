import os, sys
import numpy as np
import scipy.misc

old_maps_dir = '/home/vlebedev/skinhack/wrinkles/mapped_2'
old_imgs_dir = '/home/vlebedev/skinhack/wrinkles/package_2'
new_maps_dir = '/home/vlebedev/skinhack/wrinkles_small/mapped_2'
new_imgs_dir = '/home/vlebedev/skinhack/wrinkles_small/package_2'

def list_files(p):
    return [os.path.join(p, x) for x in sorted(os.listdir(p))]
maps_files = list_files(old_maps_dir)
imgs_files = list_files(old_imgs_dir)

for i in range(len(maps_files)):
    stripped = os.path.basename(maps_files[i]).split('.')[0]
    m = scipy.misc.imread(maps_files[i])
    #extract red stripes
    mask = (1.0*m[:,:,0] - m[:,:,1] - m[:,:,2]) > 230
    #downsample to one tenth of orig size
    mask = scipy.misc.imresize(mask.astype(np.float32), 0.1) > 0
    m = scipy.misc.imread(imgs_files[i])
    img = scipy.misc.imresize(m, 0.1)
    
    scipy.misc.imsave(os.path.join(new_maps_dir, stripped + '.png'), mask)
    scipy.misc.imsave(os.path.join(new_imgs_dir, stripped + '.jpg'), img)
    
    sys.stdout.write('\r%d of %d' % (i + 1, len(maps_files)))
    sys.stdout.flush()
