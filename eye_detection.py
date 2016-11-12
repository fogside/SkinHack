import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.morphology import dilation
from skimage.measure import regionprops
import matplotlib.patches as mpatches
from skimage.morphology import convex_hull_image

import numpy as np
from scipy.misc import imresize
from os import listdir
from scipy import ndimage
from skimage.measure import label



folder='/Users/fogside/Projects/SkinHack/data_probe0/'
img_array = [mpimg.imread(folder+fname) for fname in listdir(folder) if fname.endswith('.tif')]
names = [name for name in listdir(folder) if name.endswith('.tif')]
base_shape = np.array(img_array[0].shape)[:2]
new_shape = np.array([250,400])
rectangles = open('rectangles.txt', 'w')


### resize ### 

for i in range (len(img_array)):
    img_array[i] = imresize(img_array[i], new_shape, interp='lanczos') / 255



def neighbourhood(x, y, img, rad):
    neigh = []
    for X in range(max(x - rad, 0), min(x + rad, img.shape[0])):
        for Y in range(max(y - rad, 0), min(y + rad, img.shape[1])):
            neigh.append(img[X][Y])
    return neigh


def filter_image(img_input):

    '''
    
    len(image_input.shape) == 3
    
    '''
    
    filter = np.array([[[0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]],
                       [[-0.5, -0.5, -0.5],
                        [-0.5,    4, -0.5],
                        [-0.5, -0.5, -0.5]],
                       [[0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]]])

    img = ndimage.convolve(img_input, filter)
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    rgb = [r, g, b]

    index = r < (5 * np.mean(r) + np.min(r)) / 6
    r[index] = 0
    # index = r >= (5 * np.mean(r) + np.min(r)) / 6
    r[~index] = 1
    R = np.copy(r)

    # Пройдем по пикселям, проверяя соседей в некотором радиусе. Если слишком много белых соседей, забеляем пиксель.
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            if np.mean(np.array((neighbourhood(i, j, r, 5)))) > 0.4:
                R[i][j] = 1

    return(1-R)


def find_eye(label_img, img):
	# image_label_overlay = label2rgb(label_image, image=img)

	# fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
	# ax.imshow(image_label_overlay)

	for region in regionprops(label_image):
	    # skip small images
	    if region.area < 100:
	        continue

	    # draw rectangle around segmented coins
	    minr, minc, maxr, maxc = region.bbox
	    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
	                              fill=False, edgecolor='red', linewidth=2)
	    # ax.add_patch(rect)

	img[minr:maxr, minc:maxc] = convex_hull_image(img[minr:maxr, minc:maxc])
	binary_mask = imresize(img, base_shape)/255
	### binarization ###
	binary_mask = (binary_mask>0.01)*1

	### find right rectangles ###
	koeff = base_shape/new_shape

	return binary_mask, koeff[0]*minr, koeff[0]*maxr, koeff[1]*minc, koeff[1]*maxc


for i, img in enumerate(img_array):

	img = filter_image(img)
	img = dilation(img)
	## all white color is background
	label_image = label(img, background=0)
	binary_mask, minr, maxr, minc, maxc = find_eye(label_image, img)
	np.save(names[i], binary_mask)
	rectangles.write("{},{},{},{},{}\n".format(names[i], minr, maxr, minc, maxc))
	print('Process for {}, name {} has been complited'.format(i, names[i]))

rectangles.close()
