import matplotlib.pyplot as plt
from skimage import io, filters,exposure
import numpy as np
from skimage.morphology import disk
import skimage
print(skimage.__version__)
import matplotlib.pyplot as plt
import math

import ij_thresholds as ij_th

dir_ = "/Volumes/Seagate Backup Plus Drive/Nazario"
file_ = "45_DAPI.tif (blue).tif"

DAPI_img = io.imread(dir_ + '/' + file_)
DAPI_img = exposure.rescale_intensity(DAPI_img)

DAPI_img = filters.rank.median(DAPI_img, disk(8))

(fig,ax) = skimage.filters.try_all_threshold(DAPI_img, figsize=(16, 10))
plt.savefig(dir_ + '/compare_th.pdf')

start=filters.threshold_otsu(DAPI_img)
print(start)
print(np.mean(DAPI_img))
li_th1 = ij_th.ImageJ_Li_threshold(DAPI_img, tolerance=10, initial_guess=start)
print(li_th1)
io.imsave(dir_ + '/' + 'imagej_li_th.tif', (DAPI_img > li_th1).astype('uint8')*255)


li_th2 = filters.threshold_li(DAPI_img, tolerance=1, initial_guess=start)
print(li_th2)
io.imsave(dir_ + '/' + 'skimage_li_th.tif', (DAPI_img > li_th2).astype('uint8')*255)

IM_th = ij_th.ImageJ_Li_threshold(DAPI_img)
print(IM_th)
io.imsave(dir_ + '/' + 'imagej_IM_th.tif', (DAPI_img > IM_th).astype('uint8')*255)

# dir_ = "/Users/sarahkeegan/Dropbox/mac_files/fenyolab/data_and_results/davoli/Less beautiful FISH/LoVo"
# file_ = "5_RFP.tif"
#
# def invert(image):
#     return np.max(image) - image
# def reshape_to_rgb(grey_img):
#     #makes single color channel image into rgb
#     ret_img = np.zeros(shape=[grey_img.shape[0],grey_img.shape[1],3], dtype='uint8')
#     grey_img_=img_as_ubyte(grey_img)
#
#     ret_img[:,:,0]=grey_img_
#     ret_img[:, :, 1] = grey_img_
#     ret_img[:, :, 2] = grey_img_
#
#     return ret_img
#
# img = io.imread(dir_ + '/' + file_)
# img = img[:,:,0]
#
# #try tophat
# if(True):
#     ds_list=range(1,10,1)
#     for ds in ds_list:
#         img_th = morphology.white_tophat(img, selem=morphology.disk(ds), )
#         if (True): #ds == 1):
#             minus_img = img - img_th
#             minus_img = exposure.rescale_intensity(minus_img)
#             io.imsave(dir_ + '/th_test/minus_th_' + str(ds) + '.tif', minus_img)
#
#         img_th = exposure.rescale_intensity(img_th)
#         io.imsave(dir_ + '/th_test/th_'+str(ds)+'.tif', img_th)
#
#
#
#         meas_img_displ = reshape_to_rgb(img_th)
#         blobs = feature.blob_log(img_th, min_sigma=1, max_sigma=5, num_sigma=5, threshold=0.2, overlap=0.75,exclude_border=0)
#         for blob_i, blob in enumerate(blobs):
#             radius = (2 * blob[2]) ** 0.5
#             rr, cc = draw.circle_perimeter(int(blob[0]), int(blob[1]), int(radius * 3),
#                                            method='bresenham', shape=meas_img_displ.shape)
#             meas_img_displ[rr, cc] = [0, 0, 255]
#
#         io.imsave(dir_ + '/th_test/th_' + str(ds) + '_blobs.tif', meas_img_displ)
#
# # try rb-like background subtract
# if(False):
#
#     img_ = invert(img)
#     io.imsave(dir_ + '/th_test/inverted.tif', img_)
#
#     background = filters.threshold_local(img_, 51, offset=0, method='median') #np.percentile(img_, 1)
#     io.imsave(dir_ + '/th_test/background.tif', background)
#
#     bg_corrected = invert(img_ - background)
#     bg_corrected=bg_corrected.astype('uint8')
#     io.imsave(dir_ + '/th_test/bg_corrected.tif', bg_corrected)
#
#     if(True):
#         ds_list = range(1, 6, 1)
#         for ds in ds_list:
#             img_th = morphology.white_tophat(img, selem=morphology.disk(ds), )
#             io.imsave(dir_ + '/th_test/th_' + str(ds) + '_bg_corr.tif', img_th)
#
# if(False):
#     high=0 #4
#     for low in [0.5,1,1.5,2,2.5,3]:
#         filtered_image = filters.difference_of_gaussians(img, low, ) #4) #high_sigma=4,)
#         fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 12), dpi=300)
#         ax.imshow(filtered_image, cmap='gray')
#         fig.tight_layout()
#         fig.savefig(dir_ + '/th_test/dog_img'+str(low)+'_' + str(high) + '.tif')