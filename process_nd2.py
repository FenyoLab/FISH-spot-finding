import os
from skimage import io
import numpy as np
import nd2
import glob

base_dir = "/Users/snk218/Dropbox/mac_files/holtlab/data_and_results/Farida_LINE1"
folder="20221030_LINE1"

param_file=f"{base_dir}/spot_counting_parameters"

for file in glob.glob(f"{base_dir}/{folder}/*.nd2"):
    file_root=os.path.splitext(os.path.split(file)[1])[0]

    f = nd2.ND2File(file)
    images = f.asarray()
    f.close()

    tiff_stack = np.zeros(shape=[images.shape[0], images.shape[2], images.shape[3], 3], dtype=images.dtype)

    for i in range(images.shape[0]):
        cur_img = np.asarray(images[i])
        tiff_stack[i] = np.stack([cur_img[0],cur_img[3],cur_img[1]],axis=-1)

    #print(tiff_stack.shape)
    io.imsave(f"{base_dir}/{folder}/tifs/{file_root}.tif", tiff_stack)
    io.imsave(f"{base_dir}/{folder}/tifs/{file_root}-BF.tif", images[:,4,:,:])


    #spot_img = np.amax(images[:,0,:,:],0)
    #nucl_img = np.round(np.mean(images[:,1,:,:],0)).astype(images.dtype)

    #io.imsave(f"{base_dir}/{folder}/tifs/{file_root}.tif", images)

    #full_img = np.stack([spot_img,np.zeros_like(spot_img),nucl_img],axis=2)

    #io.imsave(f"{base_dir}/{folder}/tifs/{file_root}-ch0.tif", spot_img)
    #io.imsave(f"{base_dir}/{folder}/tifs/{file_root}-ch1.tif", nucl_img)

    #io.imsave(f"{base_dir}/{folder}/tifs/{file_root}-zproj.tif", full_img)

    #f.close()
