import os
import nd2
import glob
from tifffile import imread, imwrite

base_dir = "/Users/snk218/Dropbox (NYU Langone Health)/mac_files/holtlab/data_and_results/Farida_LINE1"
folder="20221030_LINE1"

param_file=f"{base_dir}/spot_counting_parameters"

for file in glob.glob(f"{base_dir}/{folder}/*.nd2"):
    file_root=os.path.splitext(os.path.split(file)[1])[0]

    f = nd2.ND2File(file)
    images = f.asarray()
    f.close()

    imwrite(f"{base_dir}/{folder}/tifs/{file_root}.tif",
            images,
            imagej=True,
            metadata={'axes': 'ZCYX'}
            )


