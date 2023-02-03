# read in the roi file and separate channel for nuclei detection
# for each roi, get bounding box, apply threshold, should be one object: nucleus
# count spots in the nucleus and in the cytoplasm

# 3D spot finding
from read_roi import read_roi_zip
from skimage import feature, io, exposure, draw
import numpy as np
from tifffile import imwrite
import glob
import os
import pandas as pd


def save_blobs_on_movie(blobs_list, movie, file_name):
    # on each slice, label blobs: combine as one movie
    labels = np.zeros_like(movie)
    for p, r, c, sigma in blobs_list:
        radius = 2 * np.sqrt(sigma)

        # draw circle on image
        rr, cc = draw.circle_perimeter(int(r), int(c), int(radius), shape=labels[0].shape)
        labels[int(p), rr, cc] = 65535

    ij_stack = np.stack([movie, labels], axis=1)

    imwrite(file_name, ij_stack, imagej=True, metadata={'axes': 'ZCYX'})


def make_mask_from_rois(rois, img_shape):
    final_img = np.zeros(img_shape, dtype='uint8')
    label = 1
    label_to_roi = {}

    for key in rois.keys():
        unknown_roi = False
        roi = rois[key]

        if (roi['type'] == 'polygon' or
                (roi['type'] == 'freehand' and 'x' in roi and 'y' in roi) or
                (roi['type'] == 'traced' and 'x' in roi and 'y' in roi)):

            col_coords = roi['x']
            row_coords = roi['y']
            rr, cc = draw.polygon(row_coords, col_coords, shape=img_shape)

        elif (roi['type'] == 'rectangle'):

            rr, cc = draw.rectangle((roi['top'], roi['left']),
                                    extent=(roi['height'], roi['width']),
                                    shape=img_shape)
            rr = rr.astype('int')
            cc = cc.astype('int')

        elif (roi['type'] == 'oval'):

            rr, cc = draw.ellipse(roi['top'] + roi['height'] / 2,
                                  roi['left'] + roi['width'] / 2,
                                  roi['height'] / 2,
                                  roi['width'] / 2,
                                  shape=img_shape)
        else:
            unknown_roi = True

        if (not unknown_roi):
            final_img[rr, cc] = label
            label_to_roi[label] = key
            label += 1

    return final_img, label_to_roi


def find_spots(full_stack, rois, save_pre='',
               spot_ch=0, nucl_ch=1, coloc_ch=3,
               blob_th=0.02, blob_th_rel=None,
               blob_min_s=1, blob_max_s=3):

    # (1) Detect blobs
    spot_stack = full_stack[:, spot_ch, :, :]
    spot_stack = exposure.rescale_intensity(spot_stack)

    print("Detecting blobs...")
    blobs = feature.blob_log(spot_stack, min_sigma=blob_min_s, max_sigma=blob_max_s,
                             num_sigma=(blob_max_s-blob_min_s+1),
                             threshold=blob_th, threshold_rel=blob_th_rel, overlap=0.5)
    save_blobs_on_movie(blobs, spot_stack, f"{save_pre}marked_blobs.tif")

    print("Finished...")

    # (2) Pull out the nucleus channel and the channel for co-localization
    nuclei_stack = full_stack[:, nucl_ch, :, :]
    coloc_stack = full_stack[:, coloc_ch, :, :]

    # (3) go through blobs, label with the ROI, get the intensity of nuclei channel and the other channel
    roi_mask, labels_dict = make_mask_from_rois(rois, spot_stack[0].shape)
    io.imsave(f"{save_pre}roi_labels.tif", roi_mask)

    blobs = blobs[np.argsort(blobs[:, 0])].copy()
    blobs_arr=[]
    for p, r, c, sigma in blobs:
        radius = 2 * np.sqrt(sigma)

        # get 2d blob coordinates (as a disk)
        rr, cc = draw.disk((int(r), int(c)), int(radius), shape=spot_stack[0].shape)

        # get mean intensity for these coordinates at the specified z-level
        # simplification here since only taking mean intensity at one z-level
        mean_intens = np.mean(nuclei_stack[int(p)][rr, cc])

        mean_intens2 = np.mean(coloc_stack[int(p)][rr, cc])

        # which ROI?
        label = roi_mask[int(r)][int(c)]
        if (label > 0):
            roi_key = labels_dict[label]
        else:
            roi_key = ''

        blobs_arr.append([p, r, c, radius, mean_intens, mean_intens2, label, roi_key])

    return pd.DataFrame(blobs_arr, columns=['plane (z)', 'row (y)', 'col (x)', 'radius',
                                            'nuclei_ch_intensity', 'coloc_ch_intensity', 'label', 'roi'])


########################################################################################################################
home_dir = "/Users/snk218/Dropbox (NYU Langone Health)/mac_files"
img_dir = "holtlab/data_and_results/Farida_LINE1/spot_counting/20221030_LINE1/"

condition="no_dox" #"dox"
input_dir = f"{home_dir}/{img_dir}/{condition}"
output_dir = f"{input_dir}/results"

roi_suffix = "RoiSet"

if(condition == "dox"):
    th_settings={'zstack_001': 0.02, 'zstack_002': 0.02, 'zstack_003': 0.02, 'zstack_004': 0.02, 'zstack_005': 0.005}
    my_th2=None
    min_sigma=1
    max_sigma=3
else:
    th_settings = {'zstack_nodox_006': 0.02, 'zstack_nodox_007': 0.02, 'zstack_nodox_008': 0.02}
    my_th2=0.1
    min_sigma = 2
    max_sigma = 3

movie_files = glob.glob(f"{input_dir}/*.tif")
full_df = pd.DataFrame()
for movie_file in movie_files:
    file_root = os.path.splitext(os.path.split(movie_file)[1])[0]
    rois = read_roi_zip(f"{input_dir}/{file_root}_{roi_suffix}.zip")
    full_stack = io.imread(f"{movie_file}")

    print(movie_file)
    #if(file_root != 'zstack_005'):
    #    continue
    blobs_df = find_spots(full_stack, rois, save_pre=f"{output_dir}/{file_root}-",
                          blob_th=th_settings[file_root], blob_th_rel=my_th2,
                          blob_min_s=min_sigma, blob_max_s=max_sigma)
    blobs_df['file_name'] = file_root

    full_df = pd.concat([full_df, blobs_df], axis=0, ignore_index=True)

full_df.to_csv(f"{output_dir}/all_spots.txt", sep='\t')



if(False):

    def make_mask_from_roi(rois, roi_name, img_shape):
        # set interior of selected ROI to 1
        final_img = np.zeros(img_shape, dtype='uint8')
        for key in rois.keys():
            if (key == roi_name):
                roi = rois[key]

                if (roi['type'] == 'polygon' or
                        (roi['type'] == 'freehand' and 'x' in roi and 'y' in roi) or
                        (roi['type'] == 'traced' and 'x' in roi and 'y' in roi)):

                    col_coords = roi['x']
                    row_coords = roi['y']
                    rr, cc = draw.polygon(row_coords, col_coords, shape=img_shape)
                    final_img[rr, cc] = 1
                    bbox = (row_coords.min(), col_coords.min(), row_coords.max(), col_coords.max())

                elif (roi['type'] == 'rectangle'):

                    rr, cc = draw.rectangle((roi['top'], roi['left']),
                                            extent=(roi['height'], roi['width']),
                                            shape=img_shape)
                    rr = rr.astype('int')
                    cc = cc.astype('int')
                    final_img[rr, cc] = 1
                    bbox = (roi['top'], roi['left'], roi['top'] + roi['height'], roi['left'] + roi['width'])

                elif (roi['type'] == 'oval'):

                    rr, cc = draw.ellipse(roi['top'] + roi['height'] / 2,
                                          roi['left'] + roi['width'] / 2,
                                          roi['height'] / 2,
                                          roi['width'] / 2,
                                          shape=img_shape)
                    final_img[rr, cc] = 1
                    bbox = (roi['top'], roi['left'], roi['top'] + roi['height'], roi['left'] + roi['width'])

                else:
                    print("Unknown roi type.")

                break

        return final_img, bbox



    for movie_file in movie_files:
        file_root = os.path.splitext(os.path.split(movie_file)[1])[0]
        rois = read_roi_zip(f"{home_dir}/{img_dir}/{file_root}_RoiSet.zip")
        full_stack = io.imread(f"{movie_file}")

        print(movie_file)

        # (1) Detect blobs
        spot_stack = full_stack[:, spot_ch, :, :]
        spot_stack = exposure.rescale_intensity(spot_stack)
        blobs = feature.blob_log(spot_stack, min_sigma=1, max_sigma=3, num_sigma=10, threshold=0.01, overlap=0.5)
        save_blobs_on_movie(blobs, spot_stack, f"{home_dir}/{img_dir}/results/{file_root}-marked_blobs.tif")

        # (2) Pull out the nucleus channel and max z-project
        nuclei_stack = full_stack[:, nucl_ch, :, :]
        nuclei_zproj = np.max(full_stack, axis=0)

        # (3) read ROIs, segment blobs to ROI, and to nucleus within ROI
        for key in rois:
            print(key)
            roi_mask, roi_bbox = make_mask_from_roi(rois, key, spot_stack[0].shape)

            # segment nuclei only within roi bbox
            nuclei_image = nuclei_zproj[roi_bbox[0]:roi_bbox[2], roi_bbox[1]:roi_bbox[3]]

            # saturate 6% of pixels (3% on top and 3% on bottom)
            saturate_perc = 3
            p1, p2 = np.percentile(nuclei_image, (saturate_perc, 100 - saturate_perc))
            nuclei_image = exposure.rescale_intensity(nuclei_image, in_range=(p1, p2))

            # median filter
            disk_size = 4
            nuclei_image = filters.median(nuclei_image, morphology.disk(disk_size))

            # threshold, fill holes
            otsu_th = filters.threshold_otsu(nuclei_image)
            nuclei_mask = nuclei_image > otsu_th
            nuclei_mask = binary_fill_holes(nuclei_mask)

            # label, take largest object
            labeled_image = measure.label(nuclei_mask)
            props_table = measure.regionprops_table(labeled_image, properties=('label', 'bbox', 'area', 'coords'))
            props_table.sort_values(by='area', ascending=False, inplace=True)

            # fix coords back to original image (currently relative to bbox)
            rr,cc = props_table.iloc[0]['coords']
            rr = rr + roi_bbox[0]
            cc = cc + roi_bbox[1]










