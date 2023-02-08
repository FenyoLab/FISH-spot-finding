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
import random
from sklearn.cluster import KMeans
from skimage import filters

def save_blobs_on_movie(blobs_df, movie, file_name):
    # on each slice, label blobs: combine as one movie
    if ("location" in blobs_df.columns):
        labels_nucl = np.zeros_like(movie)
        labels_cyt = np.zeros_like(movie)
        for row in blobs_df.iterrows():
            if(row[1].location == 'nuclear'):
                labels=labels_nucl
            elif(row[1].location == 'cytoplasmic'):
                labels=labels_cyt

            # draw circle on image
            rr, cc = draw.circle_perimeter(int(row[1]['row (y)']), int(row[1]['col (x)']), int(row[1]['radius']),
                                           shape=labels[0].shape)
            labels[int(row[1]['plane (z)']), rr, cc] = 65535
        ij_stack = np.stack([movie, labels_nucl, labels_cyt], axis=1)
    else:
        labels = np.zeros_like(movie)
        for row in blobs_df.iterrows():
            # draw circle on image
            rr, cc = draw.circle_perimeter(int(row[1]['row (y)']), int(row[1]['col (x)']), int(row[1]['radius']),
                                           shape=labels[0].shape)
            labels[int(row[1]['plane (z)']), rr, cc] = 65535
        ij_stack = np.stack([movie, labels], axis=1)

    imwrite(file_name, ij_stack, imagej=True, metadata={'axes': 'ZCYX'})

def get_roi_coords(rois, img_shape):

    roi_to_coords = {}

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
            roi_to_coords[key] = list(zip(rr,cc))

    return roi_to_coords


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


def find_spots(full_stack, rois, labels_file_name='',
               spot_ch=3, nucl_ch=1, intens_ch=0,
               blob_th=0.02, blob_th_rel=None,
               blob_min_s=1, blob_max_s=3):

    # (1) Detect blobs
    spot_stack = full_stack[:, spot_ch, :, :]
    spot_stack_rescl = exposure.rescale_intensity(spot_stack)

    print("Detecting blobs...")
    blobs = feature.blob_log(spot_stack_rescl, min_sigma=blob_min_s, max_sigma=blob_max_s,
                             num_sigma=(blob_max_s-blob_min_s+1),
                             threshold=blob_th, threshold_rel=blob_th_rel, overlap=0.5)

    print("Finished...")

    # (2) Pull out the nucleus channel and the channel for co-localization
    nuclei_stack = full_stack[:, nucl_ch, :, :]
    coloc_stack = full_stack[:, intens_ch, :, :]

    # (3) go through blobs, label with the ROI, get the intensity of nuclei channel and the other channel
    roi_mask, labels_dict = make_mask_from_rois(rois, spot_stack[0].shape)
    if(labels_file_name):
        io.imsave(labels_file_name, roi_mask)

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
        mean_intens3 = np.mean(spot_stack[int(p)][rr, cc])

        # which ROI?
        label = roi_mask[int(r)][int(c)]
        if (label > 0):
            roi_key = labels_dict[label]
        else:
            roi_key = ''

        blobs_arr.append([p, r, c, radius, mean_intens, mean_intens2, mean_intens3, label, roi_key])

    return pd.DataFrame(blobs_arr, columns=['plane (z)', 'row (y)', 'col (x)', 'radius',
                                            'nuclei_ch_intensity', 'coloc_ch_intensity', 'spot_ch_intensity',
                                            'label', 'roi'])


def place_spots(coords, n, spot_r, z_dist, type, th, nuclei_stack):

    random_spots_dict = {}
    coords_choice = coords.copy()
    num_spots_placed=0
    while(num_spots_placed < n):

        cur_coords = random.choice(coords_choice)
        r=cur_coords[0]
        c=cur_coords[1]

        # select z-channel for spot
        p = int(random.choice(z_dist.to_numpy()))

        if((p, r, c) in random_spots_dict):
            continue

        # check intensity for this location
        rr, cc = draw.disk((int(r), int(c)), int(spot_r), shape=nuclei_stack[0].shape)

        # get mean intensity for these coordinates at the specified z-level
        spot_intensity = np.mean(nuclei_stack[int(p)][rr, cc])

        if((type == 'nuclear' and spot_intensity > th) or (type == 'cytoplasmic' and spot_intensity <= th)):
            # place spot
            random_spots_dict[(p, r, c)] = 1
            coords_choice.remove(cur_coords)
            num_spots_placed+=1

    return random_spots_dict.keys()

def randomize_spots(full_stack, rois, real_spot_df, loc_th_df, spot_ch=3, nucl_ch=1, intens_ch=0):
    # read in the roi and get the list of roi coordinates.
    # draw X number of random spots on the roi region, where X is same as detected spot count for each roi
    # spot radius will be uniform: the 'size' of the spots as quantified by blob_log is almost always the same
    # get the intensity of nuclei and RNA channels for each spot and save to file

    nuclei_stack = full_stack[:, nucl_ch, :, :]
    coloc_stack = full_stack[:, intens_ch, :, :]
    spot_stack = full_stack[:, spot_ch, :, :]

    blobs_arr = []

    coords_dict = get_roi_coords(rois, nuclei_stack[0].shape)
    for roi in coords_dict.keys():
        roi_coords = coords_dict[roi]

        for loc in ['cytoplasmic','nuclear']:
            cur_spot_df = real_spot_df[(real_spot_df.roi==roi) & (real_spot_df.location==loc)]
            num_spots = cur_spot_df.shape[0]
            if(num_spots > 0):

                cutoff = loc_th_df[loc_th_df['roi']==roi].iloc[0]['th']
                radius = cur_spot_df[cur_spot_df.roi == roi]['radius'].mean()
                z_ch_dist = cur_spot_df[cur_spot_df.roi == roi]['plane (z)']

                spot_positions = place_spots(roi_coords, num_spots, radius, z_ch_dist, loc, cutoff, nuclei_stack)
                for (p,r,c) in spot_positions:
                    rr, cc = draw.disk((int(r), int(c)), int(radius), shape=nuclei_stack[0].shape)

                    # get mean intensity for these coordinates at the specified z-level
                    mean_intens = np.mean(nuclei_stack[int(p)][rr, cc])
                    mean_intens2 = np.mean(coloc_stack[int(p)][rr, cc])
                    mean_intens3 = np.mean(spot_stack[int(p)][rr, cc])

                    blobs_arr.append([p, r, c, radius, mean_intens, mean_intens2, mean_intens3, loc, roi])

    return pd.DataFrame(blobs_arr, columns=['plane (z)', 'row (y)', 'col (x)', 'radius',
                                            'nuclei_ch_intensity', 'coloc_ch_intensity', 'spot_ch_intensity',
                                            'location', 'roi'])


def locate_spots(spot_df):

    spot_df['location']=''
    output_arr = []
    for roi in spot_df.roi.unique():
        data = spot_df[spot_df.roi == roi]['nuclei_ch_intensity']

        if (len(data) > 1):
            kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(data.to_numpy().reshape(-1, 1))

            if (data[kmeans.labels_ == 0].max() < data[kmeans.labels_ == 1].min()):
                th1 = (data[kmeans.labels_ == 0].max() + data[kmeans.labels_ == 1].min()) / 2
            else:
                th1 = (data[kmeans.labels_ == 1].max() + data[kmeans.labels_ == 0].min()) / 2

            th2 = filters.threshold_otsu(data)

            final_th = (th1+th2)/2

            spot_df.loc[(spot_df.roi == roi) & (spot_df.nuclei_ch_intensity > final_th), 'location'] = 'nuclear'
            spot_df.loc[(spot_df.roi == roi) & (spot_df.nuclei_ch_intensity <= final_th), 'location'] = 'cytoplasmic'

            output_arr.append([roi, len(data), final_th, len(data[data > final_th]), len(data[data <= final_th])])
        else:
            output_arr.append([roi, len(data), 0, 0, 0])

    output_df = pd.DataFrame(output_arr, columns=["roi", "num_spots", "th", "num_nuclei_spots", "num_cyto_spots"])

    return spot_df, output_df


########################################################################################################################
home_dir = "/Users/sarahkeegan/Dropbox (NYU Langone Health)/mac_files"
img_dir = "holtlab/data_and_results/Farida_LINE1/spot_counting/20221030_LINE1/"

condition="dox" #"nodox"
input_dir = f"{home_dir}/{img_dir}/{condition}"
output_dir = f"{input_dir}/results"

roi_suffix = "RoiSet"

spot_type = "RNA" #"ORF1" #"RNA"
nuclei_channel = 1

if(spot_type == "RNA"):
    # for RNA (ch 0) spots:
    spot_channel = 0
    intensity_channel = 3
    th={'zstack_001': 0.02, 'zstack_002': 0.02, 'zstack_003': 0.02, 'zstack_004': 0.02, 'zstack_005': 0.005}
    min_sigma = {'zstack_001': 1, 'zstack_002': 1, 'zstack_003': 1, 'zstack_004': 1, 'zstack_005': 1}
    max_sigma = {'zstack_001': 3, 'zstack_002': 3, 'zstack_003': 3, 'zstack_004': 3, 'zstack_005': 3}
elif(spot_type == "ORF1"):
    # for ORF1 (ch 3) spots:
    spot_channel = 3
    intensity_channel = 0

    # *********************************************************************************************************
    th = {'zstack_001': 0.025, 'zstack_002': 0.03, 'zstack_003': 0.075, 'zstack_004': 0.08, 'zstack_005': 0.075}

    min_sigma = {'zstack_001': 1, 'zstack_002': 2, 'zstack_003': 1, 'zstack_004': 1, 'zstack_005': 1}
    max_sigma = {'zstack_001': 3, 'zstack_002': 3, 'zstack_003': 3, 'zstack_004': 2, 'zstack_005': 3}
else:
    raise Exception("Error: invalid spot_type.")
th2=None

movie_files = glob.glob(f"{input_dir}/*.tif")

full_df = pd.DataFrame()
full_random_df = pd.DataFrame()
full_loc_counts_df = pd.DataFrame()

for movie_file in movie_files:
    file_root = os.path.splitext(os.path.split(movie_file)[1])[0]
    rois = read_roi_zip(f"{input_dir}/{file_root}_{roi_suffix}.zip")
    full_stack = io.imread(f"{movie_file}")

    print(movie_file)
    #if(file_root != 'zstack_005'):
    #    continue

    # Save blobs on movie, color indicates location, if using
    spot_stack = full_stack[:, spot_channel, :, :]
    nuclei_stack = full_stack[:, nuclei_channel, :, :]

    # find spots
    blobs_df = find_spots(full_stack, rois, labels_file_name=f"{output_dir}/{file_root}-roi_labels.tif",
                          spot_ch=spot_channel, nucl_ch=nuclei_channel, intens_ch=intensity_channel,
                          blob_th=th[file_root], blob_th_rel=th2,
                          blob_min_s=min_sigma[file_root], blob_max_s=max_sigma[file_root])

    # use nucleus signal to determine spot location (cytoplasmic or nuclear)
    # this adds a few columns to the data frame indicating the threshold levels and the location
    blobs_df, loc_counts_df = locate_spots(blobs_df)

    # save blobs on movie for checking - if location column exists in data frame, they will be colored by location
    save_blobs_on_movie(blobs_df, spot_stack, file_name=f"{output_dir}/{file_root}-{spot_type}-marked_blobs.tif")
    save_blobs_on_movie(blobs_df, nuclei_stack, file_name=f"{output_dir}/{file_root}-{spot_type}-marked_blobs-nucl.tif")

    # randomize spots
    random_blobs_df = randomize_spots(full_stack, rois, blobs_df, loc_counts_df,
                                      spot_ch=spot_channel, nucl_ch=nuclei_channel, intens_ch=intensity_channel)

    # save random blobs on movie for checking - location only
    save_blobs_on_movie(random_blobs_df, nuclei_stack,
                        file_name=f"{output_dir}/{file_root}-{spot_type}-marked_random_blobs-nucl.tif")

    loc_counts_df['file_name'] = file_root
    random_blobs_df['file_name'] = file_root

    full_random_df = pd.concat([full_random_df, random_blobs_df], axis=0, ignore_index=True)
    full_loc_counts_df = pd.concat([full_loc_counts_df, loc_counts_df], axis=0, ignore_index=True)

    blobs_df['file_name'] = file_root
    full_df = pd.concat([full_df, blobs_df], axis=0, ignore_index=True)

# Filter spots not within an ROI
full_df=full_df[full_df['roi']!='']
full_df.index = range(len(full_df))

full_random_df=full_random_df[full_random_df['roi']!='']
full_random_df.index = range(len(full_random_df))

full_loc_counts_df=full_loc_counts_df[full_loc_counts_df['roi']!='']
full_loc_counts_df.index = range(len(full_loc_counts_df))


# Save final output files
full_df.to_csv(f"{output_dir}/{spot_type}-all_spots.txt", sep='\t')
full_random_df.to_csv(f"{output_dir}/{spot_type}-all_random_spots.txt", sep='\t')
full_loc_counts_df.to_csv(f"{output_dir}/{spot_type}-spot_counts.txt", sep='\t')





