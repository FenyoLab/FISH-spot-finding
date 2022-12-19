"""
This script will count bright spots that are inside nuclei
For each image in folder:
(1) Identify nuclei from mean z-projection
(2) Identify spots (by blob detection) - from max z-projection
(3) Get count of spots/nuclei
(4) Output summary file
(5) Output each image with nuclei and spots outlined - so these can be examined for correctness
"""

import os
from skimage import io, exposure, segmentation, img_as_ubyte, morphology, feature, draw
import numpy as np
import pandas as pd
from util import helpers
import warnings
import glob
import cv2
import matplotlib.pyplot as plt
from nucl_id import identify_nuclei as idn
warnings.filterwarnings('ignore', message='.+ is a low contrast image', category=UserWarning)


add_blob_distance_labels=False

def detect_blobs(meas_img,
                 meas_img_displ,
                 nuc_props,
                 th_disk,
                 rescale_limits,
                 blob_params,
                 count_from_0,
                 file_pre='',
                 output_folder=''):
    nucl_spot_count = 0
    (min_row, min_col, max_row, max_col) = nuc_props.bbox
    obj_mask = nuc_props.image
    cur_img = meas_img[min_row:max_row, min_col:max_col]
    nucl_y, nucl_x = nuc_props.centroid

    cur_img_displ = helpers.reshape_to_rgb(cur_img)
    if (output_folder != ''):
        io.imsave(f"{output_folder}/orig_{file_pre}_{str(nuc_props.label)}.tif",
                  cur_img_displ)

    cur_img = helpers.do_rescale(cur_img, rescale_limits[0], rescale_limits[1])
    cur_img_displ = helpers.reshape_to_rgb(cur_img)
    if (output_folder != ''):
        io.imsave(f"{output_folder}/rescale_int_{file_pre}_{str(nuc_props.label)}.tif",
                  cur_img_displ)

    if (th_disk>0):
        # apply white top hat to subtract background/spots below a minimum size
        res = morphology.white_tophat(cur_img,
                                      footprint=morphology.disk(th_disk), )
        cur_img=cur_img-res
        res = exposure.rescale_intensity(res)
        cur_img = exposure.rescale_intensity(cur_img)

        if (output_folder != ''):
            io.imsave(f"{output_folder}/white_th_{file_pre}_{str(nuc_props.label)}.tif",
                      res)
            io.imsave(f"{output_folder}/white_th_rem_{file_pre}_{str(nuc_props.label)}.tif",
                      cur_img)
    cur_img_displ = helpers.reshape_to_rgb(cur_img)

    # blob detection
    blobs = feature.blob_log(cur_img,
                             min_sigma=blob_params[0],
                             max_sigma=blob_params[1],
                             num_sigma=blob_params[2],
                             threshold=blob_params[3],
                             overlap=blob_params[4],
                             exclude_border=0)  ## setting exclude_border to > 0 finds NO blobs ???

    # The radius of each blob is approximately sq-root of 2 * sigma
    nucl_spots = []
    for blob_i, blob in enumerate(blobs):
        radius = (2 * blob[2]) ** 0.5
        if (obj_mask[int(blob[0])][int(blob[1])]):
            nucl_spot_count += 1

            nucl_spots.append([nuc_props.label, nucl_x, nucl_y, blob_i, blob[0]+min_row, blob[1]+min_col, radius, -1, 0])

            # draw dot on entire image (easier to see)
            # draw_dot_on_img(meas_img_displ, blob[0]+min_row, blob[1]+min_col, [0, 0, 255], 0)
            rr, cc = draw.circle_perimeter(int(blob[0] + min_row), int(blob[1] + min_col), int(radius * 2),
                                               method='bresenham', shape=meas_img_displ.shape)
            meas_img_displ[rr, cc] = [0, 0, 255]

            # draw circle on nuclei image
            rr, cc = draw.circle_perimeter(int(blob[0]), int(blob[1]), int(radius),
                                           method='bresenham', shape=cur_img_displ.shape)
            cur_img_displ[rr, cc] = [0, 0, 255]

    # calculate nearest distance each blob to other blob within nuclei
    drawn_labels = []
    for spot_i, spot in enumerate(nucl_spots):
        min_d = -1
        min_d_id = -1
        for spot_j, spot_ in enumerate(nucl_spots):
            if (spot_i != spot_j):
                d = ((spot[4] - spot_[4]) ** 2 + (spot[5] - spot_[5]) ** 2) ** 0.5
                if ((min_d == -1) or (d < min_d)):
                    min_d = d
                    min_d_id = spot_[3]
        nucl_spots[spot_i][7] = min_d_id
        nucl_spots[spot_i][8] = min_d

        # draw spot dist to nearest as label
        if (not ({spot[3], min_d_id} in drawn_labels)):
            drawn_labels.append({spot[3], min_d_id})
            if(add_blob_distance_labels):
                cv2.putText(meas_img_displ,
                            str(np.round(min_d, 1)),
                            (int(spot[5]) + min_col + 5,
                             int(spot[4]) + min_row + 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            [0, 255, 0],
                            1,
                            cv2.LINE_AA)

    # add to running spot list
    if (count_from_0 and len(nucl_spots) == 0):  # add nuclei that have zero spots, if desired
        nucl_spots.append([nuc_props.label, nucl_x, nucl_y, -1, 0, 0, 0, -1, 0])

    if (output_folder != ''):
        io.imsave(f"{output_folder}/final_{file_pre}_{str(nuc_props.label)}.tif",
                  cur_img_displ)

    return nucl_spots

def count_spots(file_name, input_folder, output_folder, params,
                spot_ch=0, nucl_ch=2,
                save_extra_images=False, nucl_id_dir=''):
    if (params[input_folder]['nucl_id_contrast_enh_type'] == 'none'):
        DAPI_ce_perc = 0
    else:
        DAPI_ce_perc = float(params[input_folder]['nucl_id_ce_percentile'])

    rescale_intensity_perc = [params[input_folder]['ce_percentile_ll'], params[input_folder]['ce_percentile_ul']]

    if ('count_from_0' in params[input_folder]):
        count_from_0 = int(params[input_folder]['count_from_0'])
    else:
        count_from_0 = 0

    blob_th = float(params[input_folder]['blob_th'])
    spot_dist = int(params[input_folder]['spot_distance_cutoff'])
    if(params[input_folder]['white_tophat']):
        th_disk_size=params[input_folder]['tophat_disk_size']
    else:
        th_disk_size=0

    # load file - it is a 3D z-stack
    full_stack = io.imread(file_name)

    file_root = os.path.split(file_name)[1]
    base_file_name = os.path.splitext(file_root)[0]
    nucl_outline_file = base_file_name + '_nucl.tif'

    if(len(full_stack.shape)>3):
        GFP_img = np.amax(full_stack[:, :, :, spot_ch], 0)
        DAPI_img = np.amax(full_stack[:, :, :, nucl_ch], 0) #np.round(np.mean(full_stack[:, :, :, nucl_ch], 0)).astype(full_stack.dtype)
    else:
        DAPI_img = np.round(np.mean(full_stack, 0)).astype(full_stack.dtype)
        GFP_img = np.amax(full_stack, 0)

    DAPI_img = exposure.rescale_intensity(DAPI_img)
    GFP_img = exposure.rescale_intensity(GFP_img)

    DAPI_img=img_as_ubyte(DAPI_img)
    # GFP_img = img_as_ubyte(GFP_img)

    # identify nuclei - setup params
    nucl_id = idn.identify_nuclei_class()
    if (params[input_folder]['nucl_id_contrast_enh_type'] == 'none'):
        nucl_id.contrast_enh_type = ''
    else:
        nucl_id.contrast_enh_type = params[input_folder]['nucl_id_contrast_enh_type']

    nucl_id.contrast_enh_rescale_perc = params[input_folder]['nucl_id_ce_percentile']
    nucl_id.filter_type = 'median'
    nucl_id.filter_radius = params[input_folder]['nucl_id_med_filter_size']
    nucl_id.ws_use_erosion = False

    nucl_id.watershed = bool(params[input_folder]['nucl_id_watershed'])
    nucl_id.ws_gauss_filter_size = float(params[input_folder]['nucl_id_ws_gauss_sigma'])
    nucl_id.ws_local_min_distance = float(params[input_folder]['nucl_id_ws_min_dist'])
    nucl_id.min_solidity = float(params[input_folder]['nucl_id_min_solidity'])
    nucl_id.min_area = float(params[input_folder]['nucl_id_min_area'])
    nucl_id.max_area = float(params[input_folder]['nucl_id_max_area'])
    nucl_id.threshold_type = params[input_folder]['nucl_id_th']
    nucl_id.remove_edge = False ## TODO ***

    nucl_id.process_image(DAPI_img, nucl_id_dir)
    outline_img = np.copy(nucl_id.outline_img)

    io.imsave(f"{output_folder}/{nucl_outline_file[:-4]}_pre-filtered_mask_{str(DAPI_ce_perc)}.tif",
              nucl_id.orig_image_mask.astype('uint8') * 255)

    # label each object in outline image with its area, eccentricity and solidity
    (outline_img,
     areas,
     eccentricities,
     solidities,
     maj_ax_lens) = helpers.get_props_and_label(outline_img, nucl_id.nuclei_props)

    io.imsave(f"{output_folder}/{nucl_outline_file[:-4]}_{str(DAPI_ce_perc)}.tif", outline_img)

    # identify spots
    # for each identified nuclei, crop to bounding box
    spot_list = []

    meas_img=GFP_img
    meas_img_displ = segmentation.mark_boundaries(meas_img,
                                                  (nucl_id.nuclei_mask * nucl_id.labeled_clusters),
                                                  color=[1, 0, 0],
                                                  mode='inner')
    meas_img_displ = img_as_ubyte(meas_img_displ)
    meas_img_displ2 = img_as_ubyte(meas_img_displ).copy()

    if(save_extra_images):
        extra_img_dir=output_folder + '/nuclei'
    else:
        extra_img_dir=''

    for prop in nucl_id.nuclei_props:

        # detect blobs for each nuclei
        cur_spot_list=detect_blobs(meas_img,
                                   meas_img_displ,
                                   prop,
                                   th_disk_size,
                                   rescale_intensity_perc,
                                   (float(params[input_folder]['blob_min_sigma']),
                                    float(params[input_folder]['blob_max_sigma']),
                                    int(params[input_folder]['blob_num_sigma']),
                                    blob_th,
                                    float(params[input_folder]['blob_overlap'])),
                                   count_from_0,
                                   nucl_outline_file[:-4],
                                   extra_img_dir)
        spot_list.extend(cur_spot_list)

    params_txt = f"d{str(spot_dist)}_th{str(blob_th)}_resc{str(rescale_intensity_perc[0])}_{str(rescale_intensity_perc[1])}"
    io.imsave(f"{output_folder}/{base_file_name}_{params_txt}_blob_marked.tif", meas_img_displ)
    io.imsave(f"{output_folder}/{base_file_name}_{params_txt}_counts_marked.tif", meas_img_displ2)

    # save spot distances
    spot_cols = ['nuclei_label',
                 'nucl_x',
                 'nucl_y',
                 'spot_id',
                 'spot_x',
                 'spot_y',
                 'spot_r',
                 'dist_nearest_id',
                 'dist_nearest']

    cur_spot_df = pd.DataFrame(spot_list, columns=spot_cols)
    cur_spot_df['folder'] = input_folder
    cur_spot_df['file_name'] = file_root
    ext_cols = ['folder', 'file_name']
    ext_cols.extend(spot_cols)
    cur_spot_df = cur_spot_df[ext_cols]

    return (areas,
            eccentricities,
            solidities,
            maj_ax_lens,
            cur_spot_df)


def count_all_spots(base_dir, work_dir, param_file,
                    spot_ch=0, nucl_ch=1,
                    save_extra_images=False, nucl_id_dir=''):
    params = helpers.read_parameters_from_file(param_file)
    folders = list(params.keys())

    spot_df = pd.DataFrame()

    for folder_i, folder in enumerate(folders):
        if (not os.path.isdir(base_dir + '/' + folder)):
            print(folder, "not found, skipping...")
            continue
        print(folder)

        # set up output directory
        if (not os.path.exists(work_dir + '/' + folder)):
            os.mkdir(work_dir + '/' + folder)
        cur_work_dir = work_dir + '/' + folder

        if (save_extra_images):
            if (not os.path.exists(cur_work_dir + '/nuclei')):
                os.mkdir(cur_work_dir + '/nuclei')

        file_list = glob.glob(base_dir + '/' + folder + '/*.tif')
        if (len(file_list) == 0):
            print("No files found in folder.")
            continue

        all_nucl_areas = []
        all_nucl_eccentricities = []
        all_nucl_solidities = []
        all_nucl_maj_ax_lens = []

        for file_ in file_list:
            (areas,
             eccentricities,
             solidities,
             maj_ax_lens,
             cur_spot_df) = count_spots(file_,
                                        folder,
                                        cur_work_dir,
                                        params,
                                        spot_ch,
                                        nucl_ch,
                                        save_extra_images,
                                        nucl_id_dir)

            all_nucl_areas.extend(areas)
            all_nucl_eccentricities.extend(eccentricities)
            all_nucl_solidities.extend(solidities)
            all_nucl_maj_ax_lens.extend(maj_ax_lens)

            spot_df = spot_df.append(cur_spot_df)

        # make histogram of nuclei properties:
        plt.hist(all_nucl_areas, bins=np.arange(np.min(all_nucl_areas), np.max(all_nucl_areas), 50))
        plt.savefig(cur_work_dir + '/nuclei_areas_hist.pdf')
        plt.clf()
        plt.close()

        plt.hist(all_nucl_eccentricities,
                 bins=np.arange(np.min(all_nucl_eccentricities), np.max(all_nucl_eccentricities), 0.1))
        plt.savefig(cur_work_dir + '/nuclei_ecc_hist.pdf')
        plt.clf()
        plt.close()

        plt.hist(all_nucl_solidities,
                 bins=np.arange(np.min(all_nucl_solidities), np.max(all_nucl_solidities), 0.01))
        plt.savefig(cur_work_dir + '/nuclei_sol_hist.pdf')
        plt.clf()
        plt.close()

        plt.hist(all_nucl_maj_ax_lens,
                 bins=np.arange(np.min(all_nucl_maj_ax_lens), np.max(all_nucl_maj_ax_lens), 5))
        plt.savefig(cur_work_dir + '/nuclei_maj_ax_len_hist.pdf')
        plt.clf()
        plt.close()

    spot_df.index = range(len(spot_df))
    spot_df.to_csv(f"{work_dir}/all_data.txt", sep='\t')

def combine_spots(work_dir, param_file):
    params = helpers.read_parameters_from_file(param_file)
    folders = list(params.keys())

    #read all data and find valid spots (counting close-by spots as a single spot), get spot counts per nuclei
    df = pd.read_csv(work_dir + '/all_data.txt', sep='\t', index_col=0)

    df['valid_spot']=0
    df['spot_count']=0

    max_spots=10

    #get spot counts per nuclei, using spot distance cutoff to combine spots
    spot_counts={}
    for folder_i, folder in enumerate(folders):
        spot_counts[folder]=[]
        cur_df = df[df['folder']==folder]

        if ('count_from_0' in params[folder]):
            count_from_0 = int(params[folder]['count_from_0'])
        else:
            count_from_0 = 0

        nucl_per_spot_count=[]
        pre = f"d{str(int(params[folder]['spot_distance_cutoff']))}_"
        pre += f"th{str(float(params[folder]['blob_th']))}_"
        pre += f"resc{str(params[folder]['ce_percentile_ll'])}_{str(params[folder]['ce_percentile_ul'])}"

        for fn in cur_df['file_name'].unique():
            fn_df = cur_df[cur_df['file_name']==fn]
            hist_arr=[0,]*(max_spots+1)

            #open image to label with number of spots on nuclei
            img = io.imread(f"{work_dir}/{folder}/{str(fn[:-4])}_{pre}_counts_marked.tif")

            for nucl in fn_df['nuclei_label'].unique():
                cur_spots = fn_df[fn_df['nuclei_label']==nucl]

                valid_spot_list = []
                if(len(cur_spots)==1 and cur_spots.iloc[0]['spot_id']==-1):
                    num_valid_spots = 0 # no spots found for this nuclei
                else:
                    for row_i,row in enumerate(cur_spots.iterrows()):
                        data = row[1]
                        if(row_i > 0):
                            drop_spot=False
                            for valid_spot in valid_spot_list:
                                d=((data['spot_x']-valid_spot[1])**2+(data['spot_y']-valid_spot[2])**2)**0.5
                                if(d <= int(params[folder]['spot_distance_cutoff'])):
                                    drop_spot=True
                                    break
                            if(not drop_spot):
                                valid_spot_list.append((row[0], data['spot_x'], data['spot_y']))
                        else:
                            valid_spot_list.append((row[0], data['spot_x'], data['spot_y']))

                    # set those valid to 1
                    for valid_spot in valid_spot_list:
                        df.at[valid_spot[0], 'valid_spot'] = 1
                    num_valid_spots = len(valid_spot_list)

                for row_i, row in enumerate(cur_spots.iterrows()):
                    df.at[row[0], 'spot_count'] = num_valid_spots

                spot_counts[folder].append(num_valid_spots)

                if(num_valid_spots > max_spots):
                    hist_arr[max_spots]+=1
                else:
                    hist_arr[num_valid_spots]+=1

                text_size = 0.8
                text_c=[0,255,0]
                text_lw = 1
                cv2.putText(img,
                            str(len(valid_spot_list)),
                            (int(cur_spots.iloc[0]['nucl_x']),
                            int(cur_spots.iloc[0]['nucl_y'])),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            text_size,
                            text_c,
                            text_lw,
                            cv2.LINE_AA)
                for sp_row in cur_spots.iterrows():
                    sp=sp_row[1]
                    rr, cc = draw.circle_perimeter(int(sp['spot_x']), int(sp['spot_y']), int(sp['spot_r'] * 3),
                                                   method='bresenham', shape=img.shape)
                    img[rr, cc] = [0, 0, 255]

            hist_arr.insert(0,fn)
            nucl_per_spot_count.append(hist_arr)

            io.imsave(f"{work_dir}/{folder}/{str(fn[:-4])}_{pre}_counts_marked.tif", img)

        #save hist to csv
        cols = ['file_name']
        cols.extend(range(0, max_spots + 1))
        df_hist = pd.DataFrame(nucl_per_spot_count, columns=cols)
        if(not count_from_0): # drop the zero column
            df_hist.drop(labels=0,axis=1,inplace=True)
        df_hist=df_hist.sort_values(by='file_name')

        df_hist.to_csv(f"{work_dir}/{folder}/{pre}_counts_by_filename.txt", sep='\t')

    df.to_csv(f"{work_dir}/all_data_with_counts.txt", sep='\t')

    # plot spot counts per nuclei histogram
    spot_counts_all = []
    for folder_i, folder in enumerate(folders):

        if ('count_from_0' in params[folder]):
            count_from_0 = int(params[folder]['count_from_0'])
        else:
            count_from_0 = 0

        pre = f"d{str(int(params[folder]['spot_distance_cutoff']))}_"
        pre += f"th{str(float(params[folder]['blob_th']))}_"
        pre += f"resc{str(params[folder]['ce_percentile_ll'])}_{str(params[folder]['ce_percentile_ul'])}"

        if (len(spot_counts[folder]) > 0):
            if (count_from_0):
                min_ = 0
            else:
                min_ = 1

            plt.hist(spot_counts[folder], np.arange(min_, 8, .25), histtype='step')
            plt.savefig(f"{work_dir}/{folder}/spot_count_hist_{folder.replace('/', '-')}_{pre}.pdf")
            plt.clf()
            plt.close()

            plt.hist(spot_counts[folder],
                     np.arange(min_, 8, .25),
                     weights=np.ones(len(spot_counts[folder])) / len(spot_counts[folder]))
            #plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
            plt.ylim(0, 1)
            plt.savefig(f"{work_dir}/{folder}/spot_perc_hist_{folder.replace('/', '-')}_{pre}.pdf")
            plt.clf()
            plt.close()
            spot_counts_all.extend(spot_counts[folder])
        else:
            print("No count data for ", folder)

        # plot for all spots, (not only 'valid' spots), the distance-between-spots histogram
        for folder_i, folder in enumerate(folders):
            pre = f"d{str(int(params[folder]['spot_distance_cutoff']))}_"
            pre += f"th{str(float(params[folder]['blob_th']))}_"
            pre += f"resc{str(params[folder]['ce_percentile_ll'])}_{str(params[folder]['ce_percentile_ul'])}"

            cur_df = df[df['folder'] == folder]
            cur_df = cur_df[cur_df['dist_nearest'] > 0]
            if (len(cur_df) > 0):
                plt.hist(cur_df['dist_nearest'],
                               bins=np.arange(np.min(cur_df['dist_nearest']), np.max(cur_df['dist_nearest']), 1))
                plt.savefig(f"{work_dir}/{folder}/spot_dist_hist_{folder.replace('/', '-')}_{pre}.pdf")
                plt.clf()
                plt.close()
            else:
                print("No distance data for ", folder)


if __name__ == "__main__":
    #base_dir="/Users/snk218/Dropbox/mac_files/holtlab/data_and_results/Nestor_spot_detection/test-images_Sarah"
    base_dir="/Users/snk218/Dropbox/mac_files/holtlab/data_and_results/Farida_LINE1/spot_counting"
    work_dir="/Users/snk218/Dropbox/mac_files/holtlab/data_and_results/Farida_LINE1/spot_counting-output"
    param_file=base_dir + '/spot_counting_parameters.txt'

    count_all_spots(base_dir, work_dir, param_file,
                    spot_ch=0, nucl_ch=2,
                    save_extra_images=True, nucl_id_dir=work_dir+"/nucl_id")
    combine_spots(work_dir, param_file)