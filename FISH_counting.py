"""
This script will count bright spots that are inside nuclei
For each image in folder:
(1) Identify nuclei - blue channels
(2) Identify spots (by blob detection) - green and red channels
(3) Get count of spots/nuclei
(4) Output summary file
(5) Output each image with nuclei and spots outlined - so these can be examined for correctness
"""

import glob
import os
from skimage import io, exposure, feature, img_as_ubyte, segmentation, draw, filters, morphology
import identify_nuclei as idn
import matplotlib.pyplot as plt
import numpy as np
from cv2 import cv2
import pandas as pd
from matplotlib.ticker import PercentFormatter ### new added
np.warnings.filterwarnings('ignore') ####new add

def read_parameters_from_file(file_name):
    params = {}

    df = pd.read_table(file_name, sep='\t')
    for row in df.iterrows():
        row = row[1]
        subfolder = row['Folder_name']
        params[subfolder]={}
        for col in df.columns:
            if(col != subfolder):
                params[subfolder][col]=row[col]
    return params

def get_props_and_label(img, props, text_c=[0, 255, 0]):
    areas=[]
    eccentricities=[]
    solidities=[]
    ax_lens=[]
    text_size=0.6
    text_lw=2
    label_w = 260  # note w/h depend on text size/lw, and length of string used to label (this is hard-coded here)
    label_h = 20

    # label each object in outline image with its area, eccentricity and solidity
    for prop in props:
        r, c = prop.centroid
        areas.append(prop.area)
        eccentricities.append(prop.eccentricity)
        solidities.append(prop.solidity)
        ax_lens.append(prop.major_axis_length)
        label = (str(prop.label)+': a=' + str(round(prop.area, 0)) +
                 ' e=' + str(round(prop.eccentricity, 2)) +
                 ' s=' + str(round(prop.solidity, 2)))
        if ((r + label_h) >= len(outline_img)): r_pos = r - ((r + label_h) - len(outline_img))
        else: r_pos = r
        if ((c + label_w) >= len(outline_img[0])): c_pos = c - ((c + label_w) - len(outline_img[0]))
        else: c_pos = c
        cv2.putText(img,label,(int(c_pos),int(r_pos)),cv2.FONT_HERSHEY_SIMPLEX,
                    text_size,text_c,text_lw,cv2.LINE_AA)

    return (img, areas, eccentricities, solidities, ax_lens)

def draw_dot_on_img(draw_img, row, col, color, thickness=0):
    row = int(row)
    col = int(col)
    if(thickness==0):
        draw_img[row][col] = color
    else:
        for cur_r in range(row-thickness, row+thickness+1, 1):
            for cur_c in range(col-thickness, col+thickness+1,1):
                if(cur_r >=0 and cur_c >= 0 and cur_r < len(draw_img) and cur_c < len(draw_img[0])):
                    draw_img[cur_r][cur_c] = color

def reshape_to_rgb(grey_img):
    #makes single color channel image into rgb
    ret_img = np.zeros(shape=[grey_img.shape[0],grey_img.shape[1],3], dtype='uint8')
    grey_img_=img_as_ubyte(grey_img)

    ret_img[:,:,0]=grey_img_
    ret_img[:, :, 1] = grey_img_
    ret_img[:, :, 2] = grey_img_
    return ret_img

save_extra_images=False # this is for spot detection, set to False unless testing
temp_dir='' # this is for the nuclei detection, leave blank unless testing
do_finding=True #set to True to find spots
plot_results_hists=True #set to True to make plots of the spot-finding results

base_dir=input()
work_dir=input()

#base_dir="/Volumes/Seagate Backup Plus Drive/Nazario/testing"
#work_dir="/Volumes/Seagate Backup Plus Drive/Nazario/testing_results"

params = read_parameters_from_file(base_dir + '/spot_counting_parameters.txt')
folders=list(params.keys())
if(do_finding):
    spot_df=pd.DataFrame()

    for folder_i, folder in enumerate(folders):
        if(not os.path.isdir(base_dir + '/' + folder)):
            print(folder, "not found, skipping...")
            continue
        print(folder)
        # these params are for the blob detection step, and are not generally used since applying tophat instead
        #rescale_intensity=True
        #use_median_filter=False
        #median_filter_r=2

        #set up output directory
        if (not os.path.exists(work_dir + '/' + folder)):
            os.mkdir(work_dir + '/' + folder)
        cur_work_dir = work_dir + '/' + folder

        blob_th_GFP = float(params[folder]['blob_th_GFP'])
        blob_th_RFP = float(params[folder]['blob_th_RFP'])
        spot_dist = int(params[folder]['spot_distance_cutoff'])
        if(params[folder]['nucl_id_contrast_enh_type'] == 'none'):
            DAPI_ce_perc = 0
        else:
            DAPI_ce_perc = float(params[folder]['nucl_id_ce_percentile'])

        #open 3 files, one for each channel
        file_list = glob.glob(base_dir + '/' + folder + '/*_DAPI.tif')
        all_nucl_areas=[]
        all_nucl_eccentricities=[]
        all_nucl_solidities=[]
        all_nucl_maj_ax_lens=[]
        for file_ in file_list:
            DAPI_file = os.path.split(file_)[1]
            base_file_name = DAPI_file[:-9]
            GFP_file = base_file_name+'_GFP.tif'
            RFP_file = base_file_name + '_RFP.tif'
            nucl_outline_file = base_file_name + '_nucl.tif'

            #load files - each file has 3 channels (2 are blank)
            try:
                DAPI_img = io.imread(base_dir + '/' + folder + '/' + DAPI_file)[:,:,2]
                GFP_img = io.imread(base_dir + '/' + folder + '/' + GFP_file)[:,:,1]
                RFP_img = io.imread(base_dir + '/' + folder + '/' + RFP_file)[:,:,0]
            except FileNotFoundError as e:
                print("GFP/RFP files not found for '",DAPI_file, "' in folder '", folder, "'")
                continue

            if (save_extra_images):
                if (not os.path.exists(cur_work_dir + '/nuclei')):
                    os.mkdir(cur_work_dir + '/nuclei')

            DAPI_img = exposure.rescale_intensity(DAPI_img)
            GFP_img = exposure.rescale_intensity(GFP_img)
            RFP_img = exposure.rescale_intensity(RFP_img)

            #identify nuclei in DAPI - setup params
            nucl_id = idn.identify_nuclei_class()
            if(params[folder]['nucl_id_contrast_enh_type'] == 'none'):
                nucl_id.contrast_enh_type = ''
            else:
                nucl_id.contrast_enh_type = params[folder]['nucl_id_contrast_enh_type']

            nucl_id.contrast_enh_rescale_perc = params[folder]['nucl_id_ce_percentile']
            nucl_id.filter_type = 'median'
            nucl_id.filter_radius = params[folder]['nucl_id_med_filter_size']
            nucl_id.ws_use_erosion = False

            nucl_id.watershed = bool(params[folder]['nucl_id_watershed'])
            nucl_id.ws_gauss_filter_size = float(params[folder]['nucl_id_ws_gauss_sigma'])
            nucl_id.ws_local_min_distance = float(params[folder]['nucl_id_ws_min_dist'])
            nucl_id.min_solidity = float(params[folder]['nucl_id_min_solidity'])
            nucl_id.min_area = float(params[folder]['nucl_id_min_area'])
            nucl_id.max_area = float(params[folder]['nucl_id_max_area'])
            nucl_id.threshold_type = params[folder]['nucl_id_th']
            nucl_id.remove_edge = True

            nucl_id.process_image(DAPI_img, temp_dir)
            outline_img = np.copy(nucl_id.outline_img)

            io.imsave(cur_work_dir + '/' + nucl_outline_file[:-4]+'_pre-filtered_mask_'+str(DAPI_ce_perc)+'.tif',
                      nucl_id.orig_image_mask.astype('uint8')*255)

            # label each object in outline image with its area, eccentricity and solidity
            (outline_img, areas, eccentricities, solidities, maj_ax_lens) = get_props_and_label(outline_img, nucl_id.nuclei_props)
            all_nucl_areas.extend(areas)
            all_nucl_eccentricities.extend(eccentricities)
            all_nucl_solidities.extend(solidities)
            all_nucl_maj_ax_lens.extend(maj_ax_lens)

            io.imsave(cur_work_dir + '/' + nucl_outline_file[:-4]+'_'+str(DAPI_ce_perc)+'.tif', outline_img)

            #identify spots
            #for each identified nuclei, crop to bounding box
            labels=['GFP','RFP']
            spot_list = []
            for meas_img_i,meas_img in enumerate([GFP_img,RFP_img]):
                meas_img_displ = segmentation.mark_boundaries(meas_img, (nucl_id.nuclei_mask*nucl_id.labeled_clusters),
                                                              color=[1, 0, 0], mode='inner')
                meas_img_displ = img_as_ubyte(meas_img_displ)
                meas_img_displ2 = img_as_ubyte(meas_img_displ).copy()

                #blob th can be different for GFP/RFP
                if(meas_img_i == 0):
                    blob_th=blob_th_GFP
                else:
                    blob_th=blob_th_RFP

                for prop in nucl_id.nuclei_props:
                    nucl_spot_count = 0
                    (min_row, min_col, max_row, max_col) = prop.bbox
                    obj_mask = prop.image
                    cur_img = meas_img[min_row:max_row,min_col:max_col]
                    nucl_y,nucl_x = prop.centroid

                    cur_img_displ = reshape_to_rgb(cur_img)
                    if(save_extra_images):
                        io.imsave(cur_work_dir + '/nuclei/orig_' + nucl_outline_file[:-4] + "_" + str(prop.label) +
                                  '_' + labels[meas_img_i] + '.tif', cur_img_displ)

                    if(params[folder]['white_tophat']):
                        # apply white top hat to subtract background/spots below a minimum size
                        cur_img = morphology.white_tophat(cur_img, selem=morphology.disk(params[folder]['tophat_disk_size']), )
                        cur_img = exposure.rescale_intensity(cur_img)
                        cur_img_displ = reshape_to_rgb(cur_img)
                        if (save_extra_images):
                            io.imsave(cur_work_dir + '/nuclei/white_th_' + nucl_outline_file[:-4] + "_" + str(prop.label) +
                                      '_' + labels[meas_img_i] + '.tif', cur_img_displ)

                    # if(False): #rescale_intensity):
                    #     cur_img = exposure.rescale_intensity(cur_img)
                    #     cur_img_displ = reshape_to_rgb(cur_img)
                    #     io.imsave(base_dir + '/' + folder + '/nuclei/plain_' + nucl_outline_file[:-4] + "_" + str(prop.label) +
                    #               '_' + labels[meas_img_i] + '.tif', cur_img_displ)

                    # if(False): #use_median_filter):
                    #     cur_img = filters.rank.median(cur_img, morphology.disk(median_filter_r))
                    #     cur_img_displ = reshape_to_rgb(cur_img)
                    #     io.imsave(base_dir + '/' + folder + '/nuclei/plain_med_' + nucl_outline_file[:-4] + "_" + str(prop.label) +
                    #           '_' + labels[meas_img_i] + '.tif', cur_img_displ)

                    # blob detection
                    blobs=feature.blob_log(cur_img, min_sigma=float(params[folder]['blob_min_sigma']),
                                           max_sigma=float(params[folder]['blob_max_sigma']),
                                           num_sigma=int(params[folder]['blob_num_sigma']),
                                           threshold=blob_th,
                                           overlap=float(params[folder]['blob_overlap']),
                                           exclude_border=0) ## setting exclude_border to > 0 finds NO blobs ???
                    # The radius of each blob is approximately sq-root of 2 * sigma
                    nucl_spots=[]
                    for blob_i,blob in enumerate(blobs):
                        radius=(2*blob[2])**0.5
                        if(obj_mask[int(blob[0])][int(blob[1])]):
                            nucl_spot_count += 1

                            nucl_spots.append([labels[meas_img_i],prop.label,nucl_x,nucl_y,blob_i,blob[0],blob[1],radius,-1,0])

                            #draw dot on entire image (easier to see)
                            #draw_dot_on_img(meas_img_displ, blob[0]+min_row, blob[1]+min_col, [0, 0, 255], 0)
                            rr, cc = draw.circle_perimeter(int(blob[0]+min_row), int(blob[1]+min_col), int(radius*3),
                                                           method='bresenham', shape=meas_img_displ.shape)
                            meas_img_displ[rr, cc] = [0, 0, 255]

                            #draw circle on nuclei image
                            rr, cc = draw.circle_perimeter(int(blob[0]), int(blob[1]), int(radius),
                                                           method='bresenham', shape=cur_img_displ.shape)
                            cur_img_displ[rr, cc] = [0, 0, 255]

                    #calculate nearest distance each blob to other blob within nuclei
                    drawn_labels=[]
                    for spot_i,spot in enumerate(nucl_spots):
                        min_d = -1
                        min_d_id = -1
                        for spot_j,spot_ in enumerate(nucl_spots):
                            if(spot_i!=spot_j):
                                d=((spot[5]-spot_[5])**2 + (spot[6]-spot_[6])**2)**0.5
                                if((min_d==-1) or (d < min_d)):
                                    min_d=d
                                    min_d_id=spot_[4]
                        nucl_spots[spot_i][8]=min_d_id
                        nucl_spots[spot_i][9] = min_d

                        #draw spot dist to nearest as label
                        if(not ({spot[4],min_d_id} in drawn_labels)):
                            drawn_labels.append({spot[4],min_d_id})
                            cv2.putText(meas_img_displ, str(np.round(min_d,1)), (int(spot[6])+min_col+5, int(spot[5])+min_row+5), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4, [0,255,0], 1, cv2.LINE_AA)

                    # add to running spot list
                    spot_list.extend(nucl_spots)
                    if (save_extra_images):
                        io.imsave(cur_work_dir + '/nuclei/' + nucl_outline_file[:-4] + "_" + str(prop.label) +
                              '_' + labels[meas_img_i] + '_'+str(spot_dist)+'_' +str(blob_th)+'_'+str(DAPI_ce_perc)+
                                  '.tif', cur_img_displ)

                if(meas_img_i==0):
                    if(save_extra_images):
                        io.imsave(cur_work_dir + '/' + GFP_file[:-4]+'_'+str(spot_dist)+'_' +str(blob_th)+'_'+
                                  str(DAPI_ce_perc)+'_blob_marked.tif', meas_img_displ)
                    io.imsave(cur_work_dir + '/' + GFP_file[:-4] + '_'+str(spot_dist)+'_' +str(blob_th)+'_'+
                              str(DAPI_ce_perc)+'_counts_marked.tif', meas_img_displ2)
                else:
                    if (save_extra_images):
                        io.imsave(cur_work_dir + '/' + RFP_file[:-4] + '_'+str(spot_dist)+'_' +str(blob_th)+'_'+
                                  str(DAPI_ce_perc)+'_blob_marked.tif', meas_img_displ)
                    io.imsave(cur_work_dir + '/' + RFP_file[:-4] + '_'+str(spot_dist)+'_' +str(blob_th)+'_'+
                              str(DAPI_ce_perc)+'_counts_marked.tif', meas_img_displ2)

            #save spot distances
            spot_cols=['type','nuclei_label','nucl_x','nucl_y','spot_id','spot_x','spot_y','spot_r','dist_nearest_id','dist_nearest']
            cur_spot_df = pd.DataFrame(spot_list, columns=spot_cols)
            cur_spot_df['folder']=folder
            cur_spot_df['file_name']=base_file_name
            ext_cols=['folder','file_name']
            ext_cols.extend(spot_cols)
            cur_spot_df = cur_spot_df[ext_cols]
            spot_df = spot_df.append(cur_spot_df)

        #make histogram of nuclei properties:
        plt.hist(all_nucl_areas, bins=np.arange(np.min(all_nucl_areas), np.max(all_nucl_areas), 50))
        plt.savefig(cur_work_dir + '/' + str(DAPI_ce_perc) + '_nuclei_areas_hist.pdf')
        plt.clf()

        plt.hist(all_nucl_eccentricities, bins=np.arange(np.min(all_nucl_eccentricities), np.max(all_nucl_eccentricities), 0.1))
        plt.savefig(cur_work_dir + '/' + str(DAPI_ce_perc)+ '_nuclei_ecc_hist.pdf')
        plt.clf()

        plt.hist(all_nucl_solidities, bins=np.arange(np.min(all_nucl_solidities), np.max(all_nucl_solidities), 0.01))
        plt.savefig(cur_work_dir + '/' + str(DAPI_ce_perc) + '_nuclei_sol_hist.pdf')
        plt.clf()

        plt.hist(all_nucl_maj_ax_lens, bins=np.arange(np.min(all_nucl_maj_ax_lens), np.max(all_nucl_maj_ax_lens), 5))
        plt.savefig(cur_work_dir + '/' + str(DAPI_ce_perc) + '_nuclei_maj_ax_len_hist.pdf')
        plt.clf()

    # ** NOTE: spot_dist and blob_th_GFP/RFP are set separately for each folder
    # if there is > 1 folder listed in spot_counting_paramters.txt, the name of the file below
    # will contain the spot_dist and blob_th_GFP/RFP for the LAST folder that is processed. **
    spot_df.index=range(len(spot_df))
    spot_df.to_csv(work_dir + '/'+str(spot_dist)+'_GFP_' +str(blob_th_GFP)+'_RFP_'+str(blob_th_RFP)+'_'+
                   str(DAPI_ce_perc)+'_all_data.txt', sep='\t')

if(plot_results_hists):
    print("Processing results...")
    #read all data and find valid spots (counting close-by spots as a single spot), get spot counts per nuclei
    folder=folders[len(folders)-1] #file was saved with parameters for last folder in list
    last_blob_th_GFP = float(params[folder]['blob_th_GFP'])
    last_blob_th_RFP = float(params[folder]['blob_th_RFP'])
    last_spot_dist = int(params[folder]['spot_distance_cutoff'])
    if (params[folder]['nucl_id_contrast_enh_type'] == 'none'): last_DAPI_ce_perc = 0
    else: last_DAPI_ce_perc = float(params[folder]['nucl_id_ce_percentile'])
    df = pd.read_csv(work_dir + '/'+str(last_spot_dist)+'_GFP_' +str(last_blob_th_GFP)+'_RFP_'+str(last_blob_th_RFP)+
                     '_'+str(last_DAPI_ce_perc)+'_all_data.txt', sep='\t', index_col=0)

    df['valid_spot']=0
    df['spot_count']=0

    max_spots={}
    max_spots['RFP']=6
    max_spots['GFP']=8

    #get spot counts per nuclei, using spot distance cutoff to combine spots
    spot_counts={}
    for folder_i, folder in enumerate(folders):
        print(folder)
        spot_counts[folder]={}
        cur_df = df[df['folder']==folder]
        spot_dist=int(params[folder]['spot_distance_cutoff'])
        blob_th_GFP = float(params[folder]['blob_th_GFP'])
        blob_th_RFP = float(params[folder]['blob_th_RFP'])
        if (params[folder]['nucl_id_contrast_enh_type'] == 'none'): DAPI_ce_perc=0
        else: DAPI_ce_perc = float(params[folder]['nucl_id_ce_percentile'])
        nucl_per_spot_count={}
        for type in df['type'].unique(): # type: GFP or RFP
            if(type == 'GFP'):
                blob_th = blob_th_GFP
            else:
                blob_th = blob_th_RFP
            nucl_per_spot_count[type] = []
            spot_counts[folder][type]=[]
            type_df = cur_df[cur_df['type'] == type]
            for fn in type_df['file_name'].unique():
                fn_df = type_df[type_df['file_name']==fn]
                hist_arr=[0,]*max_spots[type]

                #open image to label with number of spots on nuclei
                img = io.imread(work_dir + '/' + folder + '/' + str(fn) + '_' + type +'_'+str(spot_dist)+'_' +
                                str(blob_th)+'_'+str(DAPI_ce_perc)+'_counts_marked.tif')
                for nucl in fn_df['nuclei_label'].unique():
                    cur_spots = fn_df[fn_df['nuclei_label']==nucl]

                    valid_spot_list = []
                    for row_i,row in enumerate(cur_spots.iterrows()):
                        data = row[1]
                        if(row_i > 0):
                            drop_spot=False
                            for valid_spot in valid_spot_list:
                                d=((data['spot_x']-valid_spot[1])**2+(data['spot_y']-valid_spot[2])**2)**0.5
                                if(d <= spot_dist):
                                    drop_spot=True
                                    break
                            if(not drop_spot):
                                valid_spot_list.append((row[0], data['spot_x'], data['spot_y']))
                        else:
                            valid_spot_list.append((row[0], data['spot_x'], data['spot_y']))
                    #set those valid to 1
                    for valid_spot in valid_spot_list:
                        df.at[valid_spot[0], 'valid_spot'] = 1

                    num_valid_spots = len(valid_spot_list)
                    df['spot_count']=np.where(df['nuclei_label']==nucl, num_valid_spots, df['spot_count'])
                    spot_counts[folder][type].append(num_valid_spots)

                    if(num_valid_spots > 0):
                        if(num_valid_spots > max_spots[type]):
                            hist_arr[max_spots[type]-1]+=1
                        else:
                            hist_arr[num_valid_spots-1]+=1

                    text_size = 0.8
                    text_c=[0,255,0]
                    text_lw = 1
                    cv2.putText(img, str(len(valid_spot_list)), (int(cur_spots.iloc[0]['nucl_x']),
                                int(cur_spots.iloc[0]['nucl_y'])), cv2.FONT_HERSHEY_SIMPLEX,
                                text_size, text_c, text_lw, cv2.LINE_AA)
                hist_arr.insert(0,fn)
                nucl_per_spot_count[type].append(hist_arr)
                io.imsave(work_dir + '/' + folder + '/' + str(fn) + '_' + type +'_'+str(spot_dist)+'_' +str(blob_th)+
                          '_'+str(DAPI_ce_perc)+'_counts_marked.tif', img)

        #save hist to csv
        for type in df['type'].unique():
            if (type == 'GFP'):
                blob_th = blob_th_GFP
            else:
                blob_th = blob_th_RFP
            cols=['file_name']
            cols.extend(range(1,max_spots[type]+1))
            df_hist = pd.DataFrame(nucl_per_spot_count[type], columns=cols)
            df_hist=df_hist.sort_values(by='file_name')
            df_hist.to_csv(work_dir + '/' + folder + '/' + type+'_'+str(spot_dist)+'_' +str(blob_th)+'_'+
                           str(DAPI_ce_perc)+'_counts_by_filename.txt', sep='\t')
    df.to_csv(work_dir + '/'+str(last_spot_dist)+'_GFP_' + str(last_blob_th_GFP) + '_RFP_' + str(last_blob_th_RFP)+'_'+
              str(last_DAPI_ce_perc)+'_all_data_with_counts.txt', sep='\t')

    #plot spot counts per nuclei histogram, for RFP and GFP
    spot_counts_all={}
    for type in df['type'].unique():
        spot_counts_all[type]=[]
    for folder_i, folder in enumerate(folders):
        spot_dist = int(params[folder]['spot_distance_cutoff'])
        if (params[folder]['nucl_id_contrast_enh_type'] == 'none'): DAPI_ce_perc=0
        else: DAPI_ce_perc = float(params[folder]['nucl_id_ce_percentile'])
        for type in spot_counts[folder].keys():
            if (type == 'GFP'):
                blob_th = blob_th_GFP
            else:
                blob_th = blob_th_RFP
            if(len(spot_counts[folder][type])>0):
                ret=plt.hist(spot_counts[folder][type], np.arange(0, 8, .25), histtype='step')
                plt.savefig(work_dir + '/' + folder + '/spot_count_hist_' + type + '_' + folder.replace('/','-') +
                            '_'+str(spot_dist)+ '_' +str(blob_th)+'_'+str(DAPI_ce_perc)+'.pdf')
                plt.clf()
                percent=plt.hist(spot_counts[folder][type], np.arange(0, 8, .25),
                                 weights=np.ones(len(spot_counts[folder][type])) / len(spot_counts[folder][type])) ### new added
                plt.gca().yaxis.set_major_formatter(PercentFormatter(1)) #### new added
                plt.ylim(0,1)
                plt.savefig(work_dir + '/' + folder + '/spot_perc_hist_' + type + '_' + folder.replace('/','-') +
                            '_'+str(spot_dist)+ '_' +str(blob_th)+'_'+str(DAPI_ce_perc)+'.pdf')###### new added
                plt.clf()####### new added
                spot_counts_all[type].extend(spot_counts[folder][type])
            else:
                print("No count data for ", folder, " ", type)

    #plot combined histogram for all folders
    # for type in spot_counts_all.keys():
    #     ret=plt.hist(spot_counts_all[type], np.arange(0,8,0.25), histtype='step')
    #     plt.savefig(base_dir + '/spot_count_hist_' + type + '_all_' + str(spot_dist) + '.pdf')
    #     plt.clf()

    #plot for all spots, (not only 'valid' spots), the distance-between-spots histogram
    for folder_i, folder in enumerate(folders):
        spot_dist = int(params[folder]['spot_distance_cutoff'])
        if (params[folder]['nucl_id_contrast_enh_type'] == 'none'): DAPI_ce_perc=0
        else: DAPI_ce_perc = float(params[folder]['nucl_id_ce_percentile'])
        cur_df = df[df['folder']==folder]
        for type in df['type'].unique():
            if (type == 'GFP'):
                blob_th = blob_th_GFP
            else:
                blob_th = blob_th_RFP
            type_df = cur_df[cur_df['type'] == type]
            type_df = type_df[type_df['dist_nearest']>0]
            if(len(type_df)>0):
                ret=plt.hist(type_df['dist_nearest'], bins=np.arange(np.min(type_df['dist_nearest']), np.max(type_df['dist_nearest']), 1))
                plt.savefig(work_dir + '/' + folder + '/spot_dist_hist_'+type+'_'+folder.replace('/', '-') +
                            '_'+str(spot_dist)+'_' +str(blob_th)+'_'+str(DAPI_ce_perc)+'.pdf')
                plt.clf()
            else:
                print("No distance data for ", folder, " ", type)



