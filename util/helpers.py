import pandas as pd
from skimage import exposure, img_as_ubyte
import numpy as np
import cv2
import warnings
from skimage import draw
warnings.filterwarnings('ignore', message='.+ is a low contrast image', category=UserWarning)

def make_mask_from_roi(rois, img_shape):
    # loop through ROIs, make mask with each ROI set to a label index
    final_img = np.zeros(img_shape, dtype='uint8')
    poly_error=False
    label_index=1
    for key in rois.keys():
        roi = rois[key]
        if (roi['type'] == 'polygon' or
                (roi['type'] == 'freehand' and 'x' in roi and 'y' in roi) or
                (roi['type'] == 'traced'   and 'x' in roi and 'y' in roi)):
            col_coords = roi['x']
            row_coords = roi['y']
            rr, cc = draw.polygon(row_coords, col_coords, shape=img_shape)
            final_img[rr, cc] = label_index
        elif (roi['type'] == 'rectangle'):
            rr, cc = draw.rectangle((roi['top'], roi['left']), extent=(roi['height'], roi['width']), shape=img_shape)
            rr=rr.astype('int')
            cc = cc.astype('int')
            final_img[rr, cc] = label_index
        elif (roi['type'] == 'oval'):
            rr, cc = draw.ellipse(roi['top'] + roi['height'] / 2, roi['left'] + roi['width'] / 2, roi['height'] / 2, roi['width'] / 2, shape=img_shape)
            final_img[rr, cc] = label_index
        else:
            poly_error=True

        if(not poly_error):
            label_index+=1

    return (final_img, poly_error)

def do_rescale(img, ll_perc, ul_perc):
    ll = ll_perc
    ul = 100 - ul_perc
    pmin, pmax = np.percentile(img, (ll, ul))
    return exposure.rescale_intensity(img, in_range=(int(pmin), int(pmax)))

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
        if ((r + label_h) >= len(img)): r_pos = r - ((r + label_h) - len(img))
        else: r_pos = r
        if ((c + label_w) >= len(img[0])): c_pos = c - ((c + label_w) - len(img[0]))
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