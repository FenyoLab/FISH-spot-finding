# FISH-spot-finding

This script will perform automatic counting of FISH probes.  

Individual nuclei are segmented by applying an automatic threshold to the DAPI channel after a smoothing and contrast enhancement.  Thresholded objects are filtered for area and solidity to remove erroneously segmented regions.  For probe detection within segmented nuclei, a white tophat filter is applied to remove small spurious regions and then the “blob_log” function from scikit-image package (v0.17.2) (van der Walt et al., 2014)  is utilized to identify and count fluorescent spots.  Since it was observed that some FISH probes are incorrectly doubly counted, a distance cutoff is applied so that spots within a set distance will count as one.   

This script has been tested under Python 3.7 environment.  Please see requirements.txt for package installation.

For each image in folder:

(1) Identify nuclei - blue channels
(2) Identify spots (by blob detection) - green and red channels
(3) Get count of spots/nuclei
(4) Output summary file
(5) Output each image with nuclei and spots outlined - so these can be examined for correctness
(6) Output histograms of spot counts 
