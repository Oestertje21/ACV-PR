import numpy as np
import cv2 as cv

def LBP_calculation(image : np.ndarray, cur_pix : tuple) -> np.uint8:
    LBP_value = np.uint8()
    threshold = image[cur_pix[0]][cur_pix[1]]
    
    # Specifies standard offsets for LBP generation
    ur_offset = -1  # Offset for rows above
    br_offset = 1   # Offset for rows below
    lc_offset = -1  # Offset for columns to the left
    rc_offset = 1   # Offset for columns to the right
    
    # First checks boundaries and mirrors offset for outer boundaries
    if(cur_pix[0] + ur_offset < 0):
        ur_offset = -ur_offset
    if(cur_pix[1] + lc_offset < 0):
        lc_offset = -lc_offset
    if(cur_pix[0] + br_offset >= image.shape[0]):
        br_offset = -br_offset
    if(cur_pix[1] + rc_offset >= image.shape[1]):
        rc_offset = -rc_offset

    # Checks pixels from top left in a clockwise manner
    if(image[cur_pix[0]+ur_offset][cur_pix[1]+lc_offset] >= threshold):
        LBP_value |= 1
    LBP_value <<= 1 # Shift bit left after comparison
        
    if(image[cur_pix[0]+ur_offset][cur_pix[1]] >= threshold):
        LBP_value |= 1
    LBP_value <<= 1
    
    if(image[cur_pix[0]+ur_offset][cur_pix[1]+rc_offset] >= threshold):
        LBP_value |= 1
    LBP_value <<= 1
    
    if(image[cur_pix[0]][cur_pix[1]+rc_offset] >= threshold):
        LBP_value |= 1
    LBP_value <<= 1
    
    if(image[cur_pix[0]+br_offset][cur_pix[1]+rc_offset] >= threshold):
        LBP_value |= 1
    LBP_value <<= 1
    
    if(image[cur_pix[0]+br_offset][cur_pix[1]] >= threshold):
        LBP_value |= 1
    LBP_value <<= 1
    
    if(image[cur_pix[0]+br_offset][cur_pix[1]+lc_offset] >= threshold):
        LBP_value |= 1
    LBP_value <<= 1
    
    if(image[cur_pix[0]][cur_pix[1]+lc_offset] >= threshold):
        LBP_value |= 1
                    
    return LBP_value

def LBP_loop_pixels(image : np.ndarray) -> np.ndarray:
    LBP_feature_array = np.zeros([image.shape[0], image.shape[1]], dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            LBP_feature_array[i][j] = LBP_calculation(image, (i,j))
    return LBP_feature_array
