import cv2 as cv
import numpy as np
from enum import Enum


class LTBP_method(Enum):
    MIRROR = 0
    ZERO = 1
    ORIGINAL = 2

def upper_triangle_original(image : np.ndarray, cur_pix : tuple) -> np.uint8:
    binary_code = np.uint8()
    
    if(cur_pix[0]-1 >= 0):
        # Gets value of pixel above as threshold
        threshold = image[cur_pix[0]-1][cur_pix[1]]
        
        # Specifies standard offsets for upper triangle
        ur_offset = 1
        lr_offset = -1
        cl_offset_1 = -1
        cl_offset_2 = -2
        cr_offset_1 = 1
        cr_offset_2 = 2
        
        # First checks boundaries and mirrors offset for outer boundaries
        if(cur_pix[0] + ur_offset >= image.shape[0]):
            ur_offset = -ur_offset    
        if(cur_pix[0] + lr_offset >= image.shape[0]):
            lr_offset = -lr_offset   
        if(cur_pix[1] + cl_offset_1 < 0):
            cl_offset_1 = -cl_offset_1
        if(cur_pix[1] + cl_offset_2 < 0):
            cl_offset_2 = -cl_offset_2
        if(cur_pix[1] + cr_offset_1 >= image.shape[1]):
            cr_offset_1 = -cr_offset_1
        if(cur_pix[1] + cr_offset_2 >= image.shape[1]):
            cr_offset_2 = -cr_offset_2
        
        # Checks pixels in descending order of importance
        if(image[cur_pix[0]+ur_offset][cur_pix[1]+cl_offset_2] >= threshold):
            binary_code |= 1
        binary_code <<= 1 # Shift bit left after comparison
            
        if(image[cur_pix[0]+ur_offset][cur_pix[1]+cl_offset_1] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(image[cur_pix[0]+ur_offset][cur_pix[1]] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(image[cur_pix[0]+ur_offset][cur_pix[1]+cr_offset_1] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(image[cur_pix[0]+ur_offset][cur_pix[1]+cr_offset_2] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(image[cur_pix[0]][cur_pix[1]+cr_offset_1] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(image[cur_pix[0]+lr_offset][cur_pix[1]] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(image[cur_pix[0]][cur_pix[1]+cl_offset_1] >= threshold):
            binary_code |= 1
    return binary_code
    
def upper_triangle_mirror(image : np.ndarray, cur_pix : tuple) -> np.uint8:
    threshold = image[cur_pix[0]][cur_pix[1]]
    binary_code = np.uint8()
    
    # Specifies standard offsets for upper triangle
    r_offset_1 = -1
    r_offset_2 = -2
    cl_offset_1 = -1
    cl_offset_2 = -2
    cr_offset_1 = 1
    cr_offset_2 = 2
    
    # First checks boundaries and mirrors offset for outer boundaries
    if(cur_pix[0] + r_offset_1 < 0):
        r_offset_1 = -r_offset_1
    if(cur_pix[0] + r_offset_2 < 0):
        r_offset_2 = -r_offset_2
    if(cur_pix[1] + cl_offset_1 < 0):
        cl_offset_1 = -cl_offset_1
    if(cur_pix[1] + cl_offset_2 < 0):
        cl_offset_2 = -cl_offset_2
    if(cur_pix[1] + cr_offset_1 >= image.shape[1]):
        cr_offset_1 = -cr_offset_1
    if(cur_pix[1] + cr_offset_2 >= image.shape[1]):
        cr_offset_2 = -cr_offset_2
    
    # Checks pixels in descending order of importance
    if(image[cur_pix[0]+r_offset_1][cur_pix[1]] >= threshold):
        binary_code |= 1
    binary_code <<= 1 # Shift bit left after comparison
        
    if(image[cur_pix[0]+r_offset_1][cur_pix[1]+cr_offset_1] >= threshold):
        binary_code |= 1
    binary_code <<= 1
    
    if(image[cur_pix[0]+r_offset_1][cur_pix[1]+cl_offset_1] >= threshold):
        binary_code |= 1
    binary_code <<= 1
    
    # Checking highest row
    if(image[cur_pix[0]+r_offset_2][cur_pix[1]] >= threshold):
        binary_code |= 1
    binary_code <<= 1
    
    if(image[cur_pix[0]+r_offset_2][cur_pix[1]+cr_offset_1] >= threshold):
        binary_code |= 1
    binary_code <<= 1
    
    if(image[cur_pix[0]+r_offset_2][cur_pix[1]+cl_offset_1] >= threshold):
        binary_code |= 1
    binary_code <<= 1
    
    if(image[cur_pix[0]+r_offset_2][cur_pix[1]+cr_offset_2] >= threshold):
        binary_code |= 1
    binary_code <<= 1
    
    if(image[cur_pix[0]+r_offset_2][cur_pix[1]+cl_offset_2] >= threshold):
        binary_code |= 1
    
    return binary_code
    
def upper_triangle_zero(image : np.ndarray, cur_pix : tuple) -> np.uint8:
    threshold = image[cur_pix[0]][cur_pix[1]]
    binary_code = np.uint8()
    
    # Specifies standard offsets for upper triangle
    r_offset_1 = -1
    r_offset_2 = -2
    cl_offset_1 = -1
    cl_offset_2 = -2
    cr_offset_1 = 1
    cr_offset_2 = 2
    
    # First checks row boundaries and returns respective values
    if(cur_pix[0] + r_offset_1 < 0):
        # All pixels outside of image, binary code = 0
        binary_code = 0
        return binary_code
    
    elif(cur_pix[0] + r_offset_2 < 0):
        # Bit 8 check
        if(image[cur_pix[0]+r_offset_1][cur_pix[1]] >= threshold):
            binary_code |= 1
        binary_code <<= 1 # Shift bit left after comparison
                    
        # Bit 7 check
        if(cur_pix[1]+cr_offset_1 >= image.shape[1]):
            binary_code |= 0
        elif(image[cur_pix[0]+r_offset_1][cur_pix[1]+cr_offset_1] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        # Bit 6 check
        if(cur_pix[1]+cl_offset_1 < 0):
            binary_code |= 0
        elif(image[cur_pix[0]+r_offset_1][cur_pix[1]+cl_offset_1] >= threshold):
            binary_code |= 1
        # Second row falls outside of image shift 5 to set all to 0
        binary_code <<= 5
        return binary_code
    
    # No pixels outside of image, so normal check
    else:                                                                            
        # Checks pixels in descending order of importance
        if(image[cur_pix[0]+r_offset_1][cur_pix[1]] >= threshold):
            binary_code |= 1
        binary_code <<= 1 # Shift bit left after comparison
        
        if(cur_pix[1] + cr_offset_1 >= image.shape[1]):
            binary_code |= 0
        elif(image[cur_pix[0]+r_offset_1][cur_pix[1]+cr_offset_1] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(cur_pix[1] + cl_offset_1 < 0):
            binary_code |= 0
        elif(image[cur_pix[0]+r_offset_1][cur_pix[1]+cl_offset_1] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        # Checking highest row
        if(image[cur_pix[0]+r_offset_2][cur_pix[1]] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(cur_pix[1] + cr_offset_1 >= image.shape[1]):
            binary_code |= 0
        elif(image[cur_pix[0]+r_offset_2][cur_pix[1]+cr_offset_1] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(cur_pix[1] + cl_offset_1 < 0):
            binary_code |= 0
        elif(image[cur_pix[0]+r_offset_2][cur_pix[1]+cl_offset_1] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(cur_pix[1] + cr_offset_2 >= image.shape[1]):
            binary_code |= 0
        elif(image[cur_pix[0]+r_offset_2][cur_pix[1]+cr_offset_2] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(cur_pix[1] + cl_offset_2 < 0):
            binary_code |= 0
        elif(image[cur_pix[0]+r_offset_2][cur_pix[1]+cl_offset_2] >= threshold):
            binary_code |= 1
        
        return binary_code

def right_triangle_original(image : np.ndarray, cur_pix : tuple) -> np.uint8:
    binary_code = np.uint8()
        
    if(cur_pix[1]+1 < image.shape[1]): 
        threshold = image[cur_pix[0]][cur_pix[1]+1]
        
        # Specifies standard offsets for right triangle
        cl_offset = -1
        cr_offset = 1
        ur_offset_1 = -1
        ur_offset_2 = -2
        lr_offset_1 = 1
        lr_offset_2 = 2
        
        # First checks boundaries and mirrors offset for outer boundaries
        if(cur_pix[1] + cr_offset >= image.shape[1]):
            cr_offset = -cr_offset 
        if(cur_pix[1] + cl_offset < 0):
            cl_offset = -cl_offset
        if(cur_pix[0] + ur_offset_1 < 0):
            ur_offset_1 = -ur_offset_1
        if(cur_pix[0] + ur_offset_2 < 0):
            ur_offset_2 = -ur_offset_2
        if(cur_pix[0] + lr_offset_1 >= image.shape[0]):
            lr_offset_1 = -lr_offset_1
        if(cur_pix[0] + lr_offset_2 >= image.shape[0]):
            lr_offset_2 = -lr_offset_2
        
        # Checks pixels in descending order of importance
        if(image[cur_pix[0]+ur_offset_2][cur_pix[1]+cr_offset] >= threshold):
            binary_code |= 1
        binary_code <<= 1 # Shift bit left after comparison
            
        if(image[cur_pix[0]+ur_offset_1][cur_pix[1]+cr_offset] >= threshold):
            binary_code |= 1
        binary_code <<= 1
            
        if(image[cur_pix[0]][cur_pix[1]+cr_offset] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(image[cur_pix[0]+lr_offset_1][cur_pix[1]+cr_offset] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(image[cur_pix[0]+lr_offset_2][cur_pix[1]+cr_offset] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(image[cur_pix[0]+lr_offset_1][cur_pix[1]] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(image[cur_pix[0]][cur_pix[1]+cl_offset] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(image[cur_pix[0]+ur_offset_1][cur_pix[1]] >= threshold):
            binary_code |= 1
    
    return binary_code

def right_triangle_mirror(image : np.ndarray, cur_pix : tuple) -> np.uint8:
    threshold = image[cur_pix[0]][cur_pix[1]]
    binary_code = np.uint8()
    
    # Specifies standard offsets for right triangle
    c_offset_1 = 1
    c_offset_2 = 2
    ur_offset_1 = -1
    ur_offset_2 = -2
    br_offset_1 = 1
    br_offset_2 = 2
    
    # First checks boundaries and mirrors offset for outer boundaries
    if(cur_pix[1] + c_offset_1 >= image.shape[1]):
        c_offset_1 = -c_offset_1
    if(cur_pix[1] + c_offset_2 >= image.shape[1]):
        c_offset_2 = -c_offset_2
    if(cur_pix[0] + ur_offset_1 < 0):
        ur_offset_1 = -ur_offset_1
    if(cur_pix[0] + ur_offset_2 < 0):
        ur_offset_2 = -ur_offset_2 
    if(cur_pix[0] + br_offset_1 >= image.shape[0]):
        br_offset_1 = -br_offset_1
    if(cur_pix[0] + br_offset_2 >= image.shape[0]):
        br_offset_2 = -br_offset_2
    
    # Checks pixels in descending order of importance
    if(image[cur_pix[0]][cur_pix[1]+c_offset_1] >= threshold):
        binary_code |= 1
    binary_code <<= 1 # Shift bit left after comparison
        
    if(image[cur_pix[0]+br_offset_1][cur_pix[1]+c_offset_1] >= threshold):
        binary_code |= 1
    binary_code <<= 1
    
    if(image[cur_pix[0]+ur_offset_1][cur_pix[1]+c_offset_1] >= threshold):
        binary_code |= 1
    binary_code <<= 1
    
    # Checking right-most column
    if(image[cur_pix[0]][cur_pix[1]+c_offset_2] >= threshold):
        binary_code |= 1
    binary_code <<= 1
    
    if(image[cur_pix[0]+br_offset_1][cur_pix[1]+c_offset_2] >= threshold):
        binary_code |= 1
    binary_code <<= 1
    
    if(image[cur_pix[0]+ur_offset_1][cur_pix[1]+c_offset_2] >= threshold):
        binary_code |= 1
    binary_code <<= 1
    
    if(image[cur_pix[0]+br_offset_2][cur_pix[1]+c_offset_2] >= threshold):
        binary_code |= 1
    binary_code <<= 1
    
    if(image[cur_pix[0]+ur_offset_2][cur_pix[1]+c_offset_2] >= threshold):
        binary_code |= 1
    
    return binary_code

def right_triangle_zero(image : np.ndarray, cur_pix : tuple) -> np.uint8:
    threshold = image[cur_pix[0]][cur_pix[1]]
    binary_code = np.uint8()
    
    # Specifies standard offsets for right triangle
    c_offset_1 = 1
    c_offset_2 = 2
    ur_offset_1 = -1
    ur_offset_2 = -2
    br_offset_1 = 1
    br_offset_2 = 2
    
    if(cur_pix[1]+c_offset_1 >= image.shape[1]):
        # All pixels outside of image, binary code = 0
        binary_code = 0
        return binary_code
    elif(cur_pix[1]+c_offset_2 >= image.shape[1]):
        # Check bit 8
        if(image[cur_pix[0]][cur_pix[1]+c_offset_1] >= threshold):
            binary_code |= 1
        binary_code <<= 1 # Shift bit left after comparison
        
        # Check bit 7
        if(cur_pix[0]+br_offset_1 >= image.shape[0]):
            binary_code |= 0
        elif(image[cur_pix[0]+br_offset_1][cur_pix[1]+c_offset_1] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        # Check bit 6
        if(cur_pix[0]+ur_offset_1 < 0):
            binary_code |= 0
        elif(image[cur_pix[0]+ur_offset_1][cur_pix[1]+c_offset_1] >= threshold):
            binary_code |= 1
        # Second column falls outside of image shift 5 to set all to 0
        binary_code <<= 5
        return binary_code
        
    # No pixels outside of image, so normal check
    else:
        # Checks pixels in descending order of importance
        if(image[cur_pix[0]][cur_pix[1]+c_offset_1] >= threshold):
            binary_code |= 1
        binary_code <<= 1 # Shift bit left after comparison
        
        if(cur_pix[0] + br_offset_1 >= image.shape[0]):
            binary_code |= 0
        elif(image[cur_pix[0]+br_offset_1][cur_pix[1]+c_offset_1] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(cur_pix[0] + ur_offset_1 < 0):
            binary_code |= 0
        elif(image[cur_pix[0]+ur_offset_1][cur_pix[1]+c_offset_1] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        # Checking right-most column
        if(image[cur_pix[0]][cur_pix[1]+c_offset_2] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(cur_pix[0] + br_offset_1 >= image.shape[0]):
            binary_code |= 0
        elif(image[cur_pix[0]+br_offset_1][cur_pix[1]+c_offset_2] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(cur_pix[0] + ur_offset_1 < 0):
            binary_code |= 0
        elif(image[cur_pix[0]+ur_offset_1][cur_pix[1]+c_offset_2] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(cur_pix[0] + br_offset_2 >= image.shape[0]):
            binary_code |= 0
        elif(image[cur_pix[0]+br_offset_2][cur_pix[1]+c_offset_2] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(cur_pix[0] + ur_offset_2 < 0):
            binary_code |= 0
        elif(image[cur_pix[0]+ur_offset_2][cur_pix[1]+c_offset_2] >= threshold):
            binary_code |= 1
        
        return binary_code

def lower_triangle_original(image : np.ndarray, cur_pix : tuple) -> np.uint8:
    binary_code = np.uint8()
    
    if(cur_pix[0]+1 < image.shape[0]):
        threshold = image[cur_pix[0]+1][cur_pix[1]]
        
        # Specifies standard offsets for lower triangle
        lr_offset = 1
        ur_offset = -1
        cl_offset_1 = -1
        cl_offset_2 = -2
        cr_offset_1 = 1
        cr_offset_2 = 2
        
        # First checks boundaries and mirrors offset for outer boundaries
        if(cur_pix[0] + lr_offset >= image.shape[0]):
            lr_offset = -lr_offset
        if(cur_pix[0] + ur_offset <0):
            ur_offset = -ur_offset
        if(cur_pix[1] + cl_offset_1 < 0):
            cl_offset_1 = -cl_offset_1
        if(cur_pix[1] + cl_offset_2 < 0):
            cl_offset_2 = -cl_offset_2
        if(cur_pix[1] + cr_offset_1 >= image.shape[1]):
            cr_offset_1 = -cr_offset_1
        if(cur_pix[1] + cr_offset_2 >= image.shape[1]):
            cr_offset_2 = -cr_offset_2
        
        # Checks pixels in descending order of importance
        if(image[cur_pix[0]+ur_offset][cur_pix[1]] >= threshold):
            binary_code |= 1
        binary_code <<= 1 # Shift bit left after comparison
            
        if(image[cur_pix[0]][cur_pix[1]+cr_offset_1] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(image[cur_pix[0]+lr_offset][cur_pix[1]+cr_offset_2] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(image[cur_pix[0]+lr_offset][cur_pix[1]+cr_offset_1] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(image[cur_pix[0]+lr_offset][cur_pix[1]] >= threshold):
            binary_code |= 1
        binary_code <<= 1

        if(image[cur_pix[0]+lr_offset][cur_pix[1]+cl_offset_1] >= threshold):
            binary_code |= 1
        binary_code <<= 1   
     
        if(image[cur_pix[0]+lr_offset][cur_pix[1]+cl_offset_2] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(image[cur_pix[0]][cur_pix[1]+cl_offset_1] >= threshold):
            binary_code |= 1
        
    return binary_code

def lower_triangle_mirror(image : np.ndarray, cur_pix : tuple) -> np.uint8:
    threshold = image[cur_pix[0]][cur_pix[1]]
    binary_code = np.uint8()
    
    # Specifies standard offsets for lower triangle
    r_offset_1 = 1
    r_offset_2 = 2
    cl_offset_1 = -1
    cl_offset_2 = -2
    cr_offset_1 = 1
    cr_offset_2 = 2
    
    # First checks boundaries and mirrors offset for outer boundaries
    if(cur_pix[0] + r_offset_1 >= image.shape[0]):
        r_offset_1 = -r_offset_1
    if(cur_pix[0] + r_offset_2 >= image.shape[0]):
        r_offset_2 = -r_offset_2
    if(cur_pix[1] + cl_offset_1 < 0):
        cl_offset_1 = -cl_offset_1
    if(cur_pix[1] + cl_offset_2 < 0):
        cl_offset_2 = -cl_offset_2
    if(cur_pix[1] + cr_offset_1 >= image.shape[1]):
        cr_offset_1 = -cr_offset_1
    if(cur_pix[1] + cr_offset_2 >= image.shape[1]):
        cr_offset_2 = -cr_offset_2
    
    # Checks pixels in descending order of importance
    if(image[cur_pix[0]+r_offset_1][cur_pix[1]] >= threshold):
        binary_code |= 1
    binary_code <<= 1 # Shift bit left after comparison
        
    if(image[cur_pix[0]+r_offset_1][cur_pix[1]+cl_offset_1] >= threshold):
        binary_code |= 1
    binary_code <<= 1
    
    if(image[cur_pix[0]+r_offset_1][cur_pix[1]+cr_offset_1] >= threshold):
        binary_code |= 1
    binary_code <<= 1
    
    # Checking lowest row
    if(image[cur_pix[0]+r_offset_2][cur_pix[1]] >= threshold):
        binary_code |= 1
    binary_code <<= 1
    
    if(image[cur_pix[0]+r_offset_2][cur_pix[1]+cl_offset_1] >= threshold):
        binary_code |= 1
    binary_code <<= 1
    
    if(image[cur_pix[0]+r_offset_2][cur_pix[1]+cr_offset_1] >= threshold):
        binary_code |= 1
    binary_code <<= 1
    
    if(image[cur_pix[0]+r_offset_2][cur_pix[1]+cl_offset_2] >= threshold):
        binary_code |= 1
    binary_code <<= 1
    
    if(image[cur_pix[0]+r_offset_2][cur_pix[1]+cr_offset_2] >= threshold):
        binary_code |= 1
    
    return binary_code

def lower_triangle_zero(image : np.ndarray, cur_pix : tuple) -> np.uint8:
    threshold = image[cur_pix[0]][cur_pix[1]]
    binary_code = np.uint8()
    
    # Specifies standard offsets for lower triangle
    r_offset_1 = 1
    r_offset_2 = 2
    cl_offset_1 = -1
    cl_offset_2 = -2
    cr_offset_1 = 1
    cr_offset_2 = 2
    
    # First checks if any of the rows exceed the image lower boundaries
    if(cur_pix[0] + r_offset_1 >= image.shape[0]):
        # All pixels outside of image, binary code = 0
        binary_code = 0
        return binary_code
    
    elif(cur_pix[0] + r_offset_2 >= image.shape[0]):
        # Bit 8 check
        if(image[cur_pix[0]+r_offset_1][cur_pix[1]] >= threshold):
            binary_code |= 1
        binary_code <<= 1 # Shift bit left after comparison
        
        # Bit 7 check
        if(cur_pix[1] + cl_offset_1 < 0):
            binary_code |= 0
        elif(image[cur_pix[0]+r_offset_1][cur_pix[1]+cl_offset_1] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        # Bit 6 check
        if(cur_pix[1] + cr_offset_1 >= image.shape[1]):
            binary_code |= 0
        elif(image[cur_pix[0]+r_offset_1][cur_pix[1]+cr_offset_1] >= threshold):
            binary_code |= 1
            
        # Second row falls outside of image shift 5 to set all to 0
        binary_code <<= 5
        return binary_code

    # No pixels outside of image, so normal check
    else:
        # Checks pixels in descending order of importance
        if(image[cur_pix[0]+r_offset_1][cur_pix[1]] >= threshold):
            binary_code |= 1
        binary_code <<= 1 # Shift bit left after comparison
        
        if(cur_pix[1] + cl_offset_1 < 0):
            binary_code |= 0
        elif(image[cur_pix[0]+r_offset_1][cur_pix[1]+cl_offset_1] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(cur_pix[1] + cr_offset_1 >= image.shape[1]):
            binary_code |= 0
        elif(image[cur_pix[0]+r_offset_1][cur_pix[1]+cr_offset_1] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        # Checking lowest row
        if(image[cur_pix[0]+r_offset_2][cur_pix[1]] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(cur_pix[1] + cl_offset_1 < 0):
            binary_code |= 0
        elif(image[cur_pix[0]+r_offset_2][cur_pix[1]+cl_offset_1] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(cur_pix[1] + cr_offset_1 >= image.shape[1]):
            binary_code |= 0
        elif(image[cur_pix[0]+r_offset_2][cur_pix[1]+cr_offset_1] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(cur_pix[1] + cl_offset_2 < 0):
            binary_code |= 0
        elif(image[cur_pix[0]+r_offset_2][cur_pix[1]+cl_offset_2] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(cur_pix[1] + cr_offset_2 >= image.shape[1]):
            binary_code |= 0
        elif(image[cur_pix[0]+r_offset_2][cur_pix[1]+cr_offset_2] >= threshold):
            binary_code |= 1
        return binary_code 

def left_triangle_original(image : np.ndarray, cur_pix : tuple) -> np.uint8:
    binary_code = np.uint8()
    
    if(cur_pix[1]-1 >= 0):
        threshold = image[cur_pix[0]][cur_pix[1]-1]
        

        # Specifies standard offsets for left triangle
        cl_offset = -1
        cr_offset = 1
        ur_offset_1 = -1
        ur_offset_2 = -2
        lr_offset_1 = 1
        lr_offset_2 = 2
        
        # First checks boundaries and mirrors offset for outer boundaries
        if(cur_pix[1] + cl_offset < 0):
            cl_offset = -cl_offset
        if(cur_pix[1] + cr_offset >= image.shape[1]):
            cr_offset = -cr_offset
        if(cur_pix[0] + ur_offset_1 < 0):
            ur_offset_1 = -ur_offset_1
        if(cur_pix[0] + ur_offset_2 < 0):
            ur_offset_2 = -ur_offset_2 
        if(cur_pix[0] + lr_offset_1 >= image.shape[0]):
            lr_offset_1 = -lr_offset_1
        if(cur_pix[0] + lr_offset_2 >= image.shape[0]):
            lr_offset_2 = -lr_offset_2
        
        # Checks pixels in descending order of importance
        if(image[cur_pix[0]+ur_offset_2][cur_pix[1]+cl_offset] >= threshold):
            binary_code |= 1
        binary_code <<= 1 # Shift bit left after comparison
            
        if(image[cur_pix[0]+ur_offset_1][cur_pix[1]] >= threshold):
            binary_code |= 1
        binary_code <<= 1          

        if(image[cur_pix[0]][cur_pix[1]+cr_offset] >= threshold):
            binary_code |= 1
        binary_code <<= 1
             
        if(image[cur_pix[0]+lr_offset_1][cur_pix[1]] >= threshold):
            binary_code |= 1
        binary_code <<= 1     

        if(image[cur_pix[0]+lr_offset_2][cur_pix[1]+cl_offset] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(image[cur_pix[0]+lr_offset_1][cur_pix[1]+cl_offset] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(image[cur_pix[0]][cur_pix[1]+cl_offset] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(image[cur_pix[0]+ur_offset_1][cur_pix[1]+cl_offset] >= threshold):
            binary_code |= 1

    return binary_code

def left_triangle_mirror(image : np.ndarray, cur_pix : tuple) -> np.uint8:
    threshold = image[cur_pix[0]][cur_pix[1]]
    binary_code = np.uint8()

    # Specifies standard offsets for left triangle
    c_offset_1 = -1
    c_offset_2 = -2
    ur_offset_1 = -1
    ur_offset_2 = -2
    br_offset_1 = 1
    br_offset_2 = 2
    
    # First checks boundaries and mirrors offset for outer boundaries
    if(cur_pix[1] + c_offset_1 < 0):
        c_offset_1 = -c_offset_1
    if(cur_pix[1] + c_offset_2 < 0):
        c_offset_2 = -c_offset_2
    if(cur_pix[0] + ur_offset_1 < 0):
        ur_offset_1 = -ur_offset_1
    if(cur_pix[0] + ur_offset_2 < 0):
        ur_offset_2 = -ur_offset_2 
    if(cur_pix[0] + br_offset_1 >= image.shape[0]):
        br_offset_1 = -br_offset_1
    if(cur_pix[0] + br_offset_2 >= image.shape[0]):
        br_offset_2 = -br_offset_2
    
    # Checks pixels in descending order of importance
    if(image[cur_pix[0]][cur_pix[1]+c_offset_1] >= threshold):
        binary_code |= 1
    binary_code <<= 1 # Shift bit left after comparison
        
    if(image[cur_pix[0]+ur_offset_1][cur_pix[1]+c_offset_1] >= threshold):
        binary_code |= 1
    binary_code <<= 1
    
    if(image[cur_pix[0]+br_offset_1][cur_pix[1]+c_offset_1] >= threshold):
        binary_code |= 1
    binary_code <<= 1
    
    # Checking right-most column
    if(image[cur_pix[0]][cur_pix[1]+c_offset_2] >= threshold):
        binary_code |= 1
    binary_code <<= 1
    
    if(image[cur_pix[0]+ur_offset_1][cur_pix[1]+c_offset_2] >= threshold):
        binary_code |= 1
    binary_code <<= 1
    
    if(image[cur_pix[0]+br_offset_1][cur_pix[1]+c_offset_2] >= threshold):
        binary_code |= 1
    binary_code <<= 1
    
    if(image[cur_pix[0]+ur_offset_2][cur_pix[1]+c_offset_2] >= threshold):
        binary_code |= 1
    binary_code <<= 1
    
    if(image[cur_pix[0]+br_offset_2][cur_pix[1]+c_offset_2] >= threshold):
        binary_code |= 1
    
    return binary_code

def left_triangle_zero(image : np.ndarray, cur_pix : tuple) -> np.uint8:
    threshold = image[cur_pix[0]][cur_pix[1]]
    binary_code = np.uint8()

    # Specifies standard offsets for left triangle
    c_offset_1 = -1
    c_offset_2 = -2
    ur_offset_1 = -1
    ur_offset_2 = -2
    br_offset_1 = 1
    br_offset_2 = 2

    if(cur_pix[1] + c_offset_1 < 0):
        # All pixels outside of image, binary code = 0
        binary_code = 0
        return binary_code
    elif(cur_pix[1] + c_offset_2 < 0):
        # Check bit 8
        if(image[cur_pix[0]][cur_pix[1]+c_offset_1] >= threshold):
            binary_code |= 1
        binary_code <<= 1 # Shift bit left after comparison
        
        # Check bit 7
        if(cur_pix[0] + ur_offset_1 < 0):
            binary_code |= 0
        elif(image[cur_pix[0]+ur_offset_1][cur_pix[1]+c_offset_1] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        # Check bit 6
        if(cur_pix[0] + br_offset_1 >= image.shape[0]):
            binary_code |= 0
        elif(image[cur_pix[0]+br_offset_1][cur_pix[1]+c_offset_1] >= threshold):
            binary_code |= 1
            
        # Second row falls outside of image shift 5 to set all to 0
        binary_code <<= 5
        return binary_code
    
    # No pixels outside of image, so normal check
    else:
        # Checks pixels in descending order of importance
        if(image[cur_pix[0]][cur_pix[1]+c_offset_1] >= threshold):
            binary_code |= 1
        binary_code <<= 1 # Shift bit left after comparison
        
        if(cur_pix[0] + ur_offset_1 < 0):
            binary_code |= 0
        elif(image[cur_pix[0]+ur_offset_1][cur_pix[1]+c_offset_1] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(cur_pix[0] + br_offset_1 >= image.shape[0]):
            binary_code |= 0
        elif(image[cur_pix[0]+br_offset_1][cur_pix[1]+c_offset_1] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        # Checking right-most column
        if(image[cur_pix[0]][cur_pix[1]+c_offset_2] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(cur_pix[0] + ur_offset_1 < 0):
            binary_code |= 0
        elif(image[cur_pix[0]+ur_offset_1][cur_pix[1]+c_offset_2] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(cur_pix[0] + br_offset_1 >= image.shape[0]):
            binary_code |= 0
        elif(image[cur_pix[0]+br_offset_1][cur_pix[1]+c_offset_2] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(cur_pix[0] + ur_offset_2 < 0):
            binary_code |= 0
        elif(image[cur_pix[0]+ur_offset_2][cur_pix[1]+c_offset_2] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(cur_pix[0] + br_offset_2 >= image.shape[0]):
            binary_code |= 0
        elif(image[cur_pix[0]+br_offset_2][cur_pix[1]+c_offset_2] >= threshold):
            binary_code |= 1
        
        return binary_code

def LTBP_calculation_original(image : np.ndarray, cur_pix : tuple) -> np.ndarray:
    LTBP_values = np.array([0,0,0,0], dtype=np.uint8)
    LTBP_values[0] = upper_triangle_original(image, cur_pix)
    LTBP_values[1] = right_triangle_original(image, cur_pix)
    LTBP_values[2] = lower_triangle_original(image, cur_pix)
    LTBP_values[3] = left_triangle_original(image, cur_pix)
    return LTBP_values
                
def LTBP_calculation_mirror(image : np.ndarray, cur_pix : tuple) -> np.ndarray:
    LTBP_values = np.array([0,0,0,0], dtype=np.uint8)
    LTBP_values[0] = upper_triangle_mirror(image, cur_pix)
    LTBP_values[1] = right_triangle_mirror(image, cur_pix)
    LTBP_values[2] = lower_triangle_mirror(image, cur_pix)
    LTBP_values[3] = left_triangle_mirror(image, cur_pix)
    return LTBP_values

def LTBP_calculation_zero(image : np.ndarray, cur_pix : tuple) -> np.ndarray:
    LTBP_values = np.array([0,0,0,0], dtype=np.uint8)
    LTBP_values[0] = upper_triangle_zero(image, cur_pix)
    LTBP_values[1] = right_triangle_zero(image, cur_pix)
    LTBP_values[2] = lower_triangle_zero(image, cur_pix)
    LTBP_values[3] = left_triangle_zero(image, cur_pix)
    return LTBP_values

def LTBP_loop_pixels(image : np.ndarray, method : LTBP_method) -> np.ndarray:
        LTBP_feature_array = np.zeros([image.shape[0]*2, image.shape[1]*2], dtype=np.uint8)
        if(method == LTBP_method.MIRROR):
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    LTBP_values = LTBP_calculation_mirror(image, (i,j))
                    LTBP_feature_array[i*2][j*2] = LTBP_values[0]
                    LTBP_feature_array[i*2][j*2+1] = LTBP_values[1]
                    LTBP_feature_array[i*2+1][j*2+1] = LTBP_values[2]
                    LTBP_feature_array[i*2+1][j*2] = LTBP_values[3]
        elif(method == LTBP_method.ZERO):
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    LTBP_values = LTBP_calculation_zero(image, (i,j))
                    LTBP_feature_array[i*2][j*2] = LTBP_values[0]
                    LTBP_feature_array[i*2][j*2+1] = LTBP_values[1]
                    LTBP_feature_array[i*2+1][j*2+1] = LTBP_values[2]
                    LTBP_feature_array[i*2+1][j*2] = LTBP_values[3]
        elif(method == LTBP_method.ORIGINAL):
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    LTBP_values = LTBP_calculation_original(image, (i,j))
                    LTBP_feature_array[i*2][j*2] = LTBP_values[0]
                    LTBP_feature_array[i*2][j*2+1] = LTBP_values[1]
                    LTBP_feature_array[i*2+1][j*2+1] = LTBP_values[2]
                    LTBP_feature_array[i*2+1][j*2] = LTBP_values[3]
        else:
            raise ValueError("Incorrect method supplied for LTBP pattern generation")
        return LTBP_feature_array