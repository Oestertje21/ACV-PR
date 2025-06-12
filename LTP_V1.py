import cv2 as cv
import numpy as np
from enum import Enum

class LTP_Class:
    def upper_triangle_mirror(image : np.ndarray, cur_pix : tuple) -> int:
        threshold = image[cur_pix[0]][cur_pix[1]]
        binary_code = 0
        
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
        
        return int(binary_code)
        
                      
    def right_triangle_mirror(image : np.ndarray, cur_pix : tuple) -> int:
        threshold = image[cur_pix[0]][cur_pix[1]]
        binary_code = 0
        
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
        
        return int(binary_code)
        
       
    def lower_triangle_mirror(image : np.ndarray, cur_pix : tuple) -> int:
        threshold = image[cur_pix[0]][cur_pix[1]]
        binary_code = 0
        
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
           
        if(image[cur_pix[0]+r_offset_1][cur_pix[1]+cr_offset_1] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        if(image[cur_pix[0]+r_offset_1][cur_pix[1]+cl_offset_1] >= threshold):
            binary_code |= 1
        binary_code <<= 1
        
        # Checking lowest row
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
        
        return int(binary_code) 
        
    def left_triangle_mirror(image : np.ndarray, cur_pix : tuple) -> int:
        threshold = image[cur_pix[0]][cur_pix[1]]
        binary_code = 0
        
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
        
        return int(binary_code)
        
         
    def LTP_calculation_mirror(image : np.ndarray, cur_pix : tuple) -> np.ndarray:
        ltp_values = np.array([0,0,0,0])
        ltp_values[0] = LTP_Class.upper_triangle_mirror(image, cur_pix)
        ltp_values[1] = LTP_Class.right_triangle_mirror(image, cur_pix)
        ltp_values[2] = LTP_Class.lower_triangle_mirror(image, cur_pix)
        ltp_values[3] = LTP_Class.left_triangle_mirror(image, cur_pix)
        return ltp_values
    
    
    def LTP_loop_pixels(image : np.ndarray) -> np.ndarray:
        ltp_feature_array = np.zeros([image.shape[0]*2, image.shape[1]*2], dtype=np.uint8)
        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                ltp_values = LTP_Class.LTP_calculation_mirror(image, (i,j))
                ltp_feature_array[i*2][j*2] = ltp_values[0]
                ltp_feature_array[i*2][j*2+1] = ltp_values[1]
                ltp_feature_array[i*2+1][j*2+1] = ltp_values[2]
                ltp_feature_array[i*2+1][j*2] = ltp_values[3]
        return ltp_feature_array
                                                           