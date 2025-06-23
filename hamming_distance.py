import numpy as np
from enum import Enum

# Enum to specify type of binary pattern to be used in the pipeline
class binary_method(Enum):
    LTBP_ZERO = 0
    LTBP_MIRROR = 1
    LTBP_ORIGINAL = 2 # Currently not implemented in LTBP
    LBP = 3
    
def hamming_distance_no_mask(code1: np.ndarray, code2: np.ndarray, method = binary_method):
    """
    Compute Hamming distance between two binary iris codes (without masking).

    Parameters:
    - code1: Binary NumPy array (e.g., dtype=uint8, shape=(H, W) or flattened)
    - code2: Same shape as code1

    Returns:
    - hamming_dist: Fraction of bits that differ (0.0 = identical, 1.0 = totally different)
    """
     
    # Precomputed lookup table for number of set bits in a byte (0–255)
    BIT_COUNT_LOOKUP = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

    if code1.shape != code2.shape:
        raise ValueError("Iris codes must have the same shape.")

    hd = 1.0
    #code1 = np.roll(code1, shift=60, axis=1)
    
    total_bits = code1.size * 8
    
    if method == binary_method.LBP:
        for i in range(int(code1.shape[1])):
            if i != 0:
                # shift 1 columns
                code1 = np.roll(code1, shift=1, axis=1)
    else:
        for i in range(int(code1.shape[1]/2)):
            if i != 0:
                # shift 2 columns
                code1 = np.roll(code1, shift=2, axis=1)
                        
    # XOR to find differing bits
    xor_result = np.bitwise_xor(code1, code2)

    # Count differing bits
    differing_bits = np.sum(BIT_COUNT_LOOKUP[xor_result])


    current_hd = differing_bits/total_bits
    if current_hd < hd:
        hd = current_hd
    
    # Hamming distance is fraction of differing bits
    return hd

def hamming_distance(iris_code1: np.ndarray, mask1: np.ndarray, iris_code2: np.ndarray, mask2: np.ndarray, method = binary_method) -> float:
    """
    Compute Hamming distance between two binary iris codes (without masking).

    Parameters:
    - iris_code1: Binary NumPy array (e.g., dtype=uint8, shape=(H, W) or flattened)
    - code2: Same shape as iris_code1

    Returns:
    - hamming_dist: Fraction of bits that differ (0.0 = identical, 1.0 = totally different)
    """
    if iris_code1.shape != iris_code2.shape:
        raise ValueError("Iris codes must have the same shape.")

    # Precomputed lookup table for number of set bits in a byte (0–255)
    BIT_COUNT_LOOKUP = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

    # Initial hd-value for comparison
    hd = 1.0
    
    if method == binary_method.LBP:
        for i in range(int(iris_code1.shape[1])):
            if i != 0:
                # shift 1 columns
                iris_code1 = np.roll(iris_code1, shift=1, axis=1)
                mask1 = np.roll(mask1, shift=1, axis=1)
    else:
        for i in range(int(iris_code1.shape[1]/2)):
            if i != 0:
                # shift 2 columns
                iris_code1 = np.roll(iris_code1, shift=2, axis=1)
                mask1 = np.roll(mask1, shift=2, axis=1)

    combined_mask = mask1 & mask2
        
    # XOR to find differing bits
    xor_result = np.bitwise_xor(iris_code1, iris_code2)

    # Combine mask and XOR result to skip irrelevant pixels
    masked_xor_result = np.bitwise_and(xor_result, combined_mask)
        
    # Count differing bits
    differing_bits = np.sum(BIT_COUNT_LOOKUP[masked_xor_result])
    total_valid_bits = np.sum(BIT_COUNT_LOOKUP[combined_mask])
    
    current_hd = differing_bits/total_valid_bits
    if current_hd < hd:
        hd = current_hd

    # Hamming distance is fraction of differing bits
    return hd
