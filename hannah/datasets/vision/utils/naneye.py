# Naneye Decoder

import logging

from pathlib import Path 
from typing import Union

import numpy as np
import cv2

logger = logging.getLogger(__name__)

def read_naneye(data_file: Union[str, Path]):
    """Read a naneye raw aimage and decode bayer pattern

    Args:
        data_file (Union[str, Path]): path to the datafile

    Returns:
        np.ndarray: uint8 array of decoded image data
    """
    data_file_path : Path = Path(data_file)
    
    data_array = np.loadtxt(data_file_path, dtype=np.uint8)
    
    
    data_array = data_array[1:-3]
    
    
    last = None
    missing = []
    for num, line in enumerate(data_array):
        current = line[0]*256+line[1]
        
        
        if last != None: 
            if current != last+1:
                logger.warning(f"Missing lines, last: {last}, current: {current}")
                missing += range(last, current-1) # Start indexing at zero
             
        last = current

    for line_number in missing:
        logger.info(f"Interpolating missing line {line_number}")
        if line_number >= 2:
            data_array = np.insert(data_array, line_number, data_array[line_number-2], axis=0)
        else:
            #interpolate from next line FIXME: might be wrong line, if more than one consecutive lines ar missing 
            data_array = np.insert(data_array, line_number, data_array[line_number+1], axis=0)
            
        
    data_array = data_array[:,2:]
    
    data_array = cv2.cvtColor(data_array, cv2.COLOR_BAYER_GRBG2BGR)
    
    return data_array


if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("--show", action='store_true', default=False, help="show decoded image implicitly assumed if --output is not given")
    parser.add_argument("-o", "--output", default=None, help="The ouput file")

    args = parser.parse_args()
    
    print("input_file", args.input_file)
    print("output", args.output)
    print("show output", args.show) 
    
    input_file = Path(args.input_file)
    output_file = args.output
    if output_file is not None:
        output_file = Path(output_file)

    
    input_data = read_naneye(input_file)
    
    if args.show or output_file is None:
        cv2.imshow("input_data", input_data)
      
    if output_file is not None:  
        cv2.imwrite(str(output_file), input_data)
        
        
    