import pytest

from pathlib import Path 
from typing import Union

import numpy as np

cv2 = pytest.importorskip('cv2')


test_img = Path(__file__).parent.resolve() / "data" / "naneye.txt"

def read_naneye(data_file: Union[str, Path]):
    data_file_path : Path = Path(data_file)
    
    data_array = np.loadtxt(data_file_path, dtype=np.uint8)
    
    
    data_array = data_array[1:-3]
    
    
    last = None
    missing = []
    for num, line in enumerate(data_array):
        current = line[0]*256+line[1]
        
        print(num, current)
        
        if last != None: 
            if current != last+1:
                print(f"Missing {last}, {current}")
                missing += range(last, current-1) # Start indexing at zero
             
        last = current

    for line_number in missing:
        print(f"Interpolating missing line {line_number}")
        data_array = np.insert(data_array, line_number, data_array[line_number-2], axis=0)
        
    data_array = data_array[:,2:]
    
    data_array = cv2.cvtColor(data_array, cv2.COLOR_BAYER_GRBG2BGR)
    
    return data_array

def test_read():
    
    img = read_naneye(test_img)
    
    print(img)
    print("Shape: ", "x".join((str(x) for x in img.shape)))
    
    cv2.imwrite("test.jpg", img)
    
if __name__ == "__main__":
    test_read()