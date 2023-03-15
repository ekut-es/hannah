import pytest

from pathlib import Path 
from typing import Union

import numpy as np

cv2 = pytest.importorskip('cv2')

from hannah.dataset.vision.utils.naneye import read_naney

test_img = Path(__file__).parent.resolve() / "data" / "naneye.txt"

def test_read():
    
    img = read_naneye(test_img)
    
    print(img)
    print("Shape: ", "x".join((str(x) for x in img.shape)))
    
    cv2.imwrite("test.jpg", img)
    
if __name__ == "__main__":
    test_read()