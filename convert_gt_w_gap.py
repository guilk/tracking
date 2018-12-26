import os
import numpy as np

if __name__ == '__main__':
    data_file = './gt.txt'
    gt_data = np.loadtxt(data_file, delimiter=' ')
    print gt_data.shape
    pass