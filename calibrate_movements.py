##
import numpy as np
import pylab as pl
import utils
import sys
import os
import re
##
#dirname = sys.argv[1]
dirname = '/home/julien/work/madsdf/my/data/movements_hector_02_11_2013'
calib_dirname = '/home/julien/work/madsdf/my/data/calibrations'
reload(utils); utils.load_calibration_from_properties('/home/julien/work/madsdf/my/data/calibrations/1_5_9EDB.accel.properties')
##
MOVEMENT_REGEXP = r'(\w+)_movement_\d+_\d+.txt'
mvtfiles = os.listdir(dirname)
mvtfiles = filter(lambda f: re.match(MOVEMENT_REGEXP, f), mvtfiles)
mvtfiles = sorted(mvtfiles)
all_accel = []
all_gyro = []
all_labels = []
for fname in mvtfiles:
    with open(os.path.join(dirname, fname)) as f:
        accel, gyro, labels = utils.load_data(f)
    shimmerid = re.match(MOVEMENT_REGEXP, fname).group(1)
    print shimmerid
    acalib = utils.load_calibration_from_properties(os.path.join(calib_dirname,
        '1_5_%s.accel.properties' % shimmerid))
    gcalib = utils.load_calibration_from_properties(os.path.join(calib_dirname,
        '1_5_%s.gyro.properties' % shimmerid))
    accel = 9.81 * (accel - acalib['offset'][:,None]) / acalib['gain'][:,None]
    gyro = (gyro - gcalib['offset'][:,None]) / gcalib['gain'][:,None]
    all_accel.append(accel)
    all_gyro.append(gyro)
    all_labels.append(labels)
##
all_accel = np.squeeze(np.array(all_accel))
all_gyro = np.squeeze(np.array(all_gyro))
all_labels = np.squeeze(np.array(all_labels))
##
outfname = os.path.join(dirname, 'out_calib.txt')
with open(outfname, 'w') as f:
    utils.write_data(f, all_accel, all_gyro, all_labels)
##
