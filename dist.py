##
import numpy as np
import pylab as pl
import os.path
import re
import bottleneck as bn
import mlpy
import scipy.spatial.distance as spdist
import functools
import dtw
import dtw.fast
import itertools
##
def load_data(lineiter):
    labels = []
    accel = []
    gyro = []
    def parse_sensor(headername):
        vec = []
        for i, c in enumerate(['X', 'Y', 'Z']):
            dataline = next(lineiter)
            assert dataline.startswith('%s %s' % (headername, c))
            semicolon_idx = dataline.index(':')
            vec.append(np.fromstring(dataline[semicolon_idx+1:], sep=';'))
        return np.array(vec)

    cmd_re = re.compile(r'COMMAND\s+(?P<cmd>\d+)\s+SAMPLE\s+(?P<sample>\d+)')
    while lineiter:
        try:
            line = next(lineiter)
        except StopIteration:
            break

        assert line.startswith('COMMAND')
        m = re.match(cmd_re, line)
        cmdnum = int(m.groupdict()['cmd'])
        samplenum = int(m.groupdict()['sample'])
        labels.append(cmdnum)
        a = parse_sensor('Accel')
        g = parse_sensor('Gyro')
        accel.append(a)
        gyro.append(g)

        print line, ' => cmdnum : ', cmdnum

    accel = np.array(accel)
    gyro = np.array(gyro)
    labels = np.array(labels)
    return accel, gyro, labels

## Load data
BASEDIR = '/home/julien/work/madsdf/gregoire/dev/etude_mouvement/1_RAWDATA/'\
        + '2_capteurs/mouvements_plus multi/'

#with open(os.path.join(BASEDIR, 'droite.txt')) as f:
    #lines = f.readlines()
#with open('out_1-4.txt') as f:
#    lines = f.readlines()
#with open('out.txt') as f:
    #lines = f.readlines()
with open('/home/julien/work/madsdf/my/data/movements_02_08_2013/out.txt') as f:
    lines = f.readlines()

accel, gyro, labels = load_data(iter(lines))

##
def plot_cmd(accel, labels, command, rep):
    """Plot a single repetition of a command"""
    pl.title('Command %d' % command)
    data = accel[labels == command][rep, :]
    pl.plot(np.arange(data.shape[1]), data[0,:], label='x', c='r')
    pl.plot(np.arange(data.shape[1]), data[1,:], label='y', c='g')
    pl.plot(np.arange(data.shape[1]), data[2,:], label='z', c='b')
    pl.legend()
## Filtering of accel
faccel = bn.move_median(accel, 10, axis=-1)
faccel[np.isnan(faccel)] = accel[np.isnan(faccel)]
if False:
    for cmd, rep in [(1, 0), (2,3), (5, 6), (1, 4)]:
        pl.figure()
        pl.suptitle('cmd %d rep %d' % (cmd, rep))
        pl.subplot(211)
        plot_cmd(accel, labels, cmd, rep)
        pl.subplot(212)
        plot_cmd(faccel, labels, cmd, rep)
##
def get_serie(accel, command, rep):
    return accel[labels == command][rep, :]

#def dtw_3_axis(serie1, serie2):
    #total = 0.0
    #for ax in xrange(3):
        #total += mlpy.dtw_std(serie1[ax,:], serie2[ax,:], dist_only=True)
    #return total

def dist_3_axis(serie1, serie2, distfn):
    # TODO: This is wrong, need to do DTW on all 3 axis at the same time
    # (we should use the same path on all axis)
    total = 0.0
    for ax in xrange(3):
        total += distfn(serie1[ax,:], serie2[ax,:])
    return total

##
# Sort by labels
order = np.argsort(labels)
labels = labels[order]
accel = accel[order]
faccel = faccel[order]
##
#distfn = functools.partial(mlpy.dtw_std, dist_only=True)
#distfn = dtw.fast.dtw_fast
distfn = spdist.euclidean
DM = np.zeros((accel.shape[0], accel.shape[0]), dtype=float)
DM[:] = np.nan
for i in xrange(accel.shape[0]):
    for j in xrange(i+1, accel.shape[0]):
        if i != j:
            DM[i,j] = DM[j,i] = dist_3_axis(accel[i], accel[j], distfn)
    if i % 50 == 0:
        print i
##
DM = np.zeros((accel.shape[0], accel.shape[0]), dtype=float)
DM[:] = np.nan
for i in xrange(accel.shape[0]):
    for j in xrange(i+1, accel.shape[0]):
        if i != j:
            DM[i,j] = DM[j,i] = dtw.fast.dtw_fast(accel[i], accel[j])
    if i % 50 == 0:
        print i
##
pl.figure()
pl.imshow(DM, interpolation='none', aspect='auto', origin='lower')
xticks = ['%d\n(%d)'%(i, labels[i]) for i in xrange(len(labels))]
pl.xticks(np.arange(len(xticks)), xticks)
#pl.xticks(np.arange(len(labels)), labels)
pl.yticks(np.arange(len(xticks)), xticks, rotation=90)
pl.colorbar()
pl.tight_layout()
##
def plot_sample(accel, sid):
    """Plot a single repetition of a command"""
    pl.title('Sample %d' % sid)
    data = accel[sid]
    pl.plot(np.arange(data.shape[1]), data[0,:], label='x', c='r')
    pl.plot(np.arange(data.shape[1]), data[1,:], label='y', c='g')
    pl.plot(np.arange(data.shape[1]), data[2,:], label='z', c='b')
    pl.legend()
##

# There are about 60 to 70 samples per label. So we use 60-NN to relabelize
# our points
from collections import Counter
def knn_DM(DM, sampleid):
    # 60 nearest neighbors labels
    lab60 = labels[np.argsort(DM[:,sampleid])[:60]]
    c = Counter(lab60)
    # Return most common label. Make sure that ratio between second most common
    # and most common is < 0.8
    counts = c.most_common(2)
    if counts[1][1] / float(counts[0][1]) > 0.8:
        print 'Ambiguous label for sampleid = %d : counts : %s' % (sampleid,
                counts)
    #print counts
    return counts[0][0]
##
newlabels = np.array([knn_DM(DM, sampleid) for sampleid in xrange(len(labels))])

modified = np.flatnonzero(newlabels - labels)
confirmed = np.flatnonzero(newlabels - labels == 0)
## Write a new file

def write_data(f, accel, gyro, labels):
    order = np.argsort(labels)
    labels = labels[order]
    accel = accel[order]
    gyro = gyro[order]
    for k, g in itertools.groupby(range(len(labels)), key=lambda i: labels[i]):
        for sampleid, i in enumerate(g):
            f.write('COMMAND %d SAMPLE %d\n' % (k, sampleid + 1))
            def _w_axis(data, axname, axnum):
                f.write('%s : ' % axname)
                f.write(';'.join(['%.17g'%d for d in data[i, axnum, :]]) + '\n')
            _w_axis(accel, 'Accel X', 0)
            _w_axis(accel, 'Accel Y', 1)
            _w_axis(accel, 'Accel Z', 2)
            _w_axis(gyro, 'Gyro X', 0)
            _w_axis(gyro, 'Gyro Y', 1)
            _w_axis(gyro, 'Gyro Z', 2)
##
with open('out.txt', 'w') as f:
    write_data(f, accel[confirmed], gyro[confirmed], labels[confirmed])

##
cmd14 = []
for cmd in (1,2,3,4):
    cmd14 += np.flatnonzero(labels == cmd).tolist()
with open('out_1-4.txt', 'w') as f:
    write_data(f, accel[cmd14], gyro[cmd14], labels[cmd14])
##
sample1 = 18
sample2 = 151

dist, cost, path = mlpy.dtw_std(accel[sample1,0,:], accel[sample2,0,:], dist_only=False)
pl.figure()
pl.suptitle('dist = %f' % dist)
pl.subplot(211); pl.title('%d'%sample1); pl.plot(accel[sample1,0,:]); pl.ylim(0, 5000); pl.subplot(212); pl.title('%d'%sample2); pl.plot(accel[sample2,0,:]); pl.ylim(0,5000);

pl.figure()
pl.title('%d - %d' % (sample1, sample2))
pl.imshow(cost.T, origin='lower', cmap=cm.gray, interpolation='nearest')
pl.plot(path[0], path[1], 'w')
pl.xlim((-0.5, cost.shape[0]-0.5))
pl.ylim((-0.5, cost.shape[1]-0.5))
##
