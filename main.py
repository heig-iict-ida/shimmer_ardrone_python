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
import scipy.stats as stats
import itertools
import matplotlib.colors as mpcolors

import utils
## Load data
BASEDIR = '/home/julien/work/madsdf/gregoire/dev/etude_mouvement/1_RAWDATA/'\
        + '2_capteurs/mouvements_plus multi/'

#with open(os.path.join(BASEDIR, 'droite.txt')) as f:
    #lines = f.readlines()
#datafilename = '/home/julien/work/madsdf/my/data/merged_calib.txt'
#datafilename = '/home/julien/work/madsdf/my/data/left/movements_90deg/out.txt'
#datadir = '/home/julien/Dropbox/madsdf/data/right/repetitive_movements/'
datadir = '/home/julien/Dropbox/madsdf/data/left/repetitive_movements/'
datafiles = os.listdir(datadir)
datafiles = filter(lambda f: f.endswith('.txt') and 'movement' in f, datafiles)

lines = []
for fname in datafiles:
    with open(os.path.join(datadir, fname)) as f:
        lines += f.readlines()
origids = np.arange(len(datafiles))
accel, gyro, labels = utils.load_data(iter(lines))
assert origids.shape[0] == accel.shape[0]

ulabels = np.unique(labels)
nlabels = np.max(ulabels)
if False:
    pl.figure()
    pl.title('Template counts per label')
    counts, bins, patches = pl.hist(labels, bins=nlabels)
    pl.xticks(bins + 0.5, np.arange(np.min(ulabels), np.max(ulabels)+1))
##
if False:
    c = np.ceil(np.sqrt(np.max(nlabels)))
    r = np.floor(np.sqrt(np.max(nlabels)))
    for axis in [0, 1, 2]:
        pl.figure()
        pl.suptitle('Axis %d' % axis)
        for i in nlabels:
            idx = np.flatnonzero(labels == i)
            pl.subplot(c, r, i)
            pl.title('Command %d' % i)
            pl.plot(accel[idx, axis, :].T)
            #for j in idx:
                #pl.plot(accel[j, axis, :])
## Filtering of accel
faccel = bn.move_median(accel, 10, axis=-1)
faccel[np.isnan(faccel)] = accel[np.isnan(faccel)]
if False:
    for cmd, rep in [(1, 0), (2,3), (5, 2), (1, 2)]:
        pl.figure()
        pl.suptitle('cmd %d rep %d' % (cmd, rep))
        pl.subplot(211)
        utils.plot_cmd(accel, labels, cmd, rep)
        pl.subplot(212)
        utils.plot_cmd(faccel, labels, cmd, rep)
##
if False:
    right_cmd = [(1, 'GOFORWARD'), (2, 'GORIGHT'), (3, 'GOTOP'),
                 (4, 'GODOWN'), (5, 'NOTHING'), (8, 'ROTATERIGHT')]
    left_cmd = [(1, 'GOBACKWARD'), (2, 'GOLEFT'), (3, 'ROTATELEFT'), (4, 'NOTHING')]
    for cmd, cmdname in left_cmd:
        pl.figure(figsize=(24, 12.425))
        utils.plot_cmd(accel, labels, cmd, 1)
        pl.title('%s (%d)' % (cmdname, cmd))
        pl.tight_layout()
        pl.savefig('left_%s.png' % cmdname.lower())
        pl.close()
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
origids = origids[order]
##
distfn = functools.partial(mlpy.dtw_std, dist_only=True)
#distfn = dtw.fast.dtw_fast
#distfn = spdist.euclidean
DM = np.zeros((accel.shape[0], accel.shape[0]), dtype=float)
DM[:] = np.nan
##
if True:
    for i in xrange(faccel.shape[0]):
        for j in xrange(i+1, faccel.shape[0]):
            if i != j:
                DM[i,j] = DM[j,i] = dist_3_axis(faccel[i], faccel[j], distfn)
        if i % 50 == 0:
            print i
##
if False:
    for i in xrange(accel.shape[0]):
        for j in xrange(i+1, accel.shape[0]):
            if i != j:
                DM[i,j] = DM[j,i] = dist_3_axis(accel[i], accel[j], distfn)
        if i % 50 == 0:
            print i
##
if False:
    DM = np.zeros((accel.shape[0], accel.shape[0]), dtype=float)
    DM[:] = np.nan
    for i in xrange(accel.shape[0]):
        for j in xrange(i+1, accel.shape[0]):
            if i != j:
                DM[i,j] = DM[j,i] = dtw.fast.dtw_fast(accel[i], accel[j])
        if i % 50 == 0:
            print i
##
def make_ranges(serie):
    """
    Group consecutive equal entries into ranges.
    For example, [0, 2, 2, 2, 3, 3, 1, 1] would return
    [(0,0,1), # 0 in [0,1[
     (2,1,4), # 2 in [1,4[
     (3,4,6), # 3 in [4,6[
     (1,6,8)] # 1 in [6,8[
    """
    ranges = []
    cur_len = 0
    for i,e in enumerate(serie):
        if i > 0 and e != serie[i-1]:
            # Finish previous range
            ranges.append((serie[i - 1], i - cur_len, i))
            cur_len = 1
        else:
            cur_len += 1

    if cur_len != 0:
        ranges.append((serie[-1], len(serie)-cur_len, len(serie)))

    return ranges

labranges = make_ranges(labels)

fig = pl.figure()
plot = fig.add_subplot(111)
pl.title('DTW Distance between gesture templates')
plot.tick_params(axis='both', which='major', labelsize=8)
plot.tick_params(axis='both', which='minor', labelsize=8)
vmax = stats.scoreatpercentile(np.ravel(DM[np.isfinite(DM)]), 95)
#pl.imshow(DM, interpolation='none', aspect='auto', origin='lower', vmax=vmax)
pl.imshow(DM, interpolation='none', aspect='auto', origin='lower',
          norm=mpcolors.LogNorm())
xticks = ['%d\n(%d)'%(i, labels[i]) for i in xrange(len(labels))]
#pl.xticks(np.arange(len(xticks)), xticks)
#pl.yticks(np.arange(len(xticks)), xticks, rotation=90)

# Show label ranges
for lr in labranges:
    pl.axvline(lr[1] - 0.5, lw=2, color='w')
    pl.axhline(lr[1] - 0.5, lw=2, color='w')

labranges_ticks = ['%d'%lr[0] for lr in labranges]
labranges_ticks_pos = [(lr[1] + lr[2]) / 2 for lr in labranges]
pl.xticks(labranges_ticks_pos, labranges_ticks, fontsize=20)
pl.yticks(labranges_ticks_pos, labranges_ticks, rotation=90, fontsize=20)

pl.colorbar()
pl.tight_layout()
##

stranges = [19, 20, 21, 22, 23, 24]
np.argsort(DM[:,19])

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
## Write a new file
with open('out.txt', 'w') as f:
    utils.write_data(f, accel, gyro, labels)

##
sample1 = 18
sample2 = 151

dist, cost, path = mlpy.dtw_std(accel[sample1,0,:], accel[sample2,0,:],
                                dist_only=False)
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
