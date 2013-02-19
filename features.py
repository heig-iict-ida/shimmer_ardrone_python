## Load features
BASEDIR = '/home/julien/work/madsdf/gregoire/dev/etude_mouvement/2_FEATURES/'\
        + '2_capteurs/mouvements_plus multi/droite'

featuresNames = ['difference', 'ecartType', 'max', 'mean', 'median', 'min',
                 'variance']
features = {}

for featn in featuresNames:
    fname = os.path.join(BASEDIR, '%s.txt'%featn)
    fdata = np.genfromtxt(fname, skip_header=1)
    features[featn] = fdata
    assert accel.shape[0] == fdata.shape[0]
## Features boxplot grouped by commands
for command in np.unique(labels):
    pl.figure()
    pl.suptitle('Command %d' % command)
    nfeat = len(features.keys())
    r = int(np.ceil(np.sqrt(nfeat)))
    for i, featname in enumerate(features.keys()):
        pl.subplot(r, r, i + 1)
        pl.title(featname)
        data = features[featname][labels == command]
        pl.boxplot(data)
        pl.setp(pl.gca(), 'xticklabels', ['ax', 'ay', 'az', 'gx', 'gy', 'gz'])
## Features boxplot grouped by feature
for featname in features.keys():
    pl.figure()
    pl.suptitle('%s' % featname)
    ncmd = len(np.unique(labels))
    r = int(np.ceil(np.sqrt(ncmd)))
    c = int(np.round(np.sqrt(ncmd)))
    for i, command in enumerate(np.unique(labels)):
        pl.subplot(r, c, i + 1)
        pl.title('command %d' % command)
        data = features[featname][labels == command]
        pl.boxplot(data)
        pl.setp(pl.gca(), 'xticklabels', ['ax', 'ay', 'az', 'gx', 'gy', 'gz'])

##
def per_column_normalization(data):
    arr = data - bn.nanmean(data, axis=0)
    std = bn.nanstd(arr, axis=0)
    std[std == 0] = 1 # avoid division by 0
    return arr / std

## Pcolor of per-command features
for command in np.unique(labels):
    pl.figure()
    pl.suptitle('Command %d' % command)
    nfeat = len(features.keys())
    commands = np.flatnonzero(labels == command)
    M = np.zeros((len(commands), nfeat*6), dtype=float)
    xlabels = []
    for i, c in enumerate(commands):
        for j, featname in enumerate(features.keys()):
            a_min = np.min(features[featname][:,:3])
            a_max = np.max(features[featname][:,:3])
            g_min = np.min(features[featname][:,3:])
            g_max = np.max(features[featname][:,3:])
            print featname, ' amin : ', a_min, ' amax : ', a_max
            #M[i, 6*j:6*j+6] = features[featname][c]
            M[i, 6*j:6*j+3] = (features[featname][c,:3] - a_min) / (a_max - a_min)
            M[i, 6*j+3:6*j+6] = (features[featname][c,3:] - g_min) / (g_max - g_min)
            xlabels += [featname + ' ' + a
                        for a in ['ax', 'ay', 'az', 'gx', 'gy', 'gz']]

    #M /= np.std(M, axis=0)
    #M = per_column_normalization(M)
    pl.imshow(M.T, interpolation='none', aspect='auto')
    #pl.pcolor(M)
    pl.colorbar()
    pl.yticks(np.arange(M.shape[1]), xlabels)
    #break

