import numpy as np
import pylab as pl
import re
import itertools
import ConfigParser

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

def plot_cmd(accel, labels, command, rep):
    """Plot a single repetition of a command"""
    pl.title('Command %d' % command)
    data = accel[labels == command][rep, :]
    pl.plot(np.arange(data.shape[1]), data[0,:], label='x', c='r')
    pl.plot(np.arange(data.shape[1]), data[1,:], label='y', c='g')
    pl.plot(np.arange(data.shape[1]), data[2,:], label='z', c='b')
    pl.xlabel('Time (samples)')
    pl.ylabel('Acceleration (uncalibrated)')
    pl.legend()

def plot_sample(accel, sid):
    """Plot a single repetition of a command"""
    pl.title('Sample %d' % sid)
    data = accel[sid]
    pl.plot(np.arange(data.shape[1]), data[0,:], label='x', c='r')
    pl.plot(np.arange(data.shape[1]), data[1,:], label='y', c='g')
    pl.plot(np.arange(data.shape[1]), data[2,:], label='z', c='b')
    pl.legend()

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

def load_calibration_from_properties(filename):
    # Add fake section header to allow ConfigParser to read java .properties
    # http://stackoverflow.com/questions/2819696/parsing-properties-file-in-python/2819788#2819788
    class FakeSecHead(object):
      def __init__(self, fp):
        self.fp = fp
        self.sechead = '[fakehead]\n'
      def readline(self):
        if self.sechead:
          try: return self.sechead
          finally: self.sechead = None
        else: return self.fp.readline()

    config = ConfigParser.RawConfigParser()
    config.readfp(FakeSecHead(open(filename)))
    d = dict(config.items('fakehead'))
    d = {k:float(v) for k, v in d.items()}
    return {'gain': np.array([d['gain_x'], d['gain_y'], d['gain_z']]),
            'offset': np.array([d['offset_x'], d['offset_y'], d['offset_z']])}
