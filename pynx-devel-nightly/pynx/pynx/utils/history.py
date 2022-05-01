# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2016-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr

from collections import OrderedDict
import time
import timeit
import numpy as np


class PynxOrderedDict(OrderedDict):
    """
    OrderedDict with easy access to the last value.
    """

    def last_value(self):
        if len(self) == 0:
            return None
        else:
            return self[next(reversed(self))]

    def last_key(self):
        if len(self) == 0:
            return None
        else:
            return next(reversed(self))

    def as_numpy_record_array(self, title='data'):
        """
        Return dictionary as a numpy record array. Strings are converted to ASCII for h5py compatibility
        :return: the numpy record array, with two entries (cycle, value) per position
        """
        kv = []
        for k, v in self.items():
            # Special handling for values like (2,2,2) condensed as 222
            real_array = isinstance(v, np.ndarray)
            if real_array:
                if v.ndim < 1:
                    real_array = False
            if isinstance(v, tuple) or isinstance(v, list) or real_array:
                if len(v) == 2:
                    v = np.int16("%d%d" % (v[0], v[1]))
                elif len(v) == 3:
                    v = np.int16("%d%d%d" % (v[0], v[1], v[2]))
            kv.append((k, v))

        a = np.rec.array(kv, names=('cycle', title))
        tt = []
        for k, s in a.dtype.descr:
            if 'U' in s:
                s = s.replace('U', 'S')
            tt.append((k, s))
        return a.astype(tt)


class History(dict):
    """
    Class to record optimization history. It is used to store the parameters like the algorithm,
    negative log-likelihood (llk), chi^2, resolution, cputime, walltime as a function of the cycle number.
    Not all values need be stored for all cycles.
    The default fields initialized as python OrderedDict (the key being the cycle number) are:
    
      - 'time': records timeit.default_timer()
      - 'epoch': records time.time()
    """

    def __init__(self):
        super(dict, self).__init__()
        for k in ['time', 'epoch']:
            self[k] = PynxOrderedDict()
        # This records the beginning of the first algorithm
        # To be useful, insert() should be called at the beginning of the algorithms
        self.t0 = timeit.default_timer()

    def insert(self, cycle, **kwargs):
        """
        Store new values. if keys do not already exist, they are automatically added. 'time' and 'epoch' keys
        are automatically recorded, so need not be supplied.
        Args:
            cycle: the current cycle
            **kwargs: e.g. llk=2e4, algorithm='ML-Poisson'

        Returns:
            Nothing.
        """
        self['time'][cycle] = timeit.default_timer()
        self['epoch'][cycle] = time.time()
        for k, v in kwargs.items():
            if k not in self:
                self[k] = PynxOrderedDict()
            if v is str:
                self[k][cycle] = ascii(v)
            else:
                self[k][cycle] = v

    def as_numpy_record_array(self, *args):
        """
        Get stored values for one or several keys (e.g. 'time', 'llk', 'nb_photons') as a numpy record array.
        The first entry is always the cycle number. If entries are missing for a given key and a cycle number, it
        is replaced by the next recorded value.

        :param args: all the desired keys, e.g.: 'time', 'epoch', 'llk', 'algorithm'... If no args are given, all
          available keys are returned.
        :return: numpy record array, see https://docs.scipy.org/doc/numpy/user/basics.rec.html#record-arrays,
          or None if no history has been recorded
        """
        v = [[x] for x in self['time'].keys()]
        if len(v) == 0:
            return None
        if len(args) == 0:
            args = self.keys()
        # This is slow, but may be enough
        for k in args:
            last_value = list(self[k].values())[-1]
            # We use reversed because the first cycle is usually anomalous
            for vx in reversed(v):
                if vx[0] in self[k]:
                    last_value = self[k][vx[0]]
                real_array = isinstance(last_value, np.ndarray)
                if real_array:
                    if last_value.ndim < 1:
                        real_array = False
                if isinstance(last_value, tuple) or isinstance(last_value, list) or real_array:
                    if len(last_value) == 2:
                        last_value = np.int16("%d%d" % (last_value[0], last_value[1]))
                    elif len(last_value) == 3:
                        last_value = np.int16("%d%d%d" % (last_value[0], last_value[1], last_value[2]))

                vx.append(last_value)
        a = np.rec.array(v, names=['cycle'] + [k for k in args])
        # Convert unicode to ASCII arrays for h5py compatibility
        tt = []
        for k, s in a.dtype.descr:
            if 'U' in s:
                s = s.replace('U', 'S')
            tt.append((k, s))
        return a.astype(tt)


if __name__ == '__main__':
    h = History()
    for i in range(5):
        h.insert(i, llk=5.0, nb_photons=1e6)
        h.insert(i, algorithm='AP')
    h.insert(10, llk=5.0, nb_photons=1e6)
    h.insert(15, algorithm='DM')
    a = h.as_numpy_record_array('llk', 'nb_photons', 'time', 'algorithm')
    b = h['algorithm'].as_numpy_record_array()
