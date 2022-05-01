# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2016-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr


import threading
import time
import timeit
import atexit
import weakref
import pyopencl as cl


class CLWorkerThread(threading.Thread):
    """
    Generic Worker thread for parallel OpenCL execution. This class is an abstract class, doing nothing, and
    should be derived.
    """
    _instances = []
    def __init__(self, cl_dev=None, verbose=False):
        """
        Constructor.
        
        Args:
            dev: the pyopencl.Device this thread will use
            verbose: if True, will report when jobs are submitetd/finished
            cl_ctx: if passed, the thread will use this pyopencl.Context instead of creating one from
        """
        self.__class__._instances.append(weakref.ref(self))
        super(CLWorkerThread, self).__init__()
        self.cl_dev = cl_dev
        self.verbose = verbose
        self.event_start = threading.Event()
        self.event_finished = threading.Event()
        self.is_init = threading.Event()
        self.is_init.clear()
        self.join_flag = False
        self.bug_apple_cpu_workgroupsize_warning = True
        if verbose:
            print("Thread %s: finished __init__" % (self.name))

    def run(self):
        """
        Infinite loop waiting for a task to be submitted.
        The parent handler should first add any relevant data to this object, and then use eventStart.set() to begin the job,
        then wait using eventFinished.wait().
        Set join_flag=True, then eventStart.set() to get out of the loop and allow joining (finishing) this thread. This should
        be done in the parent destructor (or terminating code)
        
        Returns:
            Nothing
        """
        try_ctx = 5
        while try_ctx > 0:
            try:
                self.cl_ctx = cl.Context([self.cl_dev])
                try_ctx = 0
            except:
                try_ctx -= 1
                print("Problem initializing OpenCL context... #try%d/5, wait%3.1fs" % (5 - try_ctx, 0.1 * (5 - try_ctx)))
                time.sleep(0.1 * (5 - try_ctx))
        self.cl_queue = cl.CommandQueue(self.cl_ctx)

        self.init_cl()

        self.is_init.set()
        self.event_finished.set()

        while True:
            self.event_start.wait()
            if self.join_flag: break
            if self.verbose:
                t0 = timeit.default_timer()
                print(self.name, " ...start working...")

            self.work()

            self.event_start.clear()
            self.event_finished.set()

            if self.verbose:
                print(self.name, " ...finished work... (dt=%7.4fs)"%(timeit.default_timer()-t0))

    def init_cl(self):
        """
        Real initialization of all opencl kernels and buffers needed for calculations should be done here.
        
        Returns:
                Nothing
        """

    def work(self):
        """
        Real work to be performed should be done here.
        Returns:
            Nothing
        """

    def join(self, timeout=None):
        self.event_finished.wait()
        self.join_flag = True
        self.event_start.set()
        super(CLWorkerThread, self).join(timeout)



@atexit.register
def stop_all_workers():
    if len(CLWorkerThread._instances) == 0:
        return
    print("@Exit: joining all OpenCL worker threads")
    vt=[]
    for o in CLWorkerThread._instances:
        t = o()
        if t is not None:
            vt.append(t)
    for t in vt:
        t.event_finished.wait()
        t.join_flag = True
        t.event_start.set()
    for t in vt:
        t.join(2)
