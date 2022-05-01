# -*- coding: utf-8 -*-

# PyNX - Python tools for Nano-structures Crystallography
#   (c) 2018-present : ESRF-European Synchrotron Radiation Facility
#       authors:
#         Vincent Favre-Nicolin, favre@esrf.fr


import socket
import time
import timeit
import sys
import os
import subprocess
import gc
import sqlite3
import numpy as np
from pynx.utils.matplotlib import pyplot as plt
from pynx.processing_unit import has_cuda, has_opencl, default_processing_unit
from pynx.utils.math import test_smaller_primes

if has_opencl:
    from pynx.processing_unit import opencl_device, cl_processing_unit
    import pyopencl as cl
    import pyopencl.array as cla
    import pyopencl.tools as cl_tools
    from pynx.processing_unit.opencl_device import cl_device_fft_speed
if has_cuda:
    from pynx.processing_unit import cuda_device, cu_processing_unit
    import pycuda.driver as cu_drv
    import pycuda.gpuarray as cua
    from pynx.processing_unit.cuda_device import cuda_device_fft_speed
from pynx.scattering.test import mrats


class SpeedTest(object):
    """
    Class for speed tests using either GPU or CPU
    """

    def __init__(self, gpu_name, language, cl_platform=None, verbose=True):
        """

        :param gpu_name: the gpu name to be tested, or 'CPU'.
        :param language: either 'cuda', 'opencl' or 'CPU'
        """
        if gpu_name is not None:
            self.gpu_name = gpu_name.lower()
        else:
            self.gpu_name = ''
        if language is not None:
            self.language = language.lower()
        else:
            self.language = ''
        self.cl_platform = cl_platform
        self.results = {}
        self.results['hostname'] = socket.gethostname()
        self.results['epoch'] = time.time()
        self.results['language'] = self.language
        self.pu = default_processing_unit
        # TODO: handle case where language is not given, CPU
        self.db_results = {}
        self.db_conn = None
        self.db_curs = None
        self.prepare_processing_unit()

    def prepare_processing_unit(self):
        """
        Prepare the processing unit
        :return:
        """
        if self.language == 'cuda':
            self.pu = cu_processing_unit.CUProcessingUnit()
            self.pu.init_cuda(gpu_name=self.gpu_name, verbose=False, test_fft=False)
            self.results['GPU'] = self.pu.cu_device.name()
            # TODO: the following should not be necessary to avoid automatic searching for GPU
            default_processing_unit.set_device(self.pu.cu_device, verbose=False, test_fft=False)

        if self.language == 'opencl':
            self.pu = cl_processing_unit.CLProcessingUnit()
            self.pu.init_cl(gpu_name=self.gpu_name, verbose=False, test_fft=False)
            self.results['GPU'] = '%s [%s]' % (self.pu.cl_device.name, self.pu.cl_device.platform.name)
            # TODO: the following should not be necessary to avoid automatic searching for GPU
            default_processing_unit.set_device(self.pu.cl_device, verbose=False, test_fft=False)

    def prepare_db(self, db_name="pynx_speed.db"):
        """
        Create database file and check that it has all the necessary columns
        :param db_name: the name of the sqlite3 database file
        :return: nothing
        """
        self.db_conn = sqlite3.connect(db_name)
        self.db_curs = self.db_conn.cursor()

        self.db_curs.execute('''CREATE TABLE IF NOT EXISTS pynx_speed_test
                     (epoch real, hostname text, language text, GPU text)''')

        for k, v in self.results.items():
            try:
                k = k.replace("*", "_")
                k = k.replace("=", "_")
                self.db_conn.execute('ALTER TABLE pynx_speed_test ADD COLUMN %s;' % k)
            except:
                # column already existed
                pass

    def test_mem_host2device(self, size, pinned=True, parallel=True, nb=1, multiplicity=1, nb_test=3,
                             detailed_timing=False, verbose=False):
        """
        Test the transfer speed between host (RAM) to and from GPU memory, optionally using pinned memory
        and parallel data transfers.
        :param size: the size of the array, i.e. the number of float32 elements to transfer
        :param pinned: if True, used pinned memory for faster transfers
        :param parallel: if True, use parallel queues/streams to perform host->device and device->host in parallel.
        :param nb: number of arrays to allocate in GPU, to test the maximum amount of memory and number of pinned arrays
        :param multiplicity: number of arrays allocated in host memory per array in GPU memory. This is used to test
                             how much pinned memory can be allocated on both sides. [CUDA only]
        :param nb_test: number of memory transfers to test (default=3), best timing is reported.
        :param verbose: if True, print some detailed progress info
        :return: The elapsed time for the calculation. The result is added to self.results.
        """
        s = "%dM" % int(round(4 * size / 1024 ** 2))
        gbytes, dt = 0, 0
        if True:  # try:
            if 'opencl' in self.language:
                pu = self.pu

                mf = cl.mem_flags
                allocator = cl_tools.DeferredAllocator(pu.cl_ctx, mf.READ_WRITE | mf.ALLOC_HOST_PTR)
                va = []
                vad = []
                if verbose:
                    print("Allocating memory, %d chunks of %8.2f Mbyte [GPU], with %d copies in host memory "
                          "[pinned=%s]..." % (nb, size * 4 / 2 ** 20, multiplicity, str(pinned)))
                # Only compute once chunk. Not constant value in case some procedure tries to be smart
                # chunk = np.random.uniform(0, 1, (size,)).astype(np.float32)
                chunk = np.linspace(0, np.pi, size, dtype=np.float32)
                for ii in range(nb):
                    va.append([])
                    if pinned:
                        # remap host array so that it is pinned
                        # This works on AMD R9, Iris pro, not Titan ?
                        # ad = cla.to_device(pu.cl_queue, a)
                        # a = ad.map_to_host(queue=pu.cl_queue, flags=cl.mem_flags.ALLOC_HOST_PTR)
                        vad.append(cla.to_device(pu.cl_queue, chunk, allocator=allocator))
                        for jj in range(multiplicity):
                            va[-1].append(vad[-1].map_to_host())
                    else:
                        for jj in range(multiplicity):
                            va[-1].append(chunk.copy())
                        vad.append(cla.to_device(pu.cl_queue, va[-1][-1]))
                    if verbose:
                        m = vad[-1].nbytes / 2 ** 20 * (ii + 1)
                        print('Allocated: %8.0f Mbyte[GPU] %8.0f Mbyte[host]' % (m, m * multiplicity))

                if parallel:
                    if verbose:
                        print("Allocating 2nd set of arrays for parallel in/out transfers")
                    vbd = []
                    vb = []
                    queue_h2d = pu.cl_queue
                    queue_d2h = cl.CommandQueue(pu.cl_ctx)
                    b = np.random.uniform(0, 1, size).astype(np.float32)
                    for ii in range(nb):
                        vb.append([])
                        if pinned:
                            vbd.append(cla.to_device(pu.cl_queue, b.copy(), allocator=allocator))
                            for jj in range(multiplicity):
                                vb[-1].append(vbd[-1].map_to_host())
                        else:
                            for jj in range(multiplicity):
                                vb[-1].append(chunk.copy())
                            vbd.append(cla.to_device(pu.cl_queue, vb[-1][-1]))
                        if verbose:
                            m = vbd[-1].nbytes / 2 ** 20 * (ii + 1 + nb)
                            print('Allocated: %8.0f Mbyte[GPU] %8.0f Mbyte[host]' % (m, m * multiplicity))
                else:
                    queue_d2h = pu.cl_queue
                    queue_h2d = pu.cl_queue
                    vb = va
                    vbd = vad

                if verbose:
                    print("Beginning test transfers...")

                pu.cl_queue.finish()
                dt = 0
                for i in range(nb_test):
                    t0 = timeit.default_timer()
                    for ii in range(nb):
                        for jj in range(multiplicity):
                            cl.enqueue_copy(queue_h2d, dest=vad[ii].data, src=va[ii][jj].data, is_blocking=False)
                            cl.enqueue_copy(queue_d2h, dest=vb[ii][jj].data, src=vbd[ii].data, is_blocking=False)
                    queue_h2d.finish()
                    queue_d2h.finish()
                    dt1 = timeit.default_timer() - t0
                    if dt == 0 or dt1 < dt:
                        dt = dt1

                gbytes = 2 * size * 4 * nb * multiplicity / dt / 1024 ** 3

                for a in vad:
                    a.data.release()
                if parallel:
                    for b in vbd:
                        b.data.release()
                del va, vb, vad, vbd

                gc.collect()
            elif 'cuda' in self.language:
                pu = self.pu
                va = []
                vad = []
                mb0, junk = cu_drv.mem_get_info()
                if verbose:
                    print("Allocating memory, %d chunks of %8.2f Mbyte [GPU], with %d copies in host memory "
                          "[pinned=%s]..." % (nb, size * 4 / 2 ** 20, multiplicity, str(pinned)))
                # Only compute once chunk. Not constant value in case some procedure tries to be smart
                # chunk = np.random.uniform(0, 1, (size,)).astype(np.float32)
                chunk = np.linspace(0, np.pi, size, dtype=np.float32)
                for ii in range(nb):
                    va.append([])
                    for jj in range(multiplicity):
                        if pinned:
                            va[-1].append(cu_drv.pagelocked_empty((size,), np.float32))
                            va[-1][-1][:] = chunk
                        else:
                            va[-1].append(chunk.copy())
                    vad.append(cua.to_gpu(va[-1][-1]))
                    mb1, junk = cu_drv.mem_get_info()
                    if verbose:
                        m = vad[-1].nbytes / 2 ** 20 * (ii + 1)
                        print('Allocated: %8.0f Mbyte[GPU] %8.0f Mbyte[host], Used: +%5.0f Mbyte[GPU]' %
                              (m, m * multiplicity, (mb0 - mb1) / 2 ** 20))

                if parallel:
                    if verbose:
                        print("Allocating 2nd set of arrays for parallel in/out transfers")
                    vb = []
                    vbd = []
                    cu_stream_in = cu_drv.Stream()
                    cu_stream_out = cu_drv.Stream()
                    for ii in range(nb):
                        vb.append([])
                        for jj in range(multiplicity):
                            if pinned:
                                vb[-1].append(cu_drv.pagelocked_empty((size,), np.float32))
                                vb[-1][-1][:] = chunk
                            else:
                                vb[-1].append(chunk.copy())
                        vbd.append(cua.to_gpu(vb[-1][-1]))
                        mb1, junk = cu_drv.mem_get_info()
                        if verbose:
                            m = vbd[-1].nbytes / 2 ** 20 * (ii + 1 + nb)
                            print('Allocated: %8.0f Mbyte[GPU] %8.0f Mbyte[host], Used: +%5.0f Mbyte[GPU]' %
                                  (m, m * multiplicity, (mb0 - mb1) / 2 ** 20))
                else:
                    cu_stream_in = cu_drv.Stream()
                    cu_stream_out = cu_stream_in
                    vb = va
                    vbd = vad
                if verbose:
                    print("Beginning test transfers...")
                cu_stream_in.synchronize()
                cu_stream_out.synchronize()
                for i in range(nb_test):
                    t0 = timeit.default_timer()
                    for ii in range(nb):
                        for jj in range(multiplicity):
                            cu_drv.memcpy_htod_async(dest=vad[ii].gpudata, src=va[ii][jj], stream=cu_stream_in)
                            cu_drv.memcpy_dtoh_async(dest=vb[ii][jj], src=vbd[ii].gpudata, stream=cu_stream_out)
                    cu_stream_in.synchronize()
                    cu_stream_out.synchronize()
                    dt1 = timeit.default_timer() - t0
                    if dt == 0 or dt1 < dt:
                        dt = dt1
                gbytes = 2 * size * 4 * nb * multiplicity / dt / 1024 ** 3

                if verbose:
                    print("Free memory...")
                for a in vad:
                    a.gpudata.free()
                if parallel:
                    for b in vbd:
                        b.gpudata.free()
                del va, vb, vad, vbd
                gc.collect()
            else:
                # TODO: test CPU speed
                pass
        # except:
        #    gbytes = -1
        #    dt = 0
        if pinned:
            s = "pinned_" + s
        if parallel:
            s = "parallel_" + s
        s += "_x%d" % nb
        if multiplicity > 1:
            s += "_x%d" % multiplicity
        s = 'mem_h2d_%s' % s
        self.results['%s_Gbytes' % (s)] = gbytes
        self.results['%s_dt' % (s)] = dt
        gbyte_total = nb * size * 4 / 1024 ** 3
        if parallel:
            gbyte_total *= 2
        print('%30s: %8.2f Gbyte/s, dt =%6.4fs (total alloc.=%8.3f GByte[GPU] %8.3f GByte[host], chunk=%8.0f MByte)' %
              (s, gbytes, dt, gbyte_total, gbyte_total * multiplicity, size * 4 / 1024 ** 2))
        return dt

    def test_mem_copy(self, size):
        """
        Test the speed for copying on-device data between two allocated arrays..
        :param size: the size of the array, i.e. the number of float32 elements to transfer
        :return: The elapsed time for the calculation. The result is added to self.results
        """
        a = np.random.uniform(0, 1, size).astype(np.float32)
        s = "%dM" % int(round(a.nbytes / 1024 ** 2))
        gbytes, dt = 0, 0
        # try:
        if True:
            if 'opencl' in self.language:
                pu = self.pu

                ad = cla.to_device(pu.cl_queue, a)
                bd = cla.Array(pu.cl_queue, size, dtype=np.float32)

                pu.cl_queue.finish()
                dt = 0
                nbtry = 5  # Return best of N tests
                nb = 20  # Perform N successive tests to avoid python launch overhead
                for i in range(nbtry):
                    t0 = timeit.default_timer()
                    for j in range(nb):
                        cl.enqueue_copy(pu.cl_queue, dest=bd.data, src=ad.data)
                        cl.enqueue_copy(pu.cl_queue, dest=ad.data, src=bd.data)
                    pu.cl_queue.finish()
                    dt1 = timeit.default_timer() - t0
                    if dt == 0 or dt1 < dt:
                        dt = dt1
                # * 4: 2 copies, each d2d copy is 1 read + 1 write
                gbytes = 4 * a.nbytes / dt / 1024 ** 3 * nb
            elif 'cuda' in self.language:
                pu = self.pu
                ad = cua.to_gpu(a)
                bd = cua.GPUArray(shape=size, dtype=np.float32)
                pu.cu_ctx.synchronize()

                dt = 0
                nbtry = 5  # Return best of N tests
                nb = 20  # Perform N successive tests to avoid python launch overhead
                ev_begin = cu_drv.Event()
                ev_end = cu_drv.Event()
                for i in range(nbtry):
                    ev_begin.record()
                    for j in range(nb):
                        cu_drv.memcpy_dtod_async(dest=bd.gpudata, src=ad.gpudata, size=a.nbytes)
                        cu_drv.memcpy_dtod_async(dest=ad.gpudata, src=bd.gpudata, size=a.nbytes)
                    ev_end.record()
                    ev_end.synchronize()
                    dt1 = ev_end.time_since(ev_begin) / 1000
                    if dt == 0 or dt1 < dt:
                        dt = dt1
                # * 4: 2 copies, each d2d copy is 1 read + 1 write
                gbytes = 4 * a.nbytes / dt / 1024 ** 3 * nb
            else:
                # TODO: test CPU speed
                pass
        # except:
        #    gbytes = -1
        #    dt = 0
        s = 'mem_copy_%s' % s
        self.results['%s_Gbytes' % (s)] = gbytes
        self.results['%s_dt' % (s)] = dt
        print('%30s: %8.2f Gbyte/s, dt =%6.4fs' % (s, gbytes, dt))
        return dt

    def test_mem_copy_peer(self, size, parallel=True):
        """
        Test the speed for copying a data array from device to device (CUDA only). This will only be tested between
        devices with the same name and peer capability.
        :param size: the size of the array, i.e. the number of float32 elements to transfer
        :param parallel: if True, use parallel streams to perform d1->d2 and d2->d1 copies simultaneously.
        :return: The elapsed time for the calculation. The result is added to self.results
        """
        a = np.random.uniform(0, 1, size).astype(np.float32)
        s = "%dM" % int(round(a.nbytes / 1024 ** 2))
        gbytes, dt = 0, 0
        if True:
            if 'opencl' in self.language:
                gbytes = -1
                dt = 0
            elif 'cuda' in self.language:
                pu = self.pu

                # Now find all devices with the same name and peer access
                d0 = pu.cu_device
                vd = []
                for i in range(cu_drv.Device.count()):
                    d = cu_drv.Device(i)
                    # print(d0.name(), '<->', d.name(), '? ', d0.can_access_peer(d))
                    if d == d0 or d.name() != d0.name():
                        continue
                    if d0.can_access_peer(d):
                        vd.append(d)

                if len(vd) > 0:
                    # Allocate the array

                    if parallel:
                        cu_stream_in = cu_drv.Stream()
                        cu_stream_out = cu_drv.Stream()
                    else:
                        cu_stream_in = cu_drv.Stream()
                        cu_stream_out = cu_stream_in

                    ctx0 = d0.make_context()
                    ad = cua.to_gpu(a)
                    ctx0.synchronize()

                    dt = 0
                    for i in range(len(vd)):
                        d = vd[i]

                        ctx1 = d.make_context()
                        ctx1.push()
                        ad1 = cua.to_gpu(a)

                        ctx1.enable_peer_access(ctx0)
                        ctx0.push()
                        ctx0.enable_peer_access(ctx1)

                        ctx1.synchronize()
                        ctx0.synchronize()

                        for ii in range(5):
                            t0 = timeit.default_timer()
                            if parallel:
                                cu_drv.memcpy_peer_async(dest=ad1.ptr, src=ad.ptr, size=a.nbytes, dest_context=ctx1,
                                                         src_context=ctx0, stream=cu_stream_out)
                                cu_drv.memcpy_peer_async(dest=ad.ptr, src=ad1.ptr, size=a.nbytes, dest_context=ctx0,
                                                         src_context=ctx1, stream=cu_stream_in)
                                cu_stream_out.synchronize()
                                cu_stream_in.synchronize()
                            else:
                                cu_drv.memcpy_peer(dest=ad1.ptr, src=ad.ptr, size=a.nbytes, dest_context=ctx1,
                                                   src_context=ctx0)
                                cu_drv.memcpy_peer(dest=ad.ptr, src=ad1.ptr, size=a.nbytes, dest_context=ctx0,
                                                   src_context=ctx1)
                                ctx0.synchronize()
                                ctx1.synchronize()
                            dt1 = timeit.default_timer() - t0
                            # print(d0.name(), '<->', d.name(), '[%d]' % ii, ': dt=%6.4fs' % dt1)
                            if dt == 0 or dt1 < dt:
                                dt = dt1
                        ctx1.detach()
                    gbytes = 2 * a.nbytes / dt / 1024 ** 3
                    # cu_drv.Context.pop()  # Does not work, or not sufficient
                    ctx0.detach()
                else:
                    gbytes = -1
                    dt = 0

            else:
                # TODO: test CPU speed
                pass
        # except:
        #    gbytes = -1
        #    dt = 0
        if parallel:
            s = "parallel_" + s
        s = 'mem_copy_peer_%s' % s
        self.results['%s_Gbytes' % (s)] = gbytes
        self.results['%s_dt' % (s)] = dt
        print('%30s: %8.2f Gbyte/s, dt =%6.4fs' % (s, gbytes, dt))
        return dt

    def test_fft_host2device_swap(self, size, n_fft=1, n_stack=3, n_iter=3):
        """
        Test FFT calculations while constantly swapping datasets in pinned memory.
        :param size: the size N of the 2D FFT which will be computed (using 16xNxN arrays)
        :param n_fft: the number of FFT + iFFT to perform between swaps. Depending on the relative speed
                       of memory transfer and FFT, a minimum number of FFT must be done before saturating
                       the GPU.
        :param n_iter: number of iterations for the calculations
        :param n_stack: number of stacks of 16xNxN to be swapped (must be>=3)
        :return: the time to perform the calculations with swap
        """
        if 'cuda' not in self.language:
            return 0
        cu_stream_fft = cu_drv.Stream()
        cu_stream_in = cu_drv.Stream()
        cu_stream_out = cu_drv.Stream()

        # Create data with pinned memory, random data to avoid any smart optimisation
        vpsi = []
        for j in range(n_stack):
            vpsi.append(cu_drv.pagelocked_empty((16, size, size), np.complex64))
            vpsi[-1][:] = np.random.uniform(0, 1, (16, size, size))
        # Allocate 3 arrays in GPU
        cu_psi = cua.to_gpu(vpsi[0])
        cu_psi_in = cua.to_gpu(vpsi[1])
        cu_psi_out = cua.to_gpu(vpsi[2])

        # First test fft on array remaining in GPU
        self.pu.fft(cu_psi, cu_psi, ndim=2)
        t0 = timeit.default_timer()
        for i in range(n_iter * n_stack):
            for k in range(n_fft):
                self.pu.fft(cu_psi, cu_psi, ndim=2)
                self.pu.ifft(cu_psi, cu_psi, ndim=2)

        self.pu.finish()
        dt0 = timeit.default_timer() - t0
        # This measures the number of Gbyte/s for which the n_fft FFT are calculated
        # If n_fft=0 this should correspond to the total bandwidth
        gbytes0 = cu_psi.size * 8 * 2 * n_fft * n_iter * n_stack / dt0 / 1024 ** 3

        # Now perform FFT while transferring in // data to and from GPU with three queues
        t0 = timeit.default_timer()
        for i in range(n_iter):
            for j in range(n_stack):
                cu_drv.memcpy_htod_async(dest=cu_psi_in.gpudata, src=vpsi[(j + 1) % n_stack],
                                         stream=cu_stream_in)
                for k in range(n_fft):
                    self.pu.fft(cu_psi, cu_psi, ndim=2)
                    self.pu.ifft(cu_psi, cu_psi, ndim=2)
                cu_drv.memcpy_dtoh_async(src=cu_psi_out.gpudata, dest=vpsi[(j - 1) % n_stack],
                                         stream=cu_stream_out)
                # Swap stacks
                cu_psi_in, cu_psi, cu_psi_out = cu_psi_out, cu_psi_in, cu_psi
                # Make sure tasks are finished in each stream before beginning a new one.
                # Use events so that the wait is done asynchronously on the GPU
                ev_fft = cu_drv.Event(cu_drv.event_flags.DISABLE_TIMING)
                ev_in = cu_drv.Event(cu_drv.event_flags.DISABLE_TIMING)
                ev_out = cu_drv.Event(cu_drv.event_flags.DISABLE_TIMING)
                ev_fft.record(cu_stream_fft)
                ev_in.record(cu_stream_in)
                ev_out.record(cu_stream_out)
                cu_stream_fft.wait_for_event(ev_in)  # Data must be arrived before being processed
                cu_stream_in.wait_for_event(ev_out)  # Data out must be finished before replacing by in
                cu_stream_out.wait_for_event(ev_fft)  # Processing must be finished before transfer out
        self.pu.finish()
        dt1 = timeit.default_timer() - t0
        gbytes1 = cu_psi.size * 8 * 2 * n_iter * n_fft * n_stack / dt1 / 1024 ** 3
        print("Time for %3d FFT of size 16x%dx%d:\n"
              "             on-GPU:%6.3fs (%8.2f Gbyte/s)\n"
              "      with h2d swap:%6.3fs (%8.2f Gbyte/s) (%2d FT+iFT per swap)" %
              (n_iter * n_stack * n_fft, size, size, dt0, gbytes0, dt1, gbytes1, n_fft))
        self.results['fft_h2d_swap_%dfft%d_Gbytes' % (n_fft, size)] = gbytes1
        self.results['fft_h2d_swap_%dfft%d_dt' % (n_fft, size)] = dt1
        return dt1

    def test_scattering(self, size):
        """
        Test using pynx.scattering.speed.mrats
        :param size: the number of atoms = number of reflections
        :return: The elapsed time for the calculation. The result is added to self.results
        """
        try:
            gflops, dt = mrats(size, size, gpu_name=self.gpu_name, verbose=True, language=self.language,
                               cl_platform=self.cl_platform, timing=True)
            gflops *= 8e-3
        except:
            gflops, dt = -1, 0
        self.results['scattering_%d_Gflops' % (size)] = gflops
        self.results['scattering_%d_dt' % (size)] = dt
        return dt

    def test_fft_2d(self, size, stack_size=32, verbose=True, nb_cycle=1):
        """
        Test a stacked (16) 2D FFT
        :param size: the size N of the 2D FFT, which will be computed as using 16xNxN array
        :return: dt=the time for one FT+FT-1. The result is added to self.results
        """
        gflops = -1
        dt = 0
        if 'opencl' in self.language:
            gflops, dt, ax = cl_device_fft_speed(self.pu.cl_device, fft_shape=(stack_size, size, size), axes=(-1, -2),
                                                 verbose=False, timing=True, shuffle_axes=True, nb_cycle=nb_cycle)
        elif 'cuda' in self.language:
            gflops, dt = cuda_device_fft_speed(self.pu.cu_device, fft_shape=(stack_size, size, size), batch=True,
                                               verbose=False, timing=True, nb_cycle=nb_cycle)
        else:
            # TODO: test CPU speed
            pass
        self.results['fft_2Dx%d_%d_Gflops' % (stack_size, size)] = gflops
        self.results['fft_2Dx%d_%d_dt' % (stack_size, size)] = dt
        if verbose:
            print('fft_2Dx%d_%d: %8.2f Gflop/s, dt =%6.4fs' % (stack_size, size, gflops, dt))
        return dt

    def test_fft_3d(self, size, verbose=True, nb_cycle=1):
        """
        Test a 3D FFT
        :param size: the size N of the 3D FFT, which will be computed using a NxNxN array
        :return: dt=the time for one FT+FT-1. The result is added to self.results
        """
        gflops = -1
        dt = 0
        try:
            if 'opencl' in self.language:
                gflops, dt, ax = cl_device_fft_speed(self.pu.cl_device, fft_shape=(size, size, size), axes=(-1, -2, -3),
                                                     verbose=False, timing=True, shuffle_axes=True, nb_cycle=nb_cycle)
            elif 'cuda' in self.language:
                gflops, dt = cuda_device_fft_speed(self.pu.cu_device, fft_shape=(size, size, size), batch=False,
                                                   verbose=False, timing=True, nb_cycle=nb_cycle)
            else:
                # TODO: test CPU speed
                pass
        except:
            pass
        self.results['fft_3D_%d_Gflops' % size] = gflops
        self.results['fft_3D_%d_dt' % size] = dt
        if verbose:
            print('fft_3D_%d: %8.2f Gflop/s, dt =%6.4fs' % (size, gflops, dt))
        return dt

    def test_ptycho(self, nb_frame, frame_size, nb_cycle=20, algo="AP"):
        """
        Run 2D ptychography speed test

        :param nb_frame:
        :param frame_size:
        :param nb_cycle:
        :param nb_obj:
        :param nb_probe:
        :param algo:
        :return: the execution time (not counting initialisation)
        """
        # TODO: move this elswhere, to avoid imports inside the function
        from pynx.ptycho import simulation, shape
        if 'opencl' in self.language:
            from pynx.ptycho import cl_operator as ops
            ops.default_processing_unit.set_device(d=self.pu.cl_device, test_fft=False, verbose=False)
        elif 'cuda' in self.language:
            from pynx.ptycho import cu_operator as ops
            ops.default_processing_unit.set_device(d=self.pu.cu_device, test_fft=False, verbose=False)
        elif 'cpu' in self.language:
            # TODO: enable CPU testing
            from pynx.ptycho import cpu_operator as ops
        else:
            # TODO: this should not be necessary, the device should have been selected by prepare_processing_unit()
            from pynx.ptycho import operator as ops
            ops.default_processing_unit.select_gpu(gpu_name=self.gpu_name, verbose=False, ranking='order')
        from pynx.ptycho.ptycho import Ptycho, PtychoData

        n = frame_size
        pixel_size_detector = 55e-6
        wavelength = 1.5e-10
        detector_distance = 1
        obj_info = {'type': 'phase_ampl', 'phase_stretch': np.pi / 2, 'alpha_win': .2}
        probe_info = {'type': 'gauss', 'sigma_pix': (40, 40), 'shape': (n, n)}

        # 50 scan positions correspond to 4 turns, 78 to 5 turns, 113 to 6 turns
        scan_info = {'type': 'spiral', 'scan_step_pix': 30, 'n_scans': nb_frame}
        data_info = {'num_phot_max': 1e9, 'bg': 0, 'wavelength': wavelength, 'detector_distance': detector_distance,
                     'detector_pixel_size': pixel_size_detector, 'noise': 'poisson'}

        # Initialisation of the simulation with specified parameters
        s = simulation.Simulation(obj_info=obj_info, probe_info=probe_info, scan_info=scan_info, data_info=data_info,
                                  verbose=False)
        s.make_data()

        # Positions from simulation are given in pixels
        posx, posy = s.scan.values

        ampl = s.amplitude.values  # square root of the measured diffraction pattern intensity
        pixel_size_object = wavelength * detector_distance / pixel_size_detector / n
        data = PtychoData(iobs=ampl ** 2, positions=(posx * pixel_size_object, posy * pixel_size_object),
                          detector_distance=1, mask=None, pixel_size_detector=pixel_size_detector,
                          wavelength=wavelength)

        p = Ptycho(probe=s.probe.values, obj=s.obj.values, data=data, background=None)

        if algo.lower() == "dm":
            op = ops.DM()
        elif algo.lower() == "ml":
            op = ops.ML()
        else:
            algo = "AP"
            op = ops.AP()

        p = op ** 5 * p
        ops.default_processing_unit.finish()

        t0 = timeit.default_timer()
        p = op ** nb_cycle * p
        ops.default_processing_unit.finish()
        dt = timeit.default_timer() - t0

        # Free memory
        p = ops.FreePU() * p
        ops.default_processing_unit.free_fft_plans()
        gc.collect()

        # 5 * n * n is a (very) conservative estimate of the number of elementwise operations.
        gflops = nb_cycle * nb_frame * (2 * 5 * n * n * np.log2(n * n) + n * n * 5) / 1e9 / dt

        self.results['ptycho_%dx%dx%d_%s_Gflops' % (nb_frame, n, n, algo)] = gflops
        self.results['ptycho_%dx%dx%d_%s_dt' % (nb_frame, n, n, algo)] = dt / nb_cycle
        print('ptycho_%dx%dx%d_%s: %8.2f Gflop/s, dt =%6.4fs/cycle' % (nb_frame, n, n, algo, gflops, dt / nb_cycle))

        return dt

    def test_ptycho_runner_simul(self, nb_frame, frame_size, algo="ML**100,DM**200,probe=1,nbprobe=1"):
        """
        Run 2D ptychography speed test using pynx-simulationpty.py runner

        :param nb_frame: number of frames
        :param frame_size: frame size in pixels
        :param algo: algorithm chains to use
        :param mpi_size: number of parallel MPI process to use [default=1, no MPI]
        :return: the algorithm execution time (not counting initialisation)
        """

        my_env = os.environ.copy()
        my_env["PYNX_PU"] = "%s.%s" % (self.language, self.gpu_name)
        args = ['pynx-simulationpty.py', 'frame_nb=%d' % nb_frame, 'frame_size=%d' % frame_size,
                'algorithm=%s' % algo, 'verbose=50', 'saveprefix=none']
        p = subprocess.Popen(args, stderr=subprocess.PIPE, stdout=subprocess.PIPE, env=my_env, shell=False)

        stdout, stderr = p.communicate(timeout=200)
        res = p.returncode
        llout = str(stdout).split('\\n')
        dt = 0
        for l in llout:
            if 'Total elapsed time for algorithms' in l:
                dt += float(l.split()[-1][:-1])

        n = frame_size
        s = 'ptycho_runner_simul_%dx%dx%d_%s_dt' % (nb_frame, n, n, algo)
        self.results[s] = dt
        print('%s: dt =%6.4fs' % (s[:-3], dt))

    def test_cdi3d(self, size=128, algo="ER**20"):
        """
        Run 3D CDI speed test
        :param size: the size of the object along one dimension
        :param algo: the algorithm string to use (ER**20 by default, can also be HIO or RAAR)
        :return: the execution time (not counting initialisation)
        """
        from pynx.cdi.cdi import CDI
        if 'opencl' in self.language:
            from pynx.cdi import cl_operator as ops
            ops.default_processing_unit.use_opencl(cl_device=self.pu.cl_device, test_fft=False, verbose=False)
        elif 'cuda' in self.language:
            from pynx.cdi import cu_operator as ops
            ops.default_processing_unit.use_cuda(cu_device=self.pu.cu_device, test_fft=False, verbose=False)
        elif 'cpu' in self.language:
            from pynx.cdi import cpu_operator as ops
        else:
            from pynx.cdi import operator as ops
            ops.default_processing_unit.select_gpu(gpu_name=self.gpu_name, verbose=False, ranking='order')

        n = size

        # Object coordinates
        tmp = np.arange(-n // 2, n // 2, dtype=np.float32)
        z, y, x = np.meshgrid(tmp, tmp, tmp, indexing='ij')

        # Parallelepiped object
        obj0 = (abs(x) < 12) * (abs(y) < 10) * (abs(z) < 16)
        # Start from a slightly loose support
        support = (abs(x) < 20) * (abs(y) < 20) * (abs(z) < 25)

        cdi = CDI(np.zeros_like(obj0), obj=obj0, support=np.fft.fftshift(support), mask=None, wavelength=1e-10,
                  pixel_size_detector=55e-6)

        cdi = ops.Calc2Obs() * cdi

        # Init object & algorithms
        cdi = ops.ER() ** 5 * cdi
        ops.default_processing_unit.finish()

        # Prepare real algo
        er = ops.ER()
        raar = ops.RAAR()
        hio = ops.HIO()

        t0 = timeit.default_timer()
        cdi = eval(algo.lower()) * cdi
        ops.default_processing_unit.finish()
        dt = timeit.default_timer() - t0

        # Free memory
        cdi = ops.FreePU() * cdi
        ops.default_processing_unit.free_fft_plans()
        gc.collect()

        nb_cycle = cdi.cycle - 5

        # 5 * n * n is a (very) conservative estimate of the number of elementwise operations.
        gflops = nb_cycle * (2 * 5 * n ** 3 * np.log2(n ** 3) + n * n * n * 5) / 1e9 / dt

        self.results['cdi3d_%dx%dx%d_%s_Gflops' % (n, n, n, algo)] = gflops
        self.results['cdi3d_%dx%dx%d_%s_dt' % (n, n, n, algo)] = dt
        print('cdi3d_%dx%dx%d_%s: %8.2f Gflop/s, dt =%6.4fs/cycle' % (n, n, n, algo, gflops, dt / nb_cycle))

        return dt

    def run_mem(self, verbose=True):
        """
        Run memory transfer/copy tests
        :param verbose: if True, verbose output
        :return: nothing
        """
        self.test_mem_host2device(2 ** 24, pinned=False, parallel=False)
        self.test_mem_host2device(2 ** 24, pinned=True, parallel=False)
        self.test_mem_host2device(2 ** 24, pinned=False, parallel=True)
        self.test_mem_host2device(2 ** 24, pinned=True, parallel=True)
        self.test_mem_copy(2 ** 24)
        if 'cuda' in self.language:
            self.test_mem_copy_peer(2 ** 24, parallel=False)
            self.test_mem_copy_peer(2 ** 24)

    def run_scattering(self, verbose=True):
        """
        Run memory scattering tests
        :param verbose: if True, verbose output
        :return: nothing
        """
        dt = self.test_scattering(int(2 ** 10))
        dt = self.test_scattering(int(2 ** 14))
        if dt < 1:
            dt = self.test_scattering(int(2 ** 18))
        if dt < 1:
            dt = self.test_scattering(int(2 ** 20))

    def run_fft2D(self, verbose=True):
        """
        Run 2D FFT tests
        :param verbose: if True, verbose output
        :return: nothing
        """
        dt = self.test_fft_2d(size=256)
        dt = self.test_fft_2d(size=1024)
        if dt < 1:
            dt = self.test_fft_2d(size=2048)
        if dt < 1:
            dt = self.test_fft_2d(size=4096)

    def run_fft3D(self, verbose=True):
        """
        Run 3D FFT tests
        :param verbose: if True, verbose output
        :return: nothing
        """
        dt = self.test_fft_3d(size=128)
        dt = self.test_fft_3d(size=256)
        if dt < 1:
            dt = self.test_fft_3d(size=512)

    def benchmark(self):
        """
        Run a series of test for mem, FFT, CDI, ptycho and return a 1-line result
        :return:
        """
        self.test_mem_host2device(2 ** 24, pinned=True, parallel=True, )
        self.test_mem_copy(2 ** 24)
        self.test_fft_3d(size=256)
        self.test_fft_3d(size=512)
        self.test_fft_3d(size=1024)
        self.test_fft_2d(size=256, stack_size=32)
        self.test_fft_2d(size=512, stack_size=32)
        self.test_fft_2d(size=1024, stack_size=32)
        self.test_cdi3d(size=512, algo="ER**100*HIO**800")
        self.test_ptycho_runner_simul(400, 256, algo="ML**100,DM**200,probe=1")
        self.test_ptycho_runner_simul(400, 512, algo="ML**100,DM**200,probe=1,nbprobe=3")

        res = ("%s" + "\t%6.2f" * 11) % \
              (self.language,
               self.results['mem_h2d_parallel_pinned_64M_x1_Gbytes'],
               self.results['mem_copy_64M_Gbytes'],
               self.results['fft_3D_256_dt'] * 1000,
               self.results['fft_3D_512_dt'] * 1000,
               self.results['fft_3D_1024_dt'] * 1000,
               self.results['fft_2Dx32_256_dt'] * 1000,
               self.results['fft_2Dx32_512_dt'] * 1000,
               self.results['fft_2Dx32_1024_dt'] * 1000,
               self.results['cdi3d_512x512x512_ER**100*HIO**800_dt'],
               self.results['ptycho_runner_simul_400x256x256_ML**100,DM**200,probe=1_dt'],
               self.results['ptycho_runner_simul_400x512x512_ML**100,DM**200,probe=1,nbprobe=3_dt'])
        print(res)
        return res

    def run(self, tests='all', export_db=None, verbose=True):
        """
        Run selected tests
        :param tests: a comma-separated string listing all desired tests.
                      Valid values are: all,mem,scattering,fft,fft2D,fft3D
        :param export_db: the name of the database to save the speed tests to. The file will be created if necessary.
        :param verbose: if True, verbose output
        :return:
        """
        for s in tests.split(','):
            if 'mem' in s.lower() or s.lower() == 'all':
                self.run_mem(verbose=verbose)
            if 'scatt' in s.lower() or s.lower() == 'all':
                self.run_scattering(verbose=verbose)
            if 'fft2d' in s.lower() or s.lower() == 'fft' or s.lower() == 'all':
                self.run_fft2D(verbose=verbose)
            if 'fft3d' in s.lower() or s.lower() == 'fft' or s.lower() == 'all':
                self.run_fft3D(verbose=verbose)
            if 'ptycho' in s.lower() or s.lower() == 'all':
                self.test_ptycho(nb_frame=64, frame_size=256, nb_cycle=20, algo='AP')
            if 'cdi' in s.lower() or s.lower() == 'all':
                self.test_cdi3d(size=128, algo='ER**20')
            if 'swap' in s.lower() or s.lower() == 'all':
                for n_fft in range(1, 5):
                    self.test_fft_host2device_swap(size=1024, n_fft=n_fft, n_iter=2, n_stack=5)
            if 'benchmark' in s.lower():
                self.benchmark()
        if export_db:
            self.export_db()

    def export_db(self):
        self.prepare_db()
        cols = ""
        vals = ""
        for k, v in self.results.items():
            if len(cols) > 0:
                cols += ','
                vals += ','
            cols += k
            if k in ['hostname', 'language', 'GPU']:
                vals += "'%s'" % (v)
            else:
                vals += str(v)
        com = "INSERT INTO pynx_speed_test (%s) VALUES (%s);" % (cols, vals)
        com = com.replace("*", "_")
        com = com.replace("=", "_")
        self.db_curs.execute(com)
        self.db_conn.commit()
        self.db_curs.close()

    def import_db(self, unique=True, gpu_name=None, language=None, cl_platform=None):
        """
        Extract the results stored in the database.

        :param unique: if True, only the latest result will be plotted for a given combination
                       of hostname + GPU + language.
        :param gpu_name: name or partial name for the GPU which should be listed
        :param language: 'cuda' or 'opencl' or 'CPU' to filter results
        :param cl_platform: the opencl platform to plot
        :return: nothing
        """
        self.db_results = {}
        self.prepare_db()
        self.db_curs.execute('select * from pynx_speed_test order by epoch')
        rr = self.db_curs.fetchall()
        tt = [x[0] for x in self.db_curs.description]
        for r in rr:
            d = {}
            for k, v in zip(tt, r):
                d[k] = v
            name = '%s[%s]\n[%s]' % (d['GPU'], d['language'], d['hostname'])
            if not unique:
                name += '\n[%s]' % (time.strftime('%Y/%m/%d %H:%M:%S', time.gmtime(d['epoch'])))
            if gpu_name is not None:
                if gpu_name.lower() not in d['GPU'].lower():
                    continue
            if language is not None:
                if language.lower() not in d['language'].lower():
                    continue
            if cl_platform is not None:
                if cl_platform.lower() not in d['cl_platform'].lower():
                    continue
            self.db_results[name] = d

    def plot(self, unique=True, gpu_name=None, language=None, cl_platform=None):
        """
        Plot all the results stored in the database.

        :param unique: if True, only the latest result will be plotted for a given combination
                       of hostname + GPU + language.
        :param gpu_name: name or partial name for the GPU which should be listed
        :param language: 'cuda' or 'opencl' or 'CPU' to filter results
        :param cl_platform: the opencl platform to plot
        :return: nothing
        """
        self.import_db(unique=unique, gpu_name=gpu_name, language=language, cl_platform=cl_platform)
        tt = list(list(self.db_results.values())[0].keys())
        for t in tt:
            if 'Gflops' in t:
                t2 = t.split('Gflops')[0] + 'dt'
                name, gflops, dt = [], [], []
                for k, v in self.db_results.items():
                    name.append(k)
                    if v[t] is None:
                        gflops.append(0)
                    else:
                        gflops.append(v[t])
                    if v[t] is None:
                        dt.append(0)
                    else:
                        dt.append(v[t2])
                print('Plotting: %s' % t)
                gflops = np.array(gflops)
                dt = np.array(dt)
                x = range(len(gflops))
                plt.figure(figsize=(12, 6))
                plt.bar(x, np.array(gflops))
                plt.xticks(x, name, rotation=90, horizontalalignment='center', verticalalignment='bottom')
                plt.ylabel(t)
                plt.ylim(10, gflops.max() * 1.05)
                ax1 = plt.gca()
                ax2 = ax1.twinx()
                ax2.set_yscale('log')
                imax = gflops.argmax()
                dtmin = dt[imax] / 1.05
                dtmax = dt[imax] * gflops[imax] / 10
                ax2.set_ylim(dtmax, dtmin)
                ax2.set_ylabel('dt(s)')
                plt.show()


def plot_fft_speed(gpu_name, nmin, nmax, ndim=2, stack_size=64, nb_cycle=4):
    """
    Plot the FFT vs the size of the ZD FFT
    :param gpu_name:
    :param language:
    :param nmin:
    :param nmax:
    :return:
    """
    import matplotlib.pyplot as plt
    langs = []
    if has_cuda:
        langs += ['CUDA']
    if has_opencl:
        langs += ['OpenCL']
    res = {}
    for lang in langs:
        if 'cl' in lang.lower():
            print("Testing OpenCL: testing axes permutations to get optimal speed (3-10x slower)")
        vn = []
        vdt = []
        vgbs = []
        s = SpeedTest(gpu_name=gpu_name, language=lang)
        max_prime = 7
        # if 'cl' in lang.lower():
        #     max_prime = 13
        for i in range(nmin, nmax):
            if test_smaller_primes(i, max_prime, required_dividers=None):
                if ndim == 2:
                    dt = s.test_fft_2d(size=i, stack_size=stack_size, verbose=False, nb_cycle=nb_cycle)
                    # 2 pairs of 1D FFT = 8 i/o (4 reads and 4 writes)
                    gbs = stack_size * i ** 2 * 8 * 8 / 1000 ** 3 / dt
                    print("%s %dD FFT 64x %3d**%d: dt=%8.5fs, %8.3fGB/s" % (lang, ndim, i, ndim, dt, gbs))
                else:
                    dt = s.test_fft_3d(size=i, verbose=False, nb_cycle=nb_cycle)
                    # 3 pairs of 1D FFT = 6 reads and 6 writes
                    gbs = i ** 3 * 8 * 12 / 1000 ** 3 / dt
                    print("%s %dD FFT %3d**%d: dt=%8.5fs, %8.3fGB/s" % (lang, ndim, i, ndim, dt, gbs))
                vn.append(i)
                vdt.append(dt)
                vgbs.append(gbs)
        res[lang] = np.array(vn, dtype=np.int32), np.array(vdt), np.array(vgbs)
    plt.figure(figsize=(12, 6))
    for lang in langs:
        vn, vdt, vgbs = res[lang]
        plt.plot(vn, vgbs, 'g.' if 'cl' in lang.lower() else 'b.', label=lang)
        idx = np.where(abs(np.log2(vn) % 1) < 1e-4)[0]
        plt.plot(vn[idx], vgbs[idx], 'g.' if 'cl' in lang.lower() else 'b.', markersize=12)
    plt.ylim(0)
    plt.ylabel("GB/s")
    plt.xlabel("FFT size")
    plt.legend()
    plt.title("%dD FFT speed" % ndim)


def main():
    gpu_name = None
    language = None
    cl_platform = None
    do_plot = False
    tests = 'all'
    for a in sys.argv[1:]:
        karg = a.split('=')
        if len(karg) == 2:
            if karg[0] == 'gpu':
                gpu_name = karg[1]
            elif 'lang' in karg[0]:
                language = karg[1]
            elif karg[0] == 'cl_platform':
                cl_platform = karg[1]
            elif karg[0] == 'test':
                tests = karg[1]
        elif a.lower() == 'plot':
            do_plot = True
    s = SpeedTest(gpu_name, language=language, cl_platform=cl_platform)
    if gpu_name is not None and language is not None:
        s.run(tests=tests, export_db='pynx_speed.db')
    else:
        print('No GPU and/or language name given, not running tests')
    if do_plot:
        s.plot(unique=True)


if __name__ == '__main__':
    main()
