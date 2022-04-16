"""
Created on Tue Nov 3 10:30:25 2020

@author: Hermes Beamline, la creme de la creme !
"""
# %% ---------------------------------
#    Imports
# -----------------------------------

import sys, os
import numpy as np
import numexpr as ne
import h5py
from PIL import Image
import time
import timeit
import ast
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from ...utils.matplotlib import pyplot as plt

from .runner import PtychoRunner, PtychoRunnerScan, PtychoRunnerException, params_generic, _pynx_version, MPI
from ...utils import phase, plot_utils
from ...ptycho import analysis

# %%-----------------------------------------------------------------------

# The beamline specific helptext
# TODO
helptext_beamline = """
Script to perform a ptychography analysis on data recorded on Hermes@Soleil (*.nxs format)

Use:     pynx-hermespty.py arg1=value1 arg2=value2 etc.
        Or you might want to launch it from a batch file

'folder' and 'distance' arguments are mandatory, all others have default value listed beneeth

Arguments that can be passed are the following:
(Note that all the arguments that are specific to PyNX can also be given in the same way, refer to it below) 


help
    Flag
    Call for help
--------------------------------------------------

folder=your/path/to/the/data
    /!\ /!\ MANDATORY ARGUMENT /!\ /!\-
    The path where the data are located (data.nxs, dark.nxs, descr.hdf5 must all be there !)
--------------------------------------------------

distance=XX.XXe-3
    /!\ /!\ MANDATORY ARGUMENT /!\ /!\-
    The distance (in METER, use SI, be serious!) between the detector and the sample.
    This is for now calculated thanks to the annulus size with a calibration
    Calibration is there : /home/experiences/hermes/com-hermes/DATA/Sample2detector_Calculator.ods
--------------------------------------------------

scan=XX,XX,etc.
    The scans you want to reconstruct. 
    If not given, will reconstruct everything within the folder
--------------------------------------------------

threshold=XX
    threshold, for the high pass filter
    It correspond to the thermal noise of the camera, after dark subtraction
--------------------------------------------------

bin=X
    The binning value (try to avoid it)
--------------------------------------------------

camcenter=x0,y0
    The center of the camera, if the auto finding of PyNX do not find the
    proper center of diffraction
    You should respect the convention camcenter=x0,y0 
    with x0 and y0 INTEGERS: the coordinates x and y of the center in pixel coordinates
    Do not call this parameter to let PyNX find the center of diffraction
--------------------------------------------------

adaptq=the way to adapt q
    Only used for energy stack (so if type=stack)
    Defines the way to adapt the data to have a constant q space
    in the reconstructions of the whole stack
    Takes only three values:
        'soft': adapt the q space changing the 'maxsize' PyNX parameter
                /!\ do it if you didn't make detector distance to move during the stack
        'hard': #### NOT IMPLEMENTED YET #####
                adapt the q space changing the 'detector distance' parameter acccording to the control program way to move it
                /!\ do it if you actually did make detector distance to move during the stack
        'none': do not adaptq
                /!\ do it only if you don't wwant to adapt q space
    The default is 'soft': it changes automatically the 'maxsize' parameter
    taking into account the starting energy and the detector distance
    to get the same q range and thus the same nbr of pixel in all the
    reconstruction of the stack
--------------------------------------------------

probediam=XX.Xe-9
    Diameter of the probe (it is 1.22 x FZP outerZone width) 
    /!\  don't take into account the defocus here /!\-
--------------------------------------------------

onlynrj=XX,XX,XX
    nrj numbers you want to reconstruct within a stack
    If you want the whole stack, don't call this parameter
--------------------------------------------------

dark_as_optim
    Flag
    Call this flag to use the dark as background for reconstruction instead of
    subtracting it to each diffraction patter before the reconstruction 
    /!\/!\     IF YOU USE THIS, YOU SHOULD THEN OPTIMIZE THE BACKGROUND    /!\/!\
    /!\/!\    SOMEWHERE IN YOUR ALGORITHM (background=1)                     /!\/!\-
--------------------------------------------------

ruchepath=/nfs/ruche-hermes/hermes-soleil/com-hermes/PTYCHO/some/path
    Path where only the Amplitude.tif will be saved
    Usefull to have directly access to it for XMCD on another computer
--------------------------------------------------

savefolder=path/to/save/folder
    #### NOT IMPLEMENTED YET ####
    Where to save the reconstruction files
    Option: savefolder=same  -->   will save the recontruction in the folder where the data are
    Default value is 'same'
--------------------------------------------------

savetiff=option
    Option to save the reconstruction as tiff files
    options are: 
        'crop': save the image of only the scanned area //// NOT CODED YET ////
        'full': save the whole reconstruction (with part aside the scanned area, INCREADIBLE!)
        'No': don't save the tiff file 
    Default Value is 'full'
--------------------------------------------------

savellk
    Flag
    If you want to save the llk values in a separate text file
    will be saved as everyllk.txt
--------------------------------------------------

 ---->    Default parameters are the following:
"""

# The beamline specific default parameters
params_beamline = {}

params_beamline["scantype"] = None
params_beamline['detectordistance'] = None
params_beamline['threshold'] = 30
params_beamline["adaptq"] = 'soft'
params_beamline["real_coordinates"] = False
params_beamline["dark_as_optim"] = False

params_beamline['algorithm'] = "AP**150,nbprobe=1,probe=1"
params_beamline['defocus'] = 0
params_beamline['maxsize'] = 1000
params_beamline['camcenter'] = None
params_beamline['pixelsize'] = 11e-6
params_beamline['detector_orientation'] = "1,1,1"
params_beamline['probe'] = "disc,61e-9"
params_beamline['onlynrj'] = None

params_beamline['verbose'] = 50
params_beamline['obj_inertia'] = 0.1
params_beamline['probe_inertia'] = 0.01
params_beamline['obj_smooth'] = 0.0
params_beamline['probe_smooth'] = 0.0

params_beamline["savetiff"] = "full"
params_beamline['savellk'] = False
params_beamline['ruchepath'] = None

sdefault = "\n"
for key, value in params_beamline.items():
    sdefault += str(key) + "=" + str(value) + "\n"

helptext_beamline = helptext_beamline + "\n" + sdefault + "\n#################################\n\n"

params_beamline['instrument'] = "Hermes@Soleil"
params_beamline['liveplot'] = True
params_beamline['saveplot'] = True
params_beamline[
    'logfile'] = "/home/experiences/hermes/com-hermes/DATA/reconstruction_log/log_reconstruction_%.2d_%d.txt" % (
    time.gmtime().tm_mon, time.gmtime().tm_year)


# %% ----------------------------------
#    Generic Functions usefull
#     for data reconstruction
# --------------------------------------

def wavelength(energy):
    return 1239.8 * 1e-9 / energy


def calcROI(startROI, L, startEnergy, thisEnergy):
    """
    Function to calculate the new ROI 
    in order to adapt Q space the software way  
    """
    camPixelSize = 11.6 * 1e-6  # camera pixel size in µm
    roi1 = camPixelSize * startROI / 2
    theta1 = np.arctan(roi1 / L) * 180 / np.pi
    ll1 = wavelength(startEnergy)
    ll2 = wavelength(thisEnergy)
    return round(L * np.tan(np.arcsin((ll2 / ll1) * np.sin(theta1 * np.pi / 180))) * 1e+6 / 11.6 * 2)


def calcDistance(startDist, startEnergy, thisEnergy):
    """
    Function to calculate the new Detector Distance
    the same way it moves with the control of hermes ptycho setup
    if the detector distance is moved the hardware way (IRL)
    """
    # TODO
    # Put here the calculation
    return startDist


# import simpleaudio as sa
# def playnote(freq):
#    fs = 44100  # 44100 samples per second
#    seconds = 2  # Note duration of 3 seconds

#    # Generate array with seconds*sample_rate steps, ranging between 0 and seconds
#    t = np.linspace(0, seconds, seconds * fs, False)

#    # Generate a 440 Hz sine wave
#    note = np.sin(freq * t * 2 * np.pi)

#    # Ensure that highest value is in 16-bit range
#    audio = note * (2**15 - 1) / np.max(np.abs(note))
#    # Convert to 16-bit data
#    audio = audio.astype(np.int16)

#    # Start playback
#    play_obj = sa.play_buffer(audio, 1, 2, fs)

#    # Wait for playback to finish before exiting
#    play_obj.wait_done()

def create_params(intial_params, beamline_params):
    """
    Create the complete params dictionay to give to pynx 
    based on the generic pynx parmeter parameter (initial_params)
    and on beamline specific parameters dictionary for Hermes (beamline_params)
    and update also the 'scan' parameter 
    
    """
    params = intial_params.copy()
    for k, v in beamline_params.items():
        params[k] = v
    return params


def try_get_scan_param(scan_descr_hd5, parampath):
    """
    Function to avoid error on getting parameter from the hdf5 file from PIXELATOR
    If the parameters exists, returns the parameter value
    Otherwise, return a string indicating that the parameter hasn't been found
    
    params: scan_descr_hd5 --> hdf5 File from the h5py library loaded from the hdf5 PIXELATOR file
            parampath --> full path to the parameter inside the hdf5 file
                              (e.g. '/entry1/collection/energy/value')
    return: paramvalue --> Value of the concerned parameter if founded,
                               otherwise a string indicating unfound parameter
    """

    try:
        paramvalue = scan_descr_hd5.get(parampath)[0]
    # If the parameter is not found, the [0] which is suppose to extract
    # the parameter will raise a TypeError : 'NoneValue'
    except TypeError:
        paramvalue = 'Parameter : ' + parampath + ' not found in scan descr hdf5 file'

    return paramvalue


# %%-----------------------------------
#    Python Class for data reconstruction
# -------------------------------------
#

# Hermes Ptycho Runner Scan class

class PtychoRunnerScanHermes(PtychoRunnerScan):
    def __init__(self, params, scan, mpi_comm=None, timings=None):
        super(PtychoRunnerScanHermes, self).__init__(params, scan, mpi_comm, timings)

    def load_scan(self):
        """
        Supersed of load_scan function from runner
        Preprocess the data and update x, y positions of the camera images
        as well as the number of images imgn
        
        Also update the energy variable here, as it is stored in the same hdf file as the coordinates

        When the type of scan is energy stack:
        ---> self.scan refer to the nrj number (from 01 to NN, NN being the total number of energies)
        When the type of scan is separate files:
        ---> self.scan refer to the scan number XX, given by the Image_YYYYMMDD_XX.hdf5 file
        """

        if self.params["scantype"] == "stack":
            print("###############\nProcessing nrj number %d \n###############" % self.scan)
            hd5_scaninfo = h5py.File(os.path.join(self.params["folder"], self.params["filedescr"]), 'r')

        else:
            print("###############\nProcessing image number %d \n###############" % self.scan)
            hd5_scaninfo = h5py.File(os.path.join(self.params["folder"], self.params["filedescr"] % self.scan), 'r')

        # Building coodinates depending if sample_x and sample_y contains the full coordinates or the square points
        if self.params["real_coordinates"]:
            self.x = hd5_scaninfo.get("/entry1/camera/sample_x")[()].astype(np.float32) * 1e-6
            self.y = hd5_scaninfo.get("/entry1/camera/sample_y")[()].astype(np.float32) * 1e-6
        else:
            sample_x = hd5_scaninfo.get("/entry1/camera/sample_x")[()].astype(np.float32) * 1e-6
            sample_y = hd5_scaninfo.get("/entry1/camera/sample_y")[()].astype(np.float32) * 1e-6
            # Calculate the number of scan points in each direction, and update the nbr of images
            self.npoints_x = len(sample_x)
            self.npoints_y = len(sample_y)
            # Initializing the coordinates
            self.x = np.empty(self.npoints_x * self.npoints_y, dtype=np.float32)
            self.y = np.empty(self.npoints_x * self.npoints_y, dtype=np.float32)

            # Here two options for the scan direction : comment the bad one
            # Option 1 : The scan is in the type X stay constant, scan Y, step X, rescan Y, etc.
            #        for i in range(self.npoints_x):
            #            for j in range(self.npoints_y):
            #                self.coord_x[j+i*self.npoints_y] = sample_x[i]
            #                self.coord_y[j+i*self.npoints_y] = sample_y[j]

            # Option 2 : The scan is in the type Y stays constant, scan X, step Y, rescan X, etc.
            for j in range(self.npoints_y):
                for i in range(self.npoints_x):
                    self.x[i + j * self.npoints_x] = sample_x[i]
                    self.y[i + j * self.npoints_x] = sample_y[j]

        # Imgn variable
        imgn = np.arange(len(self.x), dtype=np.int)

        if self.params['moduloframe'] is not None:
            n1, n2 = self.params['moduloframe']
            idx = np.where(imgn % n1 == n2)[0]
            imgn = imgn.take(idx)
            self.x = self.x.take(idx)
            self.y = self.y.take(idx)

        if self.params['maxframe'] is not None:
            N = self.params['maxframe']
            if len(imgn) > N:
                print("MAXFRAME: only using first %d frames" % (N))
                imgn = imgn[:N]
                self.x = self.x[:N]
                self.y = self.y[:N]
        self.imgn = imgn

    def load_data(self):
        """
        Supersed of load_data function from runner
        Update the raw_data variable with all the images

        When the type of scan is energy stack: 
        ---> self.scan refer to the nrj number (from 01 to NN, NN being the total number of energies)
        When the type of scan is separate files:
        ---> self.scan refer to the scan number XX, given by the Image_YYYYMMDD_XX.hdf5 file
        """

        # nxs data file
        hd5_sample = h5py.File(os.path.join(self.params["folder"], self.params["filesample"] % self.scan), 'r')

        if self.params["scantype"] == "stack":
            hd5_scaninfo = h5py.File(os.path.join(self.params["folder"], self.params["filedescr"]), 'r')

            # Retrieve the energy in keV:
            # For an energy stack, the data are saved as stack_date_scannbr_nrjXX_000001.nxs 
            # with XX being identified as the self.scan param
            # XX goes from 01 to NN, NN being the number of energies. But a list start from 0 !
            # So we deincrement the self.scan parameter to find the right energy
            self.params["nrj"] = hd5_scaninfo.get('/entry1/camera/energy')[self.scan - 1] * 1e-3
            self.params["startnrj"] = hd5_scaninfo.get('/entry1/camera/energy')[0] * 1e-3

            # define saveprefix
            self.params["saveprefix"] = os.path.join(self.params["folder"], "reconstructed",
                                                     'stack_' + self.params["date"] + '_' + self.params[
                                                         "scannbr"] + '_reconstructed',
                                                     'stack_' + self.params["date"] + '_' + self.params[
                                                         "scannbr"] + '_nrj%02d_run%02d')

            # nxs dark file
            hd5_dark = h5py.File(os.path.join(self.params["folder"], self.params["filedark"]), 'r')
            s = "njr"

        else:
            # Retrieve nrj and define saveprefix
            hd5_scaninfo = h5py.File(os.path.join(self.params["folder"], self.params["filedescr"] % self.scan), 'r')
            self.params["nrj"] = hd5_scaninfo.get('/entry1/camera/energy')[()] * 1e-3
            self.params["saveprefix"] = os.path.join(self.params["folder"], "reconstructed",
                                                     'image_' + self.params["date"] + '_%03d_reconstructed_run%02d')

            # nxs dark file
            hd5_dark = h5py.File(os.path.join(self.params["folder"], self.params["filedark"] % self.scan), 'r')
            s = "image"

        print('##############################################')
        print("Start loading data for %s number %d ; energy = %.3f eV" % (s, self.scan, self.params["nrj"] * 1e3))

        # Load the sample data from the nxs
        sample_data_entry = hd5_sample.get('/entry/scan_data/')
        # Get the name of the image data (e.g. SI107_20191201_image)
        # It is the first entry of the '/entry/scan_data/' group of nxs file.
        # Need to be modified if it is no longer the case
        sample_data_imgname = list(sample_data_entry.keys())[0]
        # Loading only the images corresponding to the imgn variable (in order to take into account moduloframe and maxframe params)
        # and broadcasting to float32. 
        # Purpose of separating into two case is to avoid making a copy if the full data are used
        # (a copy is mandatory for the sliced data, so it will be longer to load sliced data than the full ones ... )
        if self.imgn.shape[0] == sample_data_entry.get(sample_data_imgname).shape[0]:
            self.raw_data = np.array(sample_data_entry.get(sample_data_imgname), dtype=np.float32, copy=False)
        else:
            self.raw_data = np.array(sample_data_entry[sample_data_imgname][self.imgn, :, :], dtype=np.float32,
                                     copy=False)

        # Load the same way the dark from nxs
        dark_data_entry = hd5_dark.get('/entry/scan_data/')
        dark_data_imgname = list(dark_data_entry.keys())[0]
        dark_data = dark_data_entry[dark_data_imgname][1:, :, :]  # Avoiding the first dark image (a not good one)
        dark_average = np.mean(dark_data, axis=0, dtype=np.float32)

        print('##############################################')
        print("Start Preprocessing for %s %d" % (s, self.scan))

        if self.params["dark_as_optim"] == False:
            print("dark subtraction and applying threshold")
            # Subtract dark and apply threshold: parallel with numexpr version
            ne_expression = "where(data - dark > thr, data - dark, 0)"
            ne_localdict = {"data": self.raw_data, "dark": dark_average, "thr": self.params["threshold"]}
            ne.evaluate(ne_expression, local_dict=ne_localdict, out=self.raw_data)

            # Subtract dark and apply threshold: classic numpy version
            #        for i in range(len(self.raw_data)):
            #            self.raw_data[i] = np.where(self.raw_data[i]-dark_average < self.params["threshold"], 0, self.raw_data[i]-dark_average)

            # Bring dark_subtract to 0 in order not to subtract twice
            self.params["dark_subtract"] = 0

        else:
            # Load the dark in order to use it in the reconstruction
            self.dark = dark_average

            if np.any(self.raw_data < 0):
                raise PtychoRunnerException("some raw data below zero ??!!")


        # Adapt the maxsize or the detector distance if it is an energy stack
        # depending if detector distance has been moved (hard way: the detector distance changes) or not (software way: maxsize changes)
        if self.params["scantype"] == "stack" and self.scan != self.params["scan"].split(',')[0]:
            if self.params["adaptq"] == 'soft':
                self.params["maxsize"] = calcROI(self.params["maxsizeini"], self.params["detectordistance"],
                                                 self.params["startnrj"] * 1e3, self.params["nrj"] * 1e3)
            elif self.params["adaptq"] == 'hard':
                self.params["detectordistance"] = calcDistance(self.params["detectordistanceini"],
                                                               self.params["startnrj"] * 1e3, self.params["nrj"] * 1e3)

        # Check if a camera center has been given, and define the according camera roi in that case
        if self.params['camcenter'] != None:
            xmin = self.params['camcenter'][0] - self.params['maxsize'] // 2
            xmax = self.params['camcenter'][0] + self.params['maxsize'] // 2
            ymin = self.params['camcenter'][1] - self.params['maxsize'] // 2
            ymax = self.params['camcenter'][1] + self.params['maxsize'] // 2
            self.params['roi'] = '%.4d,%.4d,%.4d,%.4d' % (xmin, xmax, ymin, ymax)

        # Apply the load_data_post_process function
        # It initialize the dark, the mask and the flatfield variables
        # If none are given, initialize at 0 for dark and mask and at 1 for flatfield
        self.load_data_post_process()

        print("Preprocessing over")
        print('##############################################')
        print("Reconstruction of %s %d will start with the following parameters :" % (s, self.scan))
        self.print_params_Hermes()
        print('##############################################')

    def save(self, run, stepnum=None, algostring=None):
        """
        Overwriting of the save function
        
        Purpose:
        - updating the reconstruction log file 
        - saving llk value if user params says so
        
        The original function is called via super() 
        Only a part is added at the end
        
        Save the result of the optimization, and (if  self.params['saveplot'] is True) the corresponding plot.
        This is an internal function.

        :param run:  the run number (integer)
        :param stepnum: the step number in the set of algorithm steps
        :param algostring: the string corresponding to all the algorithms ran so far, e.g. '100DM,100AP,100ML'
        :return:
        """

        super().save(run, stepnum, algostring)

        #### ADDED PART HERE ####
        # Update the logfile here
        self.update_logfile(run)

        # Save LLKs values in dedicated text file
        if self.params['savellk']:
            self.save_llk(run)

    #### END OF ADDED PART ####

    # --------------------------------------------------------------------------
    def save_plot(self, run, stepnum=None, algostring=None, display_plot=False):
        """
        Overwriting of the save_plot function
        
        Purpose: saving also the reconstruction data as four tif files:
        - two for the object (amplitude and phase)
        - two for the probe (amplitude and phase)
        
        The original function is called via super() 
        Only a part is added at the end to do the tif files

        Save the plot to a png file.

        :param run:  the run number (integer)
        :param stepnum: the step number in the set of algorithm steps
        :param algostring: the string corresponding to all the algorithms ran so far, e.g. '100DM,100AP,100ML'
        :param display_plot: if True, the saved plot will also be displayed
        :return:
        """

        super().save_plot(run, stepnum, algostring, display_plot)

        ##################################################################################
        #        # Added part here to save as tif files also the reconstruction 

        if self.params["savetiff"] != None:

            # Get the obj, probe and scanned area for the probe and object (copy-pasted from parent function) 

            if 'split' in self.params['mpi']:
                self.p.stitch(sync=True)
                obj = self.p.mpi_obj
                scan_area_obj = self.p.get_mpi_scan_area_obj()
                scan_area_points = self.p.get_mpi_scan_area_points()
                if not self.mpi_master:
                    return
            else:
                obj = self.p.get_obj()
                scan_area_obj = self.p.get_scan_area_obj()
                scan_area_points = self.p.get_scan_area_points()
            scan_area_probe = self.p.get_scan_area_probe()

            if self.p.data.near_field or not self.params['remove_obj_phase_ramp']:
                obj = obj[0]
                probe = self.p.get_probe()[0]
            else:
                obj = phase.minimize_grad_phase(obj[0], center_phase=0, global_min=False,
                                                mask=~scan_area_obj, rebin_f=2)[0]
                probe = phase.minimize_grad_phase(self.p.get_probe()[0], center_phase=0, global_min=False,
                                                  mask=~scan_area_probe, rebin_f=2)[0]

            # Get the amplitude and the phase of object
            if self.params["savetiff"] == "full":
                obj_amp = Image.fromarray(np.abs(obj))  # object AMPLITUDE as Image 
                obj_phase = Image.fromarray(np.angle(obj))  # Object PHASE as Image 
                probe_amp = Image.fromarray(np.abs(probe))  # Probve AMPLITUDE as Image 
                probe_phase = Image.fromarray(np.angle(probe))  # Probve PHASE as Image 

            elif self.params["savetiff"] == "crop":

                # Get the indices of the object and probe where they are actually reconstructed
                xmin_obj, ymin_obj = np.argwhere(scan_area_obj).min(axis=0)
                xmax_obj, ymax_obj = np.argwhere(scan_area_obj).max(axis=0)
                xmin_probe, ymin_probe = np.argwhere(scan_area_probe).min(axis=0)
                xmax_probe, ymax_probe = np.argwhere(scan_area_probe).max(axis=0)

                # This part is needed because sometimes (why ???), scan_area_obj and scan_area_probe give a strange size of the object 
                # (especially a non square reconstructed image araise from a square scan ...)
                sizex_obj = xmax_obj - xmin_obj
                sizey_obj = ymax_obj - ymin_obj
                sizex_probe = xmax_probe - xmin_probe
                sizey_probe = ymax_probe - ymin_probe

                if sizey_obj > sizex_obj:
                    diff = sizey_obj - sizex_obj
                    xmax_obj += diff
                    print("crop obj X resized")
                elif sizex_obj > sizey_obj:
                    diff = sizex_obj - sizey_obj
                    ymax_obj += diff
                    print("crop obj Y resized")

                if sizey_probe > sizex_probe:
                    diff = sizey_probe - sizex_probe
                    xmax_probe += diff
                    print("crop probe X resized")
                elif sizex_probe > sizey_probe:
                    diff = sizex_probe - sizey_probe
                    ymax_probe += diff
                    print("crop probe Y resized")

                obj_amp = Image.fromarray(np.abs(
                    obj[xmin_obj:xmax_obj, ymin_obj:ymax_obj]))  # object AMPLITUDE as Image cropped with the right size
                obj_phase = Image.fromarray(np.angle(
                    obj[xmin_obj:xmax_obj, ymin_obj:ymax_obj]))  # Object PHASE as Image cropped with the right size
                probe_amp = Image.fromarray(np.abs(probe[xmin_probe:xmax_probe,
                                                   ymin_probe:ymax_probe]))  # Probe AMPLITUDE as Image cropped with the right size
                probe_phase = Image.fromarray(np.angle(probe[xmin_probe:xmax_probe,
                                                       ymin_probe:ymax_probe]))  # Probe PHASE as Image cropped with the right size

            # Build the saving path for all images
            savepath_obj_amp = self.params["saveprefix"] % (self.scan, run) + '_Object_Amplitude.tif'
            savepath_obj_phase = self.params["saveprefix"] % (self.scan, run) + '_Object_Phase.tif'
            savepath_probe_amp = self.params["saveprefix"] % (self.scan, run) + '_Probe_Amplitude.tif'
            savepath_probe_phase = self.params["saveprefix"] % (self.scan, run) + '_Probe_Phase.tif'

            # Save the images
            obj_amp.save(savepath_obj_amp)
            obj_phase.save(savepath_obj_phase)
            probe_amp.save(savepath_probe_amp)
            probe_phase.save(savepath_probe_phase)

    #        # Save only the object amplitude in user defined ruchepath
    #        if self.params['ruchepath'] != None:
    #            print("savepathruche = ")
    #            print(os.path.join(self.params['ruchepath'], "reconstructed", self.params["filesample"][:-10] + "_run%.2d_Object_Amplitude.tif"))
    #            savepathruche = os.path.join(self.params['ruchepath'], "reconstructed", self.params["filesample"][:-10] + "_run%.2d_Object_Amplitude.tif") %(self.scan, run)

    #            obj_amp.save(savepathruche)

    #        ## END of ADDED PART here
    #        ##########################################################

    # -------------------------------------------------------------------------
    # Logfile updating function
    def update_logfile(self, run):
        logfile = open(self.params['logfile'], 'a')
        s = ["############################################\n"]
        s.append("date of reconstruction: " + time.asctime() + "\n")
        s.append("file reconstructed: " + self.params["filesample"][:-11] % self.scan + "\n")
        s.append("Run number: " + str(run) + "\n")

        for key, value in self.params.items():
            if key in ['algorithm', 'maxsize', 'threshold', 'detectordistance', 'rebin', 'defocus', 'probe',
                       'scantype']:
                s.append(str(key) + " = " + str(value) + "\n")

        s.append('LLK=%.3f\n' % (self.p.llk_poisson / self.p.nb_obs))

        savepath = self.params["saveprefix"] % (self.scan, run)
        s.append("Saved in: " + savepath + ".cxi\n")
        s.append("END\n\n")

        logfile.writelines(s)
        logfile.close()
        print()
        print('Updated log file: ' + self.params['logfile'])
        print()

    # -------------------------------------------------------------------------
    # Saving the llk values in text file function
    def save_llk(self, run):
        all_llk = np.array([[k] for k, v in self.p.history['llk_poisson'].items()])
        headerllk = "cycle\t"
        for whichllk in ['llk_poisson', 'llk_gaussian', 'llk_euclidian']:
            headerllk += whichllk + '\t'
            thisllk = np.array([[v] for k, v in self.p.history[whichllk].items()])
            all_llk = np.concatenate((all_llk, thisllk), axis=1)
        llkfilename = self.params["saveprefix"] % (self.scan, run) + '_everyllk.txt'
        print("\nSaving all llk values in " + llkfilename + "\n")
        np.savetxt(llkfilename, all_llk, delimiter='\t', header=headerllk)

    # --------------------------------------------------------------------------
    def print_params_Hermes(self):
        for k, v in self.params.items():
            if k in ["folder", "scantype", "savellk", "detectordistance", "threshold", "adaptq", "real_coordinates",
                     "camcenter", "mpi", "nrj", "maxsize", "probe", "defocus", "algorithm", "scan", "dark_as_optim"]:
                print(k + ' : ' + str(v))


# -----------------------------------------------------------------
# Hermes Ptycho Runner class

class PtychoRunnerHermes(PtychoRunner):
    """
    Class to process a series of scans with a series of algorithms, given from the command-line
    """

    def __init__(self, argv, params, ptycho_runner_scan_class):

        self.help = False
        super(PtychoRunnerHermes, self).__init__(argv, params, ptycho_runner_scan_class)

        self.help_text += helptext_beamline

        self.redefine_scanparam()

    def parse_arg_beamline(self, k, v):
        """
        Parse argument in a beamline-specific way. This function only parses single arguments.
        If an argument is recognized and interpreted, the corresponding value is added to self.params

        This method should be superseded in a beamline/instrument-specific child class.

        Returns:
            True if the argument is interpreted, false otherwise
        """

        if 'help' in k:
            self.help = True

        elif 'folder' in k:
            if os.path.isdir(v):
                self.params['folder'] = v
                return True
            else:
                return False

        elif 'threshold' == k:
            try:
                int(eval(v))
                self.params[k] = eval(v)
                return True
            except:
                return False

        elif 'bin' in k:
            try:
                vv = int(eval(v))
                self.params['rebin'] = vv
                return True
            except:
                return False

        elif 'distance' in k:
            try:
                vv = float(eval(v))
                self.params['detectordistance'] = vv
                return True
            except:
                return False

        elif 'real_coordinates' == k:
            self.params["real_coordinates"] = True
            return True

        elif 'onlynrj' in k:
            self.params['onlynrj'] = v
            return True

        elif 'adaptq' == k:
            if 'soft' in v:
                self.params['adaptq'] = 'soft'
                return True
            elif 'hard' in v:
                self.params['adaptq'] = 'hard'
                return True
            elif 'no' in v:
                self.params['adaptq'] = None
                return True
            else:
                return False

        elif 'camcenter' == k:
            try:
                camx0, camy0 = int(v.split(',')[0]), int(v.split(',')[1])
                self.params['camcenter'] = [camx0, camy0]
                return True
            except:
                return False

        elif 'ruchepath' == k:
            if os.path.isdir(v):
                self.params['ruchepath'] = v
                return True
            elif eval(v) == None:
                self.filepaths['ruchepath'] = None
                return True
            else:
                return False

        elif 'probediam' in k:
            try:
                float(eval(v))
                self.params["probe"] = "disc," + v
                return True
            except:
                return False

        elif 'dark_as_optim' == k:
            self.params["dark_as_optim"] = True
            return True

        elif "savetif" in k:
            if "crop" in v:
                self.params["savetiff"] = "crop"
                return True
            elif "full" in v:
                self.params["savetiff"] = "full"
                return True
            elif "no" in v:
                self.params["savetiff"] = None
                return True
            else:
                return False

        elif k == 'savellk':
            self.params['savellk'] = True
            return True

    def check_params_beamline(self):
        """
        Check if self.params includes a minimal set of valid parameters, specific to a beamline.
        Derived implementations can also set default values when appropriate.

        Returns: Nothing. Will raise an exception if necessary
        """

        if self.help:
            raise PtychoRunnerException('You have required some help! No problem, here it is:')

        if self.params["folder"] == None or not os.path.isdir(self.params['folder']):
            raise PtychoRunnerException(
                'Missing or wrong data folder path. Give the following argument: folder=your/path/to/data')

        if self.params["detectordistance"] == None:
            raise PtychoRunnerException(
                'Detector distance parameter must be given. Use the following: distance=XX.XXe-3')
        
    def redefine_scanparam(self):
        """
        Computing the filenames and checking the files are there.
        
        """
        # Store the initial detector distance and maxsize in case adaptq is used
        self.params['detectordistanceini'] = self.params['detectordistance']
        self.params['maxsizeini'] = self.params['maxsize']

        # Here we retrieve the id XX for the data, the dark and the descr
        alldatafiles = [f for f in os.listdir(self.params["folder"]) if '.nxs' in f and 'dark' not in f]
        alldarkfiles = [f for f in os.listdir(self.params["folder"]) if '.nxs' in f and 'dark' in f]
        alldescrfiles = [f for f in os.listdir(self.params["folder"]) if '.hdf5' in f]

        dataids = [int(f.split('_')[2]) for f in alldatafiles]
        darkids = [int(f.split('_')[2]) for f in alldarkfiles]
        descrids = [int(f.split('.')[0].split('_')[2]) for f in alldescrfiles]

        scanfound = []
        scannames = []
        scantypes = []
        scandates = []

        if self.params["scan"] != None:
            # First case: scan nbr are given by the user:
            # Looking for the files which have the same id as the user parameter 'scan'
            # and where there is a data.nxs, a dark.nxs and a descr.hdf5 files
            try:
                scanids = [int(ii) for ii in self.params["scan"].split(',')]
            except ValueError:
                raise PtychoRunnerException("Wrong paramter 'scan'. Use scan=XX,XX,XX with XX the scan number")

            for ii, descrid in enumerate(descrids):
                if descrid in scanids and descrid in darkids and descrid in dataids:
                    scanfound.append(descrid)
                    scannames.append(alldescrfiles[ii].split('.')[0])
                    scantypes.append(alldescrfiles[ii].split('.')[0].split('_')[0].lower())
                    scandates.append(alldescrfiles[ii].split('.')[0].split('_')[1].lower())

            scannotfound = [i for i in scanids if i not in scanfound]

            if scannotfound != []:
                ss = ""
                for scannbr in scannotfound:
                    ss += '%d,' % scannbr
                ss = ss[:-1]
                raise PtychoRunnerException(
                    "Scans number %s not found in folder:%s\nIt could be because the scan nbr is wrong or because one of the file (data.nxs, dark.nxs or descr.hdf5) is missing" % (
                        ss, self.params["folder"]))

        else:
            # Second case: no scan nbr given by the user:
            # Looking for all the files in the folder
            # where there are a data.nxs, a darks.nxs and a descr.hdf5 files

            for ii, descrid in enumerate(descrids):
                if descrid in darkids and descrid in dataids:
                    scanfound.append(descrid)
                    scannames.append(alldescrfiles[ii].split('.')[0])
                    scantypes.append(alldescrfiles[ii].split('.')[0].split('_')[0].lower())
                    scandates.append(alldescrfiles[ii].split('.')[0].split('_')[1].lower())

            if scanfound == []:
                raise PtychoRunnerException(
                    "No scan found in folder:%s\nIt could be because one of the file (data.nxs, dark.nxs or descr.hdf5) is missing" % (
                        self.params["folder"]))
        # Now all the scan to be reconstructed have been found

        # Check the scan type and associate it to the 'scantype' parameter
        if 'image' in scantypes and 'stack' not in scantypes:
            self.params["scantype"] = 'image'
        elif 'stack' in scantypes and 'image' not in scantypes:
            self.params["scantype"] = 'stack'
        elif 'image' in scantypes and 'stack' in scantypes:
            raise PtychoRunnerException(
                "You asked for reconstruction of stacks and images in your folder: %s, but we cannot reconstruct like this for now. Use batch on different folder or express a scan number with scan=XXX" %
                self.params["folder"])
        else:
            raise PtychoRunnerException("Uknown file type")

        if self.params["scantype"] == 'stack':
            if len(scanfound) > 1:
                ss = ""
                for scanname in scannames:
                    ss += '%s,' % scanname
                ss = ss[:-1]
                raise PtychoRunnerException(
                    "You asked for several stacks to be reconstructed on the same batch line, or there is several hdf5 descriptor file in you folder: %s. Several scans cannot be reconstructed on the same batch line, use batch with several lines and scan=XXX to do so. Check also if there are several hdf5 with the same scan nbr in your folder" % ss)
            else:
                scannbr = '%03d' % scanfound[0]
                self.params["scannbr"] = scannbr

        if len(set(scandates)) != 1:
            raise PtychoRunnerException(
                "Ok this is a tricky error! You asked for reconstructions of files which come from different dates. Sorry it cannot be done. Use batch file and scan=XXX to do so")

        date = set(scandates).pop()
        self.params["date"] = date

        # Redefine the 'scan' parameter to fit with PyNX way to do
        scan = ""
        if self.params["scantype"] == 'image':
            # If only images, the scan parameter is the same as the user give,
            # well formated in string of int separated by commas
            scanfound.sort()
            for scanid in scanfound:
                scan += '%d,' % scanid
        else:
            # If it is a stack, the scan parameter will be the nrjs
            allnrjs = [f for f in alldatafiles if "stack_" + date + "_" + scannbr in f]
            allnrjs.sort()

            # and take into account the 'onlynrj' parameter
            if self.params["onlynrj"] == None:
                usednrj = allnrjs
            else:
                usednrj = []
                # Get the nrjs as integer for proper formatting
                try:
                    onlynrj = [int(n) for n in self.params["onlynrj"].split(',')]
                except ValueError:
                    raise PtychoRunnerException(
                        "onlynrj parameter wrongly formated: use onlynrj=XX,XX,XX,etc. with XX the nrj number in the stack, not the nrj in eV!")

                # Goes through every file of the stack and get the filename if the nrj is one of the user defined nrj
                for f in allnrjs:
                    for nrj in onlynrj:
                        if '_nrj%.2d_' % nrj in f:
                            print("nrj %.2d wil be reconstructed" % nrj)
                            usednrj.append(f)

                if len(onlynrj) != len(usednrj):
                    raise PtychoRunnerException(
                        "You have asked for nrjs that are not in the stack. Check again the 'onlynrj' parameter")

                # Format the scan parameter to be the different nrjs
            for filenrj in usednrj:
                scan += '%d,' % int(filenrj.split('_')[3][3:])

        # Complete the 'scan' parameter and remove the last comma 
        self.params["scan"] = scan[:-1]

        # Format the filenames so they can be retrieve with 'filename' % self.scan in the 'load_data' and 'load_scan' functions of PtychoHermesRunnerScan class
        if self.params["scantype"] == "stack":
            self.params[
                "filesample"] = "stack_" + date + "_" + scannbr + '_nrj%.2d_000001.nxs'  # 1st %s: YYYYMMDD ; 2nd %s: XXX
            self.params["filedark"] = "stack_" + date + "_" + scannbr + '_dark_000001.nxs'
            self.params["filedescr"] = "Stack_" + date + "_" + scannbr + '.hdf5'
            return True

        elif self.params["scantype"] == "image":
            self.params["filesample"] = 'image_' + date + '_%.3d_000001.nxs'  # 1st %s: YYYYMMDD
            self.params["filedark"] = 'image_' + date + '_%.3d_dark_000001.nxs'
            self.params["filedescr"] = 'Image_' + date + '_%.3d.hdf5'
