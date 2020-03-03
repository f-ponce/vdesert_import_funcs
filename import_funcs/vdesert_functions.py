#import packages
import numpy as np
import copy
import pandas
import os
import imp
import pickle
from scipy.interpolate import interp1d
import warnings
import time
import matplotlib.pyplot as plt
import inspect
import types
import math as mat
import seaborn as sns
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import datetime
from matplotlib import animation
from IPython.display import HTML
from tempfile import NamedTemporaryFile
import matplotlib.backends.backend_pdf
from scipy import interpolate
from itertools import groupby
import re
import csv
import copy
from fnmatch import fnmatch
import json
import h5py
import copy
import math
pi = math.pi

##########################################################################################

#import functions
#used to import data to python notebooks

def get_topics_from_hdf5(datapaths):
#gets data from hdf5 file to arrays and generates a list per topic
#each list len is the number of files in dataset

    all_params_ts = []
    all_data_params = []

    all_ros_ts = []
    all_ts = []
    all_elapsed_time = []
    all_trial_index = []
    all_trial_elapsed_time = []
    all_angle_for_autostep = []
    all_init_angle = []
    all_autostep_running = []
    all_autostep_started = []
    all_autostep_stopped = []
    all_flow_running = []
    all_flow_started = []
    all_flow_stopped = []
    all_panels_running = []
    all_panels_started = []
    all_panels_stopped = []

    all_magnotether_angle = []
    all_magnotether_ros_tstamps = []
    all_magnotether_tstamps = []

    all_motion_ros_tstamps = []
    all_motion_tstamps = []
    all_motion_setpoint = []
    all_motion_position = []

    all_ledpanels_ros_tstamps = []
    all_ledpanels_command = []
    all_ledpanels_1 = []
    all_ledpanels_2 = []
    all_ledpanels_3 = []
    all_ledpanels_4 = []
    all_ledpanels_5 = []
    all_ledpanels_6 = []

    all_alicat_ros_tstamps = []
    all_alicat_devices = []

    all_sun_ros_tstamps = []
    all_sun_red = []
    all_sun_green = []
    all_sun_blue = []
    all_sun_message = []
    all_sun_led_number = []

    for i in range(len(datapaths)):
        f = h5py.File(datapaths[i], "r")

        #parameters topic
    #     params_ts = np.asarray(f['data_params_ros_tstamps'])
    #     data_params = np.asarray(f['data_params'])

        #virtual_desert topic
        ros_ts = np.asarray(f['ros_tstamps'])
        ts = np.asarray(f['tstamps'])
        elapsed_time = np.asarray(f['elapsed_time'])
        trial_index = np.asarray(f['current_trial_index'])
        trial_elapsed_time = np.asarray(f['trial_e_time'])
        angle_for_autostep = np.asarray(f['angle'])
        init_angle = np.asarray(f['init_angle'])

        #actions
        autostep_running = np.asarray(f['autostep_action_running'])
        autostep_started = np.asarray(f['autostep_action_started'])
        autostep_stopped = np.asarray(f['autostep_action_stopped'])

        flow_running = np.asarray(f['flow_action_running'])
        flow_started = np.asarray(f['flow_action_started'])
        flow_stopped = np.asarray(f['flow_action_stopped'])

        panels_running = np.asarray(f['panels_action_running'])
        panels_started = np.asarray(f['panels_action_started'])
        panels_stopped = np.asarray(f['panels_action_stopped'])

        #magnotether_angle topic
        magnotether_angle = np.asarray(f['magnotether_angle'])
        magnotether_ros_tstamps = np.asarray(f['magnotether_ros_tstamps'])
        magnotether_tstamps = np.asarray(f['magnotether_tstamps'])

        #motion_data topic
        motion_ros_tstamps = np.asarray(f['motion_data_ros_tstamps'])
        motion_tstamps = np.asarray(f['motion_data_tstamps'])
        motion_setpoint = np.asarray(f['motion_data_setpoint'])
        motion_position = np.asarray(f['motion_data_position'])

        #ledpanels topic
        ledpanels_ros_tstamps = np.asarray(f['ledpanels_ros_tstamps'])
        ledpanels_command = np.asarray(f['ledpanels_panels_command'])
        ledpanels_1 = np.asarray(f['ledpanels_panels_arg1'])
        ledpanels_2 = np.asarray(f['ledpanels_panels_arg2'])
        ledpanels_3 = np.asarray(f['ledpanels_panels_arg3'])
        ledpanels_4 = np.asarray(f['ledpanels_panels_arg4'])
        ledpanels_5 = np.asarray(f['ledpanels_panels_arg5'])
        ledpanels_6 = np.asarray(f['ledpanels_panels_arg6'])

        #alicat topic
        alicat_ros_tstamps = np.asarray(f['alicat_ros_tstamps'])
        alicat_devices = np.asarray(f['alicat_devices'])

        #sun topic
        sun_ros_tstamps = np.asarray(f['sun_ros_tstamps'])
        sun_red = np.asarray(f['sun_red'])
        sun_green = np.asarray(f['sun_green'])
        sun_blue = np.asarray(f['sun_blue'])
        sun_message = np.asarray(f['sun_message'])
        sun_led_number = np.asarray(f['sun_led_number'])

    #     all_params_ts.append(params_ts)
    #     all_data_params.append(data_params)

        all_ros_ts.append(ros_ts)
        all_ts.append(ts)
        all_elapsed_time.append(elapsed_time)
        all_trial_index.append(trial_index)
        all_trial_elapsed_time.append(trial_elapsed_time)
        all_angle_for_autostep.append(angle_for_autostep)
        all_init_angle.append(init_angle)
        all_autostep_running.append(autostep_running)
        all_autostep_started.append(autostep_started)
        all_autostep_stopped.append(autostep_stopped)
        all_flow_running.append(flow_running)
        all_flow_started.append(flow_started)
        all_flow_stopped.append(flow_stopped)
        all_panels_running.append(panels_running)
        all_panels_started.append(panels_started)
        all_panels_stopped.append(panels_stopped)
        all_magnotether_angle.append(magnotether_angle)
        all_magnotether_ros_tstamps.append(magnotether_ros_tstamps)
        all_magnotether_tstamps.append(magnotether_tstamps)
        all_motion_ros_tstamps.append(motion_ros_tstamps)
        all_motion_tstamps.append(motion_tstamps)
        all_motion_setpoint.append(motion_setpoint)
        all_motion_position.append(motion_position)
        all_ledpanels_1.append(ledpanels_1)
        all_ledpanels_2.append(ledpanels_2)
        all_ledpanels_3.append(ledpanels_3)
        all_ledpanels_4.append(ledpanels_4)
        all_ledpanels_5.append(ledpanels_5)
        all_ledpanels_6.append(ledpanels_6)
        all_ledpanels_command.append(ledpanels_command)
        all_ledpanels_ros_tstamps.append(ledpanels_ros_tstamps)
        all_alicat_ros_tstamps.append(alicat_ros_tstamps)
        all_alicat_devices.append(alicat_devices)
        all_sun_ros_tstamps.append(sun_ros_tstamps)
        all_sun_red.append(sun_red)
        all_sun_green.append(sun_green)
        all_sun_blue.append(sun_blue)
        all_sun_led_number.append(sun_led_number)

        return  all_ts,\
all_elapsed_time,\
all_trial_index,\
all_trial_elapsed_time,\
all_angle_for_autostep,\
all_init_angle,\
all_autostep_running,\
all_autostep_started,\
all_autostep_stopped,\
all_flow_running,\
all_flow_started,\
all_flow_stopped,\
all_panels_running,\
all_panels_started,\
all_panels_stopped,\
all_magnotether_angle,\
all_magnotether_ros_tstamps,\
all_magnotether_tstamps,\
all_motion_ros_tstamps,\
all_motion_tstamps,\
all_motion_setpoint,\
all_motion_position,\
all_ledpanels_1,\
all_ledpanels_2,\
all_ledpanels_3,\
all_ledpanels_4,\
all_ledpanels_5,\
all_ledpanels_6,\
all_ledpanels_command,\
all_ledpanels_ros_tstamps,\
all_alicat_ros_tstamps,\
all_alicat_devices,\
all_sun_ros_tstamps,\
all_sun_red,\
all_sun_green,\
all_sun_blue,\
all_sun_led_number

##########################################################################################

def get_all_trial_start_n_end_times(datapaths, number_trials, all_elapsed_time, all_trial_index):
#this gets the start and end times (using all_elapsed_time and all_trial_index)
#when the trial changes in the virtual desert node
#generates a list of arrays, list len is number of files
#datapaths is a list of paths to each file in dataset
#number_trials is the number of trials in experiment

    all_start_times = []
    all_end_times = []
    for i in range(len(datapaths)):
        start_times_trials = []
        end_times_trials = []
        for j in range(number_trials):
            start_time = all_elapsed_time[i][np.where(all_trial_index[i]==j)][0]
            end_time = all_elapsed_time[i][np.where(all_trial_index[i]==j)][-1]
            start_times_trials.append(start_time)
            end_times_trials.append(end_time)
        all_start_times.append(start_times_trials)
        all_end_times.append(end_times_trials)

    return all_start_times, all_end_times

##########################################################################################

def get_all_trial_start_n_end_frames(datapaths, number_trials, all_trial_index):
#this gets the start and end frames (using all_trial_index)
#when the trial changes in the virtual desert node
#generates a list of arrays, list len is number of files
#datapaths is a list of paths to each file in dataset
#number_trials is the number of trials in experiment

    all_start_frames = []
    all_end_frames = []
    for i in range(len(datapaths)):
        start_frames_trials = []
        end_frames_trials = []
        for j in range(number_trials):
            start_frame = [np.where(all_trial_index[i]==j)][0][0][0]
            end_frame = [np.where(all_trial_index[i]==j)][0][0][-1]
            start_frames_trials.append(start_frame)
            end_frames_trials.append(end_frame)
        all_start_frames.append(start_frames_trials)
        all_end_frames.append(end_frames_trials)

    return all_start_frames, all_end_frames

##########################################################################################

def get_elapsed_time(my_list):
#for list of lists
#getting elapsed time of time stamps to use
    all_t_ellapsed = []
    for i in range(len(my_list)):
        t_ellapsed = my_list[i] - my_list[i][0]
        all_t_ellapsed.append(t_ellapsed)
    return all_t_ellapsed

##########################################################################################
#interpolation of magnotether angles

def get_all_magnotether_interp_angles (all_magnotether_angle, all_t_ellapsed, reg_t):
#all_magnotether_angle: list of magnotether angles
#all_t_ellapsed: list of elapsed times of each file
#reg_t: vector of evenly spaced time

    all_magnotether_interp_angles = []
    for i in range(len(all_magnotether_angle)):
        mysecs_np = all_t_ellapsed[i]
        myangles_np = all_magnotether_angle[i]
        f_a = interp1d(mysecs_np, myangles_np, bounds_error=False)
        reg_a = f_a(reg_t)
        all_magnotether_interp_angles.append(reg_a)
    return all_magnotether_interp_angles

##########################################################################################

def find_nearest(array, value):
#for list of lists
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

##########################################################################################

def get_start_n_end_times_m (all_start_times, all_end_times, reg_t):
#get the closest times that correspond
#to start and end times (that come from vdesert node) in the reg_t
    all_start_times_m = []
    all_end_times_m = []
    for i in range(len(all_start_times)):
        start_times_trials_m = []
        end_times_trials_m = []
        for j in range(len(all_start_times[0])):
            start_times_m = find_nearest(reg_t, all_start_times[i][j])
            end_times_m = find_nearest(reg_t, all_end_times[i][j])
            start_times_trials_m.append(start_times_m)
            end_times_trials_m.append(end_times_m)
        all_start_times_m.append(start_times_trials_m)
        all_end_times_m.append(end_times_trials_m)
    return all_start_times_m, all_end_times_m

def get_start_n_end_frames_m (all_start_times, all_end_times, reg_t):
#get the closest times that correspond
#to start and end times (that come from vdesert node) in the reg_t
    all_start_frames_m = []
    all_end_frames_m = []
    for i in range(len(all_start_times)):
        start_frames_trials_m = []
        end_frames_trials_m = []
        for j in range(len(all_start_times[0])):
            start_frames_m = find_nearest_idx(reg_t, all_start_times[i][j])
            end_frames_m = find_nearest_idx(reg_t, all_end_times[i][j])
            start_frames_trials_m.append(start_frames_m)
            end_frames_trials_m.append(end_frames_m)
        all_start_frames_m.append(start_frames_trials_m)
        all_end_frames_m.append(end_frames_trials_m)
    return all_start_frames_m, all_end_frames_m

##########################################################################################

def get_first_n_last_minute_trial (all_start_frames_m, all_end_frames_m, number_frames_per_sec):
#get start frames for last minute and end frames first min
    all_start_frames_m_lm = []
    all_end_frames_m_fm = []
    for i in range(len(all_start_frames_m)):
        start_frames_m_lm_trials = []
        end_frames_m_fm_trials = []
        for j in range(len(all_start_frames_m[0])):
            start_frames_m_lm = all_end_frames_m[i][j] - number_frames_per_sec*60 #getting start frames last min
            end_frames_m_fm = all_start_frames_m[i][j] + number_frames_per_sec*60 #getting end frames first min
            start_frames_m_lm_trials.append(start_frames_m_lm)
            end_frames_m_fm_trials.append(end_frames_m_fm)
        all_start_frames_m_lm.append(start_frames_m_lm_trials)
        all_end_frames_m_fm.append(end_frames_m_fm_trials)
    return all_start_frames_m_lm, all_end_frames_m_fm

##########################################################################################

def get_idx_panels_commands(ledpanels_command):
#gets the indeces of the set pattern id command and
# the gain command
    idx_pat_command = [i for i, x in enumerate(ledpanels_command)
               if x == b'set_pattern_id']
    idx_gain_command = [i for i, x in enumerate(ledpanels_command)
               if x == b'send_gain_bias']
    idx_stop_command = [i for i, x in enumerate(ledpanels_command)
               if x == b'stop']
    return idx_pat_command, idx_gain_command, idx_stop_command

##########################################################################################
