#import packages

import os
import sys
from scipy.signal import butter,filtfilt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy

sys.path.append("/Users/fponce/Documents/vdesert/vdesert_funcs/")
import plot_funcs.mpl_functions as mpl

sns.set()
sns.set_style()

##############################################################################

# # # default values
# fVec, fs, headingVeloThresh, headingVeloCeil, magThresh, magCeil = \
# np.zeros_like(flyAngle),30., 60., 2500., 6., 180.
# #64.,60.,2500.(ceil),6.,180. #the 10 could be 10
# #30., 20., 240., 10., 140. #testing values
#
# deffVec, deffs, defheadingVeloThresh, defheadingVeloCeil, defmagThresh, defmagCeil = \
# fVec, fs, headingVeloThresh, headingVeloCeil, magThresh, magCeil

#############################################################################

#findSacs func
def findSacs_n_plot_test(flyAngle,flyAngle2,fly_n,sFigPath, plotQ,  *args):

# Finds local min-/maxima that are within threshold/ceiling bounds of wing steering velocity (headingVelo)
# Subsequently, integrates the unidirectional headingVelo sections surrounding local extremes,
#and retain only those saccades that supersede magnitude thresholds.
# Usage:
# $ import Sac_ID as sid
# $ SAmx, SAmn, SVmx, SVmn = sid.findSacs(flyAngle)
# IN:
# flyAngle: the Fly Angle filtered
# flyAngle2: the Fly Angle more heavily filtered
# OUT:
# SAmx, SAmn, SVmx, SVmn: np.nan vectors with saccade Angles or saccade Velocities in positive
#and negative directions, respectively

# Extra IN (if you don't want to use defaults):
# All you need to call this function is above this line. rest is extra.
# fVec: A np.zeros vector with NaN entries where the fly is not flying
# fs: the sampling frequency
# headingVeloThresh, headingVeloCeil, magThresh, magCeil: thresholds and magnitudes

#############################################################################
# # default values
    # fVec, fs, headingVeloThresh, headingVeloCeil, magThresh, magCeil = \
    # np.zeros_like(flyAngle),30., 60., 2500., 6., 180.

    #parameters used before 08/20/20
    # fVec, fs, headingVeloThresh, headingVeloCeil, magThresh, magCeil = \
    # np.zeros_like(flyAngle),30., 40., 2500., 6., 180.

    # #parameters used before 08/20/20
    # fVec, fs, headingVeloThresh, headingVeloCeil, magThresh, magCeil = \
    # np.zeros_like(flyAngle),30., 40., 2500., 6., 180.

    #parameters used on 08/20/20
    fVec, fs, headingVeloThresh, headingVeloCeil, magThresh, magCeil = \
    np.zeros_like(flyAngle),30., 30., 2500., 6., 180.

    #64.,60.,2500.(ceil),6.,180. #the 10 could be 10
    #30., 20., 240., 10., 140. #testing values

    deffVec, deffs, defheadingVeloThresh, defheadingVeloCeil, defmagThresh, defmagCeil = \
    fVec, fs, headingVeloThresh, headingVeloCeil, magThresh, magCeil
#############################################################################
## Local functions (inside this function, so they don't need to be separately defined)

    def butter_lowpass(th_order=4, highcut=260, fs=32.):

        nyq = 0.5 * fs
        high = highcut / nyq
        return butter(th_order, high, btype='low')

    def filtTrace(inArr, fs):

        th_order, highcut = 4, 6.#highcut is 6 based of Maimons paper, could be up to 9
        d, c = butter_lowpass(th_order, highcut, fs)
        return filtfilt(d, c, inArr)

    def calcMag(idx, headingVelo, flyAngleFilt):
        # calculate difference in heading angle between nearest local extremes (of opposite sign)

        k=0
        chkMore=True
        refSign=np.sign(headingVelo[idx])
        while chkMore:
            k += 1
            compSign=np.sign(headingVelo[idx-k])
            if refSign != compSign: # if not equal
                onIdx=idx-k+1
                chkMore=False
        k=0
        chkMore=True
        while chkMore:
            k += 1
            compSign=np.sign(headingVelo[idx+k])
            if refSign != compSign:
                offIdx=idx+k
                chkMore=False
        # integrate: BR (behavioral response) = dWSA(offIdx)-dWSA(onIdx)
        return flyAngleFilt[offIdx] - flyAngleFilt[onIdx], onIdx, offIdx


    def sacOnly(inVec, fVec):
        fVecSqz = np.squeeze(fVec)
        outVec = np.array( np.sum([inVec, fVecSqz], axis=0) ).transpose()
        outVec[outVec == 0] = np.nan
        return outVec


###########################################################################################
    ## check for arguments
    varargin = args
    nargin = 1 + len(varargin)
    if nargin == 1:
        fVec, fs, headingVeloThresh, headingVeloCeil, magThresh, magCeil = \
        deffVec, deffs, defheadingVeloThresh, defheadingVeloCeil, defmagThresh, defmagCeil
    elif nargin == 2:
        fs, headingVeloThresh, headingVeloCeil, magThresh, magCeil = \
        deffs, defheadingVeloThresh, defheadingVeloCeil, defmagThresh, defmagCeil
        fVec = varargin[0]
    elif nargin == 3:
        headingVeloThresh, headingVeloCeil, magThresh, magCeil = \
        defheadingVeloThresh, defheadingVeloCeil, defmagThresh, defmagCeil
        fVec, fs = varargin[0], varargin[1]
    elif nargin == 4:
        headingVeloCeil, magThresh, magCeil = defheadingVeloCeil, defmagThresh, defmagCeil
        fVec, fs, headingVeloThresh = varargin[0], varargin[1], varargin[2]
    elif nargin == 5:
        magThresh, magCeil = defmagThresh, defmagCeil
        fVec, fs, headingVeloThresh, headingVeloCeil = varargin[0], varargin[1], varargin[2], varargin[3]
    elif nargin == 6:
        magCeil = defmagCeil
        fVec, fs, headingVeloThresh, headingVeloCeil, magThresh = varargin[0],
        varargin[1], varargin[2], varargin[3], varargin[4]
    elif nargin == 7:
        fVec, fs, headingVeloThresh, headingVeloCeil, magThresh, magCeil = varargin[0],
        varargin[1], varargin[2], varargin[3], varargin[4], varargin[5]
    else:
        print ('could not assign all parameters')
    ###########################################################################################

    ## Low-pass filter the incoming angle vector
    #flyAngleFilt= filtTrace(flyAngle, fs)
    flyAngleFilt = copy.deepcopy(flyAngle)
    flyAngleFilt2 = copy.deepcopy(flyAngle2)
    #flyAngleFilt = flyAngle
    ## d/dt the filtered angle vector

    headingVelo=np.zeros_like(flyAngleFilt)
    for k in range(len(flyAngleFilt)-1):
        headingVelo[k]=(flyAngleFilt[k+1]-flyAngleFilt[k])*fs

    headingVelo2=np.zeros_like(flyAngleFilt2)
    for k in range(len(flyAngleFilt2)-1):
        headingVelo2[k]=(flyAngleFilt2[k+1]-flyAngleFilt2[k])*fs

    # In words: find local extremes that exceed headingVelo threshold during flight
    # subsequently, integrate the unidirectional headingVelo sections surrounding local extremes.

    ## Place holders
    headingVeloMxThr,   headingVeloMnThr= np.zeros_like(headingVelo2), np.zeros_like(headingVelo2)
    SheadingVeloMx,     SheadingVeloMn  = np.zeros_like(headingVelo2), np.zeros_like(headingVelo2)
    lcMxMagTemp,    lcMnMagTemp = np.zeros_like(headingVelo2), np.zeros_like(headingVelo2)
    lcMxMag,        lcMnMag     = np.zeros_like(headingVelo2), np.zeros_like(headingVelo2)
    sacMx,          sacMn       = np.zeros_like(headingVelo2), np.zeros_like(headingVelo2)
    lcMnonIdx,      lcMnoffIdx  = np.zeros_like(headingVelo2), np.zeros_like(headingVelo2)
    lcMxonIdx,      lcMxoffIdx  = np.zeros_like(headingVelo2), np.zeros_like(headingVelo2)
    lcMxdeltaIdx,   lcMndeltaIdx= np.zeros_like(headingVelo2), np.zeros_like(headingVelo2)
    SacAngMx,       SacAngMn    = np.zeros_like(headingVelo2), np.zeros_like(headingVelo2)
    SacIdxMx,       SacIdxMn    = np.zeros_like(headingVelo2), np.zeros_like(headingVelo2)

    ###########################################################################################
    ## Find local extremes in the rate, that exceed the saccade velocity threshold
    #0.5 not trivially chosen
    ###########################################################################################

    reboundIds = int(np.floor(.5*fs))

    for idx in range (reboundIds,len(headingVelo2)-reboundIds):
        if headingVelo2[idx]>=headingVelo2[idx-1] and headingVelo2[idx]>=headingVelo2[idx+1] and \
        headingVelo2[idx]>headingVeloThresh and headingVelo2[idx]<headingVeloCeil:
            headingVeloMxThr[idx] = headingVelo2[idx]
        elif headingVelo2[idx]<=headingVelo2[idx-1] and headingVelo2[idx]<=headingVelo2[idx+1] and \
        headingVelo2[idx]<-headingVeloThresh and headingVelo2[idx]>-headingVeloCeil:
            headingVeloMnThr[idx] = headingVelo2[idx]

    ## Retain extremes that are less than magnitude limit (velocity dependent)
    #(no longer: also exceed the saccade magnitude threshold)
    # Integrate unidirectional headingVelo sections surrounding local extremes to obtain saccade magnitude

    for idx in range (int(1.4*reboundIds),len(headingVelo2)-int(1.4*reboundIds)):
        if abs(headingVeloMxThr[idx])>0: # for every local max
            lcSteerMag, onIdx, offIdx = calcMag(idx, headingVelo2, flyAngleFilt)
            if lcSteerMag < magCeil:
                lcMxMagTemp[idx] = lcSteerMag
                lcMxonIdx[idx] = onIdx
                lcMxoffIdx[idx] = offIdx
                lcMxdeltaIdx[idx] = offIdx-idx
        if abs(headingVeloMnThr[idx])>0: # for every local minimum
            lcSteerMag, onIdx, offIdx = calcMag(idx, headingVelo2, flyAngleFilt)
            if lcSteerMag > -magCeil:
                lcMnMagTemp[idx] = lcSteerMag
                lcMnonIdx[idx] = onIdx
                lcMnoffIdx[idx] = offIdx
                lcMndeltaIdx[idx] = offIdx-idx

    ## Obtain saccades, retained extremes that are not 'rebounds' (stable pre-onset activity for edgeSec/4).

    for idx in range (reboundIds,len(headingVelo2)-reboundIds):
        if lcMxMagTemp[idx]>0: # for every local maximum
            if abs(np.mean(lcMnMagTemp[idx-reboundIds:idx]))<magThresh/reboundIds and \
            ~np.isnan(fVec[idx-reboundIds:idx+reboundIds]).any(axis=0):
                SheadingVeloMx[idx] = headingVelo[idx]
                lcMxMag[idx] = lcMxMagTemp[idx]
                sacMx[int(lcMxonIdx[idx]):int(lcMxoffIdx[idx])] = flyAngleFilt \
                [int(lcMxonIdx[idx]):int(lcMxoffIdx[idx])]
                sacPeakIdx = int(idx+lcMxdeltaIdx[idx])
                SacAngMx[sacPeakIdx] = flyAngleFilt[sacPeakIdx] # such that saccades can be indicated at peaks
                SacIdxMx[idx] = sacPeakIdx # indices of saccades, to allow cropping of saccades
        elif lcMnMagTemp[idx]<0: # for every local minimum
            if abs(np.mean(lcMxMagTemp[idx-reboundIds:idx]))<magThresh/reboundIds and \
            ~np.isnan(fVec[idx-reboundIds:idx+reboundIds]).any(axis=0):
                SheadingVeloMn[idx] = headingVelo[idx]
                lcMnMag[idx] = lcMnMagTemp[idx]
                sacMn[int(lcMnonIdx[idx]):int(lcMnoffIdx[idx])] = flyAngleFilt \
                [int(lcMnonIdx[idx]):int(lcMnoffIdx[idx])]
                sacPeakIdx = int(idx+lcMndeltaIdx[idx])
                SacAngMn[sacPeakIdx] = flyAngleFilt[sacPeakIdx] # such that saccades can be indicated at peaks
                SacIdxMn[idx] = sacPeakIdx # indices of saccades, to allow cropping of saccades
    ## only when fly is flying, set all other indices to nan
    SVMxThr, SVMnThr = sacOnly(headingVeloMxThr, fVec), sacOnly(headingVeloMnThr, fVec) # diagnostic

    #SMagMx, SMagMn   = sacOnly(lcMxMag, fVec), sacOnly(lcMnMag, fVec) # diagnostic
    SSctnMx, SSctnMn = sacOnly(sacMx, fVec), sacOnly(sacMn, fVec)

    ## plot to verify
    if plotQ:
        pltSacs(flyAngle, fly_n, sFigPath, headingVelo2, SVMxThr, SVMnThr, flyAngleFilt, SSctnMx, SSctnMn)

    SMagMx, SMagMn = sacOnly(lcMxMag, fVec), sacOnly(lcMnMag, fVec)
    SAmx, SAmn = sacOnly(SacAngMx, fVec), sacOnly(SacAngMn, fVec)
    SVmx, SVmn = sacOnly(SheadingVeloMx, fVec), sacOnly(SheadingVeloMn, fVec)
    SIdxmx, SIdxmn = sacOnly(SacIdxMx, fVec), sacOnly(SacIdxMn, fVec)

    # print('-------------------------')
    # print(len(SAmx[~np.isnan(SAmx)]))
    # print(len(SVmx[~np.isnan(SVmx)]))
    # print(len(SMagMx[~np.isnan(SMagMx)]))
    #
    # print('---')
    # print(len(SAmn[~np.isnan(SAmn)]))
    # print(len(SVmn[~np.isnan(SVmn)]))
    # print(len(SMagMn[~np.isnan(SMagMn)]))

    return SMagMx, SMagMn, SAmx, SAmn, SVmx, SVmn, headingVelo2, SVMxThr, SVMnThr, flyAngleFilt, SSctnMx, SSctnMn
    #return SMagMx, SMagMn, SAmx, SAmn, SVmx, SVmn, headingVelo2, SVMxThr, SVMnThr, flyAngleFilt, SSctnMx, SSctnMn
    #SAMx instantaneous angle where saccade happens

#def classSacs(SacIdxVec, LIds, fs):
    #    import numpy as np
    #    # Separate expansion-associated saccades (L) from spontaneous saccades (S, the remainder)
    #    # from CB paper:
    #    # "classified as loom-elicited,
    #    # when the peak L-R velocity occurred within 0.5 s of the mean response
    #    # peak relative to the end of the looming stimulus,
    #    # or classified as spontaneous"
    #    # LPkRespIdx = index where peak is expected (the mean response time)
    #    stimPeakIds = 1.2*fs
    #    LPkRespIds = LIds + stimPeakIds # behavioral response peaks at 1.2 s after stimulus onset (.2 s after max expansion reached)
    #    nanVec, SacIdxVecLm, SacIdxVecSpnt =  np.zeros_like(SacIdxVec)*np.nan, np.zeros_like(SacIdxVec)*np.nan, SacIdxVec
    #    QrtrSecFrames= .5*fs
    #    for LmNum, LmIdx in enumerate(LPkRespIds): # for every (loom) stimulus peak response index
    #        stimRangeFr, stimRangeTo  = int(LmIdx-QrtrSecFrames), int(LmIdx+QrtrSecFrames)
    #        SacIdxVecLm[stimRangeFr:stimRangeTo] = SacIdxVec[stimRangeFr:stimRangeTo] # simply copy the SacIdxVec for the stimulus windows
    #    SacIdxVecSpnt[~np.isnan(SacIdxVecLm)] = nanVec[~np.isnan(SacIdxVecLm)] # when expansion-associated saccades are present, classify as not spontaneous
    #    return SacIdxVecLm, SacIdxVecSpnt
