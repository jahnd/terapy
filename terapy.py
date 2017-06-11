# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows, argrelmin
from scipy.constants import c
from scipy.optimize import minimize
from scipy.interpolate import interp1d

class TDS:

    def __init__(self,sampleName = None):
        self.FMAX = 6e12
        self.fminCalculation = 0.3e12
        self.fmaxCalculation = 2e12

        self.fminPhase = 0.4e12
        self.fmaxPhase = 1.4e12
        self.windowSlope = 5e-12

        self.fbins = None
        self.sn = sampleName
        
    def calculateTransferFunction(self):
        self.freqs, self.fdsam, self.phasesam = calculateSpectrum(
            self.taxis, self.sample, self.FMAX, self.windowSlope, self.fbins)

        self.freqs, self.fdref, self.phaseref = calculateSpectrum(
            self.taxis, self.reference, self.FMAX, self.windowSlope, self.fbins)

        self.phasesam, b = phaseOffsetRemoval(
            self.freqs, self.phasesam, self.fminPhase, self.fmaxPhase)
        self.phaseref, a = phaseOffsetRemoval(
            self.freqs, self.phaseref, self.fminPhase, self.fmaxPhase)

        self.avg_phaseslope = b[0] - a[0]

        # Calculate full experimental, phase corrected transfer function
        self.Habs = abs(self.fdsam) / abs(self.fdref)
        self.Hphase = self.phasesam - self.phaseref
        
        #calculate transfer function in calculation domain
        self.changeCalculationDomain(self.fminCalculation,self.fmaxCalculation)

    def loadData(self, fns_ref, fns_sam, timefactor=1e-12, tmax=None):
        self.raw_taxis, self.raw_reference, self.raw_var_r = loadfiles(fns_ref)
        self.raw_taxis, self.raw_sample, self.raw_var_s = loadfiles(fns_sam)
        if self.sn is None:
            self.sn = fns_sam[0].split('/')[-1]

        self.raw_taxis *= timefactor
        self.cropTimeData(tmax)

    def cropTimeData(self, tmax):
        # crop data to desired range
        t, self.reference = cropTimeData(self.raw_taxis, self.raw_reference, tmax)
        t, self.sample = cropTimeData(self.raw_taxis, self.raw_sample, tmax)
        t, self.var_r = cropTimeData(self.raw_taxis, self.raw_var_r, tmax)
        self.taxis, self.var_s = cropTimeData(self.raw_taxis, self.raw_var_s, tmax)
        
    def changeCalculationDomain(self,fmin,fmax):
        self.fminCalculation = fmin
        self.fmaxCalculation = fmax
        self.fr, self.Har = cropFrequencyData(self.freqs, self.Habs, 
                                    self.fminCalculation, self.fmaxCalculation)
        self.fr, self.Hpr = cropFrequencyData(self.freqs, self.Hphase, 
                                    self.fminCalculation, self.fmaxCalculation)

    def changePhaseInterpolationDomain(self,fmin,fmax):
        self.fminPhase = fmin
        self.fmaxPhase = fmax
        self.calculateTransferFunction()

    def plotTimeDomainData(self):
        plt.figure()
        plt.plot(self.taxis * 1e12, self.sample, 'k')
        plt.plot(self.taxis * 1e12, self.reference, 'r')
        plt.xlabel('Time (ps)')
        plt.ylabel('Amplitude')

    def plotFrequencyDomainData(self):
        plt.figure()
        plt.subplot(121)
        plt.plot(self.freqs / 1e12, 20 * np.log10(abs(self.fdsam)), 'k')
        plt.plot(self.freqs / 1e12, 20 * np.log10(abs(self.fdref)), 'r')
        plt.axvspan(self.fminPhase / 1e12, self.fmaxPhase / 1e12,
                    color='k', alpha=0.2)
        plt.axvspan(self.fminCalculation / 1e12, self.fmaxCalculation / 1e12,
                    color='b', alpha=0.1)
        plt.xlabel('Frequency (THz)')
        plt.ylabel('Amplitude (dB)')
        plt.subplot(122)
        plt.plot(self.freqs / 1e12, self.phaseref, 'r')
        plt.plot(self.freqs / 1e12, self.phasesam, 'k')
        plt.axvspan(self.fminPhase / 1e12, self.fmaxPhase / 1e12,
                    color='k', alpha=0.2)
        plt.axvspan(self.fminCalculation / 1e12, self.fmaxCalculation / 1e12,
                    color='b', alpha=0.1)
        plt.xlabel('Frequency (THz)')
        plt.ylabel('unwrapped phase (rad)')
        plt.tight_layout()

    def plotTransferFunction(self):
        plt.figure()
        ax1 = plt.gca()
        ax1.plot(self.freqs / 1e12, self.Habs, linewidth=2)
        plt.axvspan(self.fminCalculation / 1e12, self.fmaxCalculation / 1e12,
                    color='b', alpha=0.1)
        ax2 = plt.twinx(plt.gca())
        ax2.plot(self.freqs / 1e12, self.Hphase, linewidth=2, color='g')
        ax2.hlines([0], self.freqs[0] / 1e12, self.freqs[-1] /1e12,
                   linestyle=':', color='g', linewidth=2)
        ax2.tick_params(axis='y', colors='g')
        ax2.set_ylabel(r'$\angle H$', color='g')
        ax1.set_ylabel(r'$|H|$')
        ax1.set_xlabel(r'Frequenz (THz)')
        plt.tight_layout()
        
    def plotRefractiveIndex(self):
        plt.figure()
        plt.subplot(211)
        plt.plot(self.fr/1e12,self.n.real,'k', label=self.sn)
        plt.xlabel('Frequenz (THz)')
        plt.ylabel('Refractive index n')
        plt.legend(loc='best')
        plt.subplot(212)
        plt.plot(self.fr/1e12,self.n.imag,'k', label=self.sn)
        plt.xlabel('Frequenz (THz)')
        plt.ylabel(r'Absorption $\kappa$')
        plt.legend(loc='best')
        plt.tight_layout()
    
class OneLayerSystem(TDS):
    def __init__(self, dmeasured = 1e-3):
        super().__init__()
        self.dmeasured = dmeasured
        
    def getnoPulses(self, n, d):
            tetalon = n * d * 2 / c
            return int((self.taxis[-1] - self.taxis[np.argmax(self.sample)]) / tetalon)

    def estimateNoEchos(self):
        
        # Calculate n and l via Etalon Frequency and phaseSlope
        phaseSlope = abs(self.avg_phaseslope)
        # find the minima of the etalon
        fEtalon = np.mean(np.diff(self.fr[argrelmin(self.Har)]))
        n_e = 1.0 / (1 - phaseSlope * fEtalon / np.pi)
        l_s = c / 2.0 * (1 / fEtalon - phaseSlope / np.pi) * 1e6

        print('{:2.2f} Average Refractive index from Phase and Etalon'.format(n_e))

        # Calculate it with the measured thickness and via the reference sample
        # delay
        deltat = self.taxis[np.argmax(self.sample)] - self.taxis[np.argmax(self.reference)]
        n_avg = deltat * c / self.dmeasured + 1

        noPulses = self.getnoPulses(n_avg, self.dmeasured)

        print('{:2.2f}: Average Refractive index from Reference Sample Peak Delay'.format(
            n_avg))
        print('{:2.2f} µm Estimated Thickness from Phase and Etalon'.format(l_s))
                           
        print('{:2.0f} Number of Pulses in time window (using measured thickness)'.format(noPulses))
        print('{:2.0f} Number of Pulses in time window (using estimated thickness)'.format(self.getnoPulses(n_e, l_s * 1e-6)))
        self.noPulses = noPulses
        return noPulses
    
    
    def calculateNApproximate(self, freq, Habs, Hphase, d):
        n0 = 1
        omega = 2 * np.pi * freq
        n = n0 - c / (omega * d) * Hphase
        if isinstance(Habs[0], float):
            kappa = c / (omega * d) * \
                        (np.log(4 * n * n0 / (n + n0)**2) - np.log(Habs))
        else:
            kappa = c / (omega * d)
            kappa *= (unumpy.log(4 * n * n0 / (n + n0)**2) - unumpy.log(Habs))
        return n, -kappa
    
    def Ht(self, omega, n, d, noPulses):
        # try the full formula
        n0 = 1
        q = ((n0 - n) / (n0 + n) * np.exp(-1j * n * omega * d / c))**2
        FP = (q**(noPulses + 1) - 1) / (q - 1)
        FP *= 4 * n0 * n/(n0 + n)**2 * np.exp(-1j * (n - n0) * omega * d / c)
        return  FP


    def minme(self, n, omega, Hmeasure, d, noPulses):
        n = n[:int(len(n) / 2)] + 1j * n[int(len(n) / 2):]
        t = self.Ht(omega, n, d, noPulses)
        return np.sum((Hmeasure.real - t.real)**2 + (Hmeasure.imag - t.imag)**2)

    def calculateN(self, d = None):
        if d is None:
            d = self.dmeasured
            
        Hmeasure = self.Har * np.exp(1j * self.Hpr)
        omega = 2 * np.pi * self.fr
    
        n, kappa = self.calculateNApproximate(self.fr, self.Har, self.Hpr, d)
        ninit = np.hstack((n, kappa))
    
        # set Bounds non n and kappa
        nbound = [1, 20]
        kappabound = [-40, 0]
        bounds = np.vstack((np.ones((len(n), 2)) * nbound,
                            np.ones((len(n), 2)) * kappabound))
    
        fres = minimize(self.minme, ninit,args=(omega, Hmeasure, d, self.noPulses), bounds=bounds)
    
        ntotal = fres.x
        nt = ntotal[:int(len(ntotal) / 2)]
        nk = ntotal[int(len(ntotal) / 2):]
        self.n = nt + 1j *nk
        self.dcalculated = d
        return self.n, fres

    def calculateBestThickness(self, dstep = 5e-6, dinterval=30e-6,
                               method='TV', freqstep = 6, doPlot = False):
        # if equal 1 we do the optimization for all frequencies,
        # else skip freqstep frequencies
                
        self.fr = self.fr[::freqstep]
        self.Har = self.Har[::freqstep]
        self.Hpr = self.Hpr[::freqstep]
        
        
        ds = np.arange(self.dmeasured-dinterval,self.dmeasured+dinterval,dstep)
        
        tvs = []
        
        i = 0
        if doPlot: plt.figure()         
        for d in ds:
            ntt = self.calculateN(d)
            tvs.append(np.sum(np.abs(np.diff(ntt[0].real)) + np.abs(np.diff(ntt[0].imag))))
            if i%10 == 0: print('{:2.1f}µm done, {}/{}'.format(d*1e6,i,len(ds)))
            i+=1
            
            if doPlot:
                plt.subplot(211)
                plt.plot(self.fr/1e12,ntt[0].real,color=plt.cm.viridis(i/(len(ds)+1)))
                plt.xlabel('Frequenz (THz)')
                plt.ylabel('Refractive index n')
                
                plt.subplot(212)
                plt.plot(self.fr/1e12,ntt[0].imag,color=plt.cm.viridis(i/(len(ds)+1)))
                plt.xlabel('Frequenz (THz)')
                plt.ylabel(r'Absorption $\kappa$')

        tvs = np.array(tvs)
    
        self.changeCalculationDomain(self.fminCalculation, self.fmaxCalculation)

        dii = np.linspace(ds[0],ds[-1],len(ds)*20)
        pp = interp1d(ds, tvs,kind='cubic')
        self.dopt = dii[np.argmin(pp(dii))]
        print('Best Thickness: {:2.2f} µm'.format(self.dopt*1e6))
        
        return self.dopt, ds, tvs

    def plotTotalVariation(self, ds, tvs):
        dii = np.linspace(ds[0],ds[-1],len(ds)*20)
        pp = interp1d(ds, tvs,kind='cubic')
        
        plt.figure()
        plt.plot(ds*1e6,tvs,'k+')
        plt.plot(dii*1e6,pp(dii))
        tvv = np.amax(tvs)-np.amin(tvs)
        plt.vlines([self.dopt*1e6],np.amin(tvs)-0.1*tvv,np.amax(tvs),linestyles=':',linewidth=2)
        plt.annotate(xy=(ds[2]*1e6,np.amin(tvs)-0.1*tvv),s='Best Thickness: {:2.2f} µm'.format(self.dopt*1e6))
        plt.xlabel(r'Thickness d ($\mu$m)')
        plt.ylabel('Total Variation Value')
        plt.tight_layout()
    
    def plotRefractiveIndex(self, includeApproximation = False):
        super().plotRefractiveIndex()
        if includeApproximation:
            n, kappa = self.calculateNApproximate(self.fr, self.Har, self.Hpr, self.dcalculated)    
            f = plt.gcf()
            f.axes[0].plot(self.fr/1e12,n,'r', label='Approximate')
            f.axes[1].plot(self.fr/1e12,kappa,'r', label='Approximate')


class ThreeLayerSystem(TDS):
    
    def __init__(self, n1,n3,d1, d3, dmeasured):
        super().__init__()
        self.dmeasured = dmeasured
        self.n1 = n1
        self.n3 = n3
        self.d1 = d1
        self.d3 = d3
    
    def calculatetmax(self):
        n1 = np.mean(self.n1.real)
        n3 = np.mean(self.n3.real)
        deltat = (self.taxis[np.argmax(self.sample)]-self.taxis[np.argmax(self.reference)])
        
        opticalThickness = c*deltat-(n1-1)*self.d1-(n3-1)*self.d3
        n2_avg  = opticalThickness/self.dmeasured+1
        tmax = self.taxis[-1]-self.taxis[np.argmax(self.sample)]
        
        return n2_avg, tmax

    
    def calculateCoefficient(l,f,ns,ds,n_medium):
        #this might be easier done
        tt  = t(n_medium,ns[0])*t(ns[-1],n_medium)
        indexes = getIndex(l)
        diffs = np.diff(indexes)
        k=2*np.pi*f/c
        tt*=np.exp(-1j*np.sum(ns[indexes]*ds[indexes], axis=0)*k)
        tt*=np.exp(1j*np.sum(ds, axis=0)*n_medium*k)
        for i in range(len(diffs)):
            if diffs[i] == 1: #go left right
                tt*=t(ns[indexes[i]],ns[indexes[i]+1])
                #print('tlr')
            elif diffs[i] == -1: #go right left
                tt*=t(ns[indexes[i]],ns[indexes[i]-1])
                #print('trl')
            elif diffs[i] == 0 and l[i] > 0:
                if len(ns) == indexes[i]+1:
                    #we are at the outermost boundary
                    no = n_medium
                else:
                    no = ns[indexes[i]+1]
                tt*= r(ns[indexes[i]],no)
                #print('rr')
    
            elif diffs[i] == 0 and l[i]<0:
                if indexes[i] == 0:
                    #we are at the innermost boundary
                    no = n_medium
                else:
                    no = ns[indexes[i]-1]
                tt*= r(ns[indexes[i]],no)
                #print('rl')
        return tt
    
    def getH(ns,ds,fs,n_medium,tmax):
        l = calculatePossibleWays(ns,ds,tmax)
        ps = 0
        for ll in l:
            ps+=calculateCoefficient(ll,fs,n_medium)
        return ps

def calculatePossibleWays(ns,ds,tmax=100e-12):
    # some checks
    nLayers = len(ns)
    if len(ds) is not len(ns):
        print('thickness and refractive index must be given for each layer')
        return []
    if tmax < np.sum(ns*ds/c):
        print('tmax too short, direct time through sample is {:2.0f} ps'.format(np.sum(ns*ds/c)*1e12))
        return []
    #if tmax > np.sum(ns*ds/c)*4:
    #    print('I think too many ways have been included, used tmax = {:2.0f} ps instead'.format(np.sum(ns*ds/c)*1e12*4))
    #    tmax = np.sum(ns*ds/c)*4
     
    def hitInterface(i,path):
        if getTime(path,ns,ds) > tmax:
            #print('giving up this path, traveltime: {:2.2f}'.format(getTime(path,ns,ds)*1e12))
            return 0
        if i==0: #we are at the first interface
            path.append(1)
            hitInterface(i+1,path)
        elif i==nLayers: #we are at the last interface
            l.append(np.copy(path))
            path.append(-1)
            hitInterface(i-1,path)
        else: #we are in the medium
            p1 = path + [1]
            p2 = path + [-1]
            hitInterface(i+1,p1)
            hitInterface(i-1,p2)
        return 0
    l = [] #stores where to got 
    hitInterface(1,[1])
    return l

def getIndex(l):
    if len(l)==1:
        return np.array([0])
    else:
        return np.array(np.convolve(np.cumsum(l),np.ones((2,))*0.5,'same'),dtype=int)

def getTime(l,ns,ds):
    layers=getIndex(l)
    return np.sum(ns[layers]*ds[layers])/c

def t(n1,n2):
    return 2*n1/(n1+n2)

def r(n1,n2):
    return (n1-n2)/(n1+n2)

def cropFrequencyData(freqs, data, fmin, fmax):
    ix = np.all(np.vstack((freqs > fmin, freqs < fmax)), axis=0)
    fr = freqs[ix]
    data = data[ix]
    return fr, data


def cropTimeData(taxis, data, tmax):
    if tmax is not None:
        ix = taxis < tmax
        taxis = taxis[ix]
        data = data[ix]
    return taxis, data


def loadfiles(fns):
    if len(fns) == 0:
        print('No filename given')
        return

    data = np.loadtxt(fns[0])
    taxis = data[:, 0]
    d = np.zeros((len(taxis), len(fns)))
    d[:, 0] = data[:, 1]
    for i in range(1, len(fns)):
        d[:, i] = np.loadtxt(fns[i])[:, 1]

    return taxis, np.mean(d, axis=1), np.var(d, axis=1)


def phaseOffsetRemoval(freq, phase, fmin=0.5, fmax=2):
    # fmin = 0.5 # frequency domain for interpolation
    #fmax = 2

    # extract slope and offset in range fmin,fmax
    ix = np.all(np.vstack((freq > fmin, freq < fmax)), axis=0)

    # reduce for fit
    fred = freq[ix]
    phred = phase[ix]

    a = np.polyfit(fred, phred, 1)

    return phase - a[1], a


def calculateSpectrum(taxis, data, fmax=0, slopeTime=5e-12, fbins=None):
    # slopeTime = 5 #rising and falling time in ps
    # fbins = None #target frequency resolution in GHz; Set to None if no zeropadding wanted
    # tmax = None #set time limit;

    dt = np.mean(np.diff(taxis))

    # apply window function
    N = int(slopeTime / dt)

    window = windows.blackman(2 * N)
    window = np.hstack((window[:N], np.ones(len(taxis) - 2 * N,), window[N:]))

    data *= window

    # fourier transform to target frequency resolution by zeropadding
    if fbins is not None:
        N = int(1 / dt / fbins )
    else:
        N = len(taxis)

    fd = np.fft.rfft(data, N)
    freq = np.fft.rfftfreq(N, dt)
    phase = np.unwrap(np.angle(fd))

    return freq[freq < fmax], fd[freq < fmax], phase[freq < fmax]

if __name__ == '__main__':
    import glob
    
# %% Three Layer System Example
    refn = glob.glob('Oil/Emitter glass/M1/*Reference*')
    samn = glob.glob('Oil/Emitter glass/M1/*Cuv*')

    glass1 = OneLayerSystem(710e-6)
    glass1.loadData(refn, samn)
    glass1.cropTimeData(200e-12)
    glass1.fbins = 10e9 
    glass1.calculateTransferFunction()
    glass1.estimateNoEchos()
    #dopt, ds, tvs = glass1.calculateBestThickness()
    glass1.calculateN(707.4e-6)

    refn = glob.glob('Oil/Receiver glass/M1/*Reference*')
    samn = glob.glob('Oil/Receiver glass/M1/*Cuv*')

    glass2 = OneLayerSystem(710e-6)
    glass2.loadData(refn, samn)
    glass2.cropTimeData(200e-12)
    glass2.fbins = 10e9 
    glass2.calculateTransferFunction()
    glass2.estimateNoEchos()
    #dopt, ds, tvs = glass2.calculateBestThickness()
    glass2.calculateN(710.6e-6)

