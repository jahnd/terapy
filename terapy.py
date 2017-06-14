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
from scipy.stats import t as studentt
from uncertainties import unumpy


class TDS:

    def __init__(self,sampleName = None):
        self.FMAX = 6e12
        self.fminCalculation = 0.3e12
        self.fmaxCalculation = 2e12

        self.fminPhase = 0.4e12
        self.fmaxPhase = 2e12
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
        #not so nice at the moment:
        self.noDatas = min(len(fns_ref),len(fns_sam))
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

    def calculateUncertaintyH(self):
        #propagate variance to spectrum
        fvars = np.sum(self.var_s)
        fvarr = np.sum(self.var_r)

        ufdsam_r = unumpy.uarray(self.fdsam.real, fvars**0.5)
        ufdsam_i = unumpy.uarray(self.fdsam.imag, fvars**0.5)
        ufdref_r = unumpy.uarray(self.fdref.real, fvarr**0.5)
        ufdref_i = unumpy.uarray(self.fdref.imag, fvarr**0.5)

        #calculate angle and phase uncertainty
        uphase_ref = unumpy.arctan2(ufdref_i, ufdref_r)
        uphase_sam = unumpy.arctan2(ufdsam_i, ufdsam_r)

        uphase_sam_stddevs = unumpy.std_devs(uphase_sam)
        uphase_ref_stddevs = unumpy.std_devs(uphase_ref)

        # do the phase interpolation only considering the nominal values
        phasesamC, b = phaseOffsetRemoval(self.freqs, np.unwrap(
            unumpy.nominal_values(uphase_sam)), self.fminPhase, self.fmaxPhase)
        phaserefC, a = phaseOffsetRemoval(self.freqs, np.unwrap(
            unumpy.nominal_values(uphase_ref)), self.fminPhase, self.fmaxPhase)

        uphase_ref = unumpy.uarray(phaserefC, uphase_ref_stddevs)
        uphase_sam = unumpy.uarray(phasesamC, uphase_sam_stddevs)

        ufabss = (ufdsam_r**2 + ufdsam_i**2)**0.5
        ufabsr = (ufdref_r**2 + ufdref_i**2)**0.5
        uHabs = ufabss / ufabsr
        uHphase = uphase_sam - uphase_ref

        #calculate real and imaginary uncertainty
        ufdsam_r = unumpy.uarray(self.fdsam.real, fvars**0.5)
        ufdsam_i = unumpy.uarray(self.fdsam.imag, fvars**0.5)
        ufdref_r = unumpy.uarray(self.fdref.real, fvarr**0.5)
        ufdref_i = unumpy.uarray(self.fdref.imag, fvarr**0.5)

        Hr = (ufdsam_r * ufdref_r + ufdsam_i * ufdref_i) / \
            (ufdref_r**2 + ufdref_i**2)
        Hi = (ufdsam_i * ufdref_r - ufdsam_r * ufdref_i) / \
            (ufdref_r**2 + ufdref_i**2)

        self.uHabs = uHabs
        self.uHphase = uHphase
        self.uHreal = Hr
        self.uHimag = Hi
        return uHabs, uHphase, Hr, Hi
    
    def plotTimeDomainData(self):
        plt.figure()
        plt.plot(self.taxis * 1e12, self.sample, 'k',label='Sample')
        plt.plot(self.taxis * 1e12, self.reference, 'r',label='Reference')
        plt.xlabel('Time (ps)')
        plt.ylabel('Amplitude')
        plt.legend()
        
    def plotTimeDomainDataUnc(self):
        self.plotTimeDomainData()
        
        refi = studentt.interval(0.95,self.noDatas-1, loc=self.reference, scale=self.var_r**0.5/self.noDatas**0.5)
        sami = studentt.interval(0.95,self.noDatas-1, loc=self.sample, scale=self.var_s**0.5/self.noDatas**0.5)
        plt.fill_between(self.taxis*1e12,self.reference-self.var_r**0.5,self.reference+self.var_r**0.5,color='r',label=r'1$\sigma$',alpha=0.2)
        plt.fill_between(self.taxis*1e12,self.sample-self.var_s**0.5,self.sample+self.var_s**0.5,color='k',alpha=0.2)
        #plt.errorbar(self.taxis*1e12,self.sample, yerr=self.var_s**0.5,color='r',label='Reference')
        plt.plot(self.taxis*1e12,refi[0],'r--',label='95 % confidence')
        plt.plot(self.taxis*1e12,refi[1],'r--')
        plt.plot(self.taxis*1e12,sami[0],'k--')
        plt.plot(self.taxis*1e12,sami[1],'k--')
        plt.legend()

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
        ax1.plot(self.freqs / 1e12, self.Habs, 'k', linewidth=2)
        plt.axvspan(self.fminCalculation / 1e12, self.fmaxCalculation / 1e12,
                    color='b', alpha=0.1)
        ax2 = plt.twinx(plt.gca())
        ax2.plot(self.freqs / 1e12, self.Hphase, linewidth=2, color='r')
        ax2.hlines([0], self.freqs[0] / 1e12, self.freqs[-1] /1e12,
                   linestyle=':', color='r', linewidth=2)
        ax2.tick_params(axis='y', colors='r')
        ax2.set_ylabel(r'$\angle H$', color='r')
        ax1.set_ylabel(r'$|H|$')
        ax1.set_xlabel(r'Frequenz (THz)')
        plt.tight_layout()
    
    def plotTransferFunctionUnc(self):
        self.plotTransferFunction()
        ax1, ax2 = plt.gcf().axes
        lo = unumpy.nominal_values(self.uHabs) - unumpy.std_devs(self.uHabs)
        hi = unumpy.nominal_values(self.uHabs) + unumpy.std_devs(self.uHabs)
        ax1.fill_between(self.freqs / 1e12, lo, hi,color='k', alpha=0.2 )
        lo = unumpy.nominal_values(self.uHphase) - unumpy.std_devs(self.uHphase)
        hi = unumpy.nominal_values(self.uHphase) + unumpy.std_devs(self.uHphase)
        ax2.fill_between(self.freqs / 1e12, lo, hi , color='r', alpha=0.2)
        plt.tight_layout()
    
    def plotTransferFuncRealImagUnc(self, f=2):
        plt.figure()
        lo = unumpy.nominal_values(self.uHreal) - f * unumpy.std_devs(self.uHreal)
        hi = unumpy.nominal_values(self.uHreal) + f * unumpy.std_devs(self.uHreal)
        plt.fill_between(self.freqs / 1e12, lo, hi,color='r', alpha=0.2 )
        plt.plot(self.freqs/1e12, unumpy.nominal_values(self.uHreal),'k')
        lo = unumpy.nominal_values(self.uHimag) - f * unumpy.std_devs(self.uHimag)
        hi = unumpy.nominal_values(self.uHimag) + f * unumpy.std_devs(self.uHimag)
        plt.fill_between(self.freqs / 1e12, lo, hi,color='r', alpha=0.2 )
        plt.plot(self.freqs/1e12, unumpy.nominal_values(self.uHimag),'k')
        plt.xlim(self.fminCalculation/1e12, self.fmaxCalculation/1e12)
        plt.ylim(-1,1)
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
    
    Jac = None
    
    def __init__(self, dmeasured = 1e-3):
        super().__init__()
        self.dmeasured = dmeasured
        self.Jac = None
        
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
            kappa = c/(omega*d)*(unumpy.log(4*n*n0/(n + n0)**2) - unumpy.log(Habs))
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

        dii = np.arange(ds[0],ds[-2],0.25e-6)
        pp = interp1d(ds, tvs,kind='cubic')
        self.dopt = dii[np.argmin(pp(dii))]
        print('Best Thickness: {:2.2f} µm'.format(self.dopt*1e6))
        
        return self.dopt, ds, tvs
    
    def getJac(self, omegaa, na, da, noPulses):
        if OneLayerSystem.Jac is None:
            # calculate derivative analytically
            import sympy
            n1, n, kappa, d, k0, M = sympy.symbols('n1 n kappa d k0 M')
            expr1 = ((n1 - n - 1j * kappa) / (n1 + n + 1j * kappa)
                     * sympy.exp(-1j * k0 * d * (n + 1j * kappa)))**2
            expr = 4 * n1 * (n + 1j * kappa) / (n1 + n + 1j * kappa)**2 * sympy.exp(-1j * \
                             k0 * d * (n + 1j * kappa - n1)) * (expr1**(M + 1) - 1) / (expr1 - 1)
    
            diff = sympy.diff(expr, n)
            df = sympy.lambdify((n1, n, kappa, d, k0, M), diff, "numpy")
            OneLayerSystem.Jac = df
        return OneLayerSystem.Jac(1, na.real, na.imag, da, omegaa / c, noPulses)
    
    def calculateUncertaintyOpticalConstants(self):
        fr, Hr = cropFrequencyData(self.freqs, self.uHreal, 
                                   self.fminCalculation, self.fmaxCalculation)
        fr, Hi = cropFrequencyData(self.freqs, self.uHimag, 
                                   self.fminCalculation, self.fmaxCalculation)

        a = self.getJac(2 * np.pi * fr, self.n, self.dcalculated, self.noPulses)
        a1 = 1 / np.abs(a)**2 * a

        self.stdn = (a1.real**2 * unumpy.std_devs(Hr)**2 +
                a1.imag**2 * unumpy.std_devs(Hi)**2)**0.5
        self.stdk = (a1.imag**2 * unumpy.std_devs(Hr)**2 +
                a1.real**2 * unumpy.std_devs(Hi)**2)**0.5
        return self.stdn, self.stdk    

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
    
    def plotRefractiveIndexUnc(self, includeApproximation = False):
        self.plotRefractiveIndex(includeApproximation)
        ax1, ax2 = plt.gcf().axes
        ax1.fill_between(self.fr/1e12, self.n.real - self.stdn, self.n.real + self.stdn,color='r', alpha=0.2)
        ax2.fill_between(self.fr/1e12, self.n.imag - self.stdk, self.n.imag + self.stdk,color='r', alpha=0.2)
        ni = studentt.interval(0.95,self.noDatas-1, loc=self.n.real, scale = self.stdn/self.noDatas**0.5)
        ki = studentt.interval(0.95,self.noDatas-1, loc=self.n.imag, scale = self.stdk/self.noDatas**0.5)
        ax1.plot(self.fr/1e12,ni[0],'k--',label='95 % confidence')
        ax1.plot(self.fr/1e12,ni[1],'k--')
        ax2.plot(self.fr/1e12,ki[0],'k--')
        ax2.plot(self.fr/1e12,ki[1],'k--')
        if includeApproximation:
            fr, uHabs = cropFrequencyData(self.freqs, self.uHabs, self.fminCalculation, self.fmaxCalculation)
            fr, uHphase = cropFrequencyData(self.freqs, self.uHphase, self.fminCalculation, self.fmaxCalculation)
            un, uk = self.calculateNApproximate(fr, uHabs, uHphase, self.dcalculated)
            lo = unumpy.nominal_values(un) - unumpy.std_devs(un)
            hi = unumpy.nominal_values(un) + unumpy.std_devs(un)
            ax1.fill_between(fr/1e12, lo, hi ,color='r', alpha=0.2)
            lo = unumpy.nominal_values(uk) - unumpy.std_devs(uk)
            hi = unumpy.nominal_values(uk) + unumpy.std_devs(uk)
            ax2.fill_between(fr/1e12, lo, hi ,color='r', alpha=0.2)
    
    def plotAlpha(self):
        
        ni = unumpy.uarray(self.n.imag, self.stdk)
        alpha = -ni * 2 * (2*np.pi*self.fr)/c*1e-2
        itk = studentt.interval(0.95,self.noDatas-1, loc=unumpy.nominal_values(alpha),
                         scale=unumpy.std_devs(alpha)/self.noDatas**0.5)
        plt.figure()
        plt.plot(self.fr/1e12, unumpy.nominal_values(alpha),'k',label=r'$\alpha$')
        lo = unumpy.nominal_values(alpha) - unumpy.std_devs(alpha)
        hi = unumpy.nominal_values(alpha) + unumpy.std_devs(alpha)
        plt.fill_between(self.fr/1e12, lo, hi, color = 'r', alpha=0.2, label=r'1$\sigma$')
        
        plt.plot(self.fr/1e12,itk[0],'k--',label='95 % Confidence')
        plt.plot(self.fr/1e12,itk[1],'k--')
        plt.xlabel('Frequenz (THz)')
        plt.ylabel(r'Absorption $\alpha$ (cm$^{-1}$)')
        plt.legend()
        plt.tight_layout()
    
    def SVMAF(self,interval=2):
        fr, uHr = cropFrequencyData(self.freqs, self.uHreal, self.fminCalculation, self.fmaxCalculation)
        fr, uHi = cropFrequencyData(self.freqs, self.uHimag, self.fminCalculation, self.fmaxCalculation)
        
        #Apply the SVMAF filter to the material parameters
        #runningMean=lambda x,N: py.hstack((x[:N-1],py.cvolve(x,py.ones((N,))/N,mode='valid')[(N-1):],x[(-N+1):]))
        runningMean=lambda x,N: np.hstack((x[:N-1],np.convolve(x,np.ones((N,))/N,mode='same')[N-1:-N+1],x[(-N+1):]))
       
        n_smoothed=runningMean(self.n,3) # no problem in doing real and imaginary part toether
        
        H_smoothed=self.Ht(2*np.pi*self.fr,n_smoothed,self.dcalculated,self.noPulses)
        
        H_r=H_smoothed.real
        H_i=H_smoothed.imag
        
        lb_r=unumpy.nominal_values(uHr) - interval*unumpy.std_devs(uHr)
        ub_r=unumpy.nominal_values(uHr) + interval*unumpy.std_devs(uHr)
        lb_i=unumpy.nominal_values(uHi) - interval*unumpy.std_devs(uHi)
        ub_i=unumpy.nominal_values(uHi) + interval*unumpy.std_devs(uHi)
        #ix=all indices for which after smoothening n H is still inbetwen the bounds        
        ix=np.all([H_r>=lb_r,H_r<ub_r,H_i>=lb_i,H_i<ub_i],axis=0)
        n_smoothed[np.logical_not(ix)] = self.n[np.logical_not(ix)]
#        #dont have a goood idea at the moment, so manually:
        print("SVMAF changed the refractive index at " + str(sum(ix)) + " frequencies")
        return n_smoothed      
    
    def applySVMAF(self, Niteration = 5, aggresivity = 3):
        for i in range(Niteration):
            self.n = self.SVMAF(3)

class ThreeLayerSystem(TDS):
    
    def __init__(self, n1,n3,d1, d3, dmeasured):
        super().__init__()
        self.dmeasured = dmeasured
        self.n1 = n1
        self.n3 = n3
        self.d1 = d1
        self.d3 = d3

    def calculateWays(self):
        n1 = np.mean(self.n1.real)
        n3 = np.mean(self.n3.real)
        deltat = (self.taxis[np.argmax(self.sample)]-self.taxis[np.argmax(self.reference)])
        
        opticalThickness = c*deltat-(n1-1)*self.d1-(n3-1)*self.d3
        n2_avg  = opticalThickness/self.dmeasured+1
        tmax = self.taxis[-1]-self.taxis[np.argmax(self.sample)]
        self.ways = calculatePossibleWays(np.array([n1,n2_avg,n3]),
                              np.array([self.d1,self.dmeasured,self.d3]),tmax)
        return self.ways

    def Ht(self, omega, ns, ds, ways):
        ps = 0
        for ll in ways:
            ps += calculateCoefficient(omega,ns,ds,1,ll)
        return ps
    
    def calculateBestThickness(self, nFilling = 1, dstep = 5e-6, dinterval=30e-6,
                               method='BruteForce', doPlot = False):
        
        self.calculateWays()
        ns = np.asarray([self.n1,np.ones(self.n1.shape,dtype=np.complex64)*nFilling,self.n3])
        ds = np.asarray([[self.d1], [self.dmeasured], [self.d3]])*np.ones(ns.shape)
        
        dd = np.arange(self.dmeasured-dinterval,self.dmeasured+dinterval, dstep)
        
        Hm = self.Har * np.exp(1j*self.Hpr)
        val = []
        
        for d in dd:
            ds[1] = d
            Ht = self.Ht(2*np.pi*self.fr, ns, ds,self.ways)
            val.append(np.sum((Hm.real-Ht.real)**2 + (Hm.imag-Ht.imag)**2))
        
        val = np.array(val)
        dii = np.arange(dd[0],dd[-1],0.25e-6)
        pp = interp1d(dd, val, kind='cubic')
        self.dopt = dii[np.argmin(pp(dii))]
        if doPlot:
            plt.figure()
            plt.plot(dd*1e6, val, 'ks', markerfacecolor='none')
            plt.plot(dii*1e6, pp(dii))
            plt.vlines([self.dopt*1e6],np.amin(val)-0.1*val,np.amax(val),linestyles=':',linewidth=2)
            plt.text(dd[2]*1e6,np.amin(val),s='Best Thickness: {:2.2f} µm'.format(self.dopt*1e6))
            plt.xlabel(r'Thickness d ($\mu$m)')
            plt.ylabel('Deviation from Measurement')
            plt.tight_layout()
        return self.dopt
        
    
def calculateCoefficient(omega,ns,ds,n_medium,l):
    #this might be easier done
    tt  = t(n_medium,ns[0])*t(ns[-1],n_medium)
    indexes = getIndex(l)
    diffs = np.diff(indexes)
    
    tt*=np.exp(-1j*np.sum(ns[indexes]*ds[indexes], axis=0)*omega/c)
    tt*=np.exp(1j*np.sum(ds, axis=0)*n_medium*omega/c)
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

    return taxis, np.mean(d, axis=1), np.var(d, axis=1, ddof=1)


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
    #glass1.cropTimeData(210e-12)
    glass1.fbins = 10e9 
    glass1.calculateTransferFunction()
    glass1.estimateNoEchos()
    dopt, ds, tvs = glass1.calculateBestThickness()
#    glass1.plotTotalVariation(ds,tvs)
    #glass1.dopt = 707.4e-6
    glass1.calculateN(707.4e-6)
#    glass1.calculateUncertaintyH()
#    glass1.calculateUncertaintyOpticalConstants()
    
#    glass1.plotRefractiveIndexUnc()
#    glass1.plotAlpha()
    
    #glass1.applySVMAF(20)
    #glass1.plotRefractiveIndexUnc()
    #glass1.plotAlpha()
    # %%
   
    refn = glob.glob('Oil/Receiver glass/M1/*Reference*')
    samn = glob.glob('Oil/Receiver glass/M1/*Cuv*')
#
    glass2 = OneLayerSystem(710e-6)
    glass2.loadData(refn, samn)
     #glass2.cropTimeData(210e-12)
    glass2.fbins = 10e9 
    glass2.calculateTransferFunction()
    glass2.estimateNoEchos()
    #dopt, ds, tvs = glass2.calculateBestThickness()
    glass2.dopt = 710.6e-6
    glass2.calculateN(glass2.dopt)
#    
#    
#%%

#%%
    emptyCuvette = ThreeLayerSystem(glass1.n, glass2.n, glass1.dopt, glass2.dopt, 5900e-6)
    refn = glob.glob('Oil/Cuvetttes/0EmptyNewCuvettes/M1/*Reference*')
    samn = glob.glob('Oil/Cuvetttes/0EmptyNewCuvettes/M1/*Cuvette1*')
    emptyCuvette.loadData(refn, samn)
    
    #Cuvette.cropTimeData(210e-12)
    emptyCuvette.fbins = 10e9
    #Cuvette.plotTimeDomainData()
    emptyCuvette.calculateTransferFunction()
    dopt = emptyCuvette.calculateBestThickness(dstep=7.5e-6, dinterval=300e-6, doPlot=True)
#    
#
