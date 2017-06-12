#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 14:53:54 2017

@author: jahndav
"""
from terapy import *
import glob

#show all options for input
refn = glob.glob('Oil/Emitter glass/M1/*Reference*')
samn = glob.glob('Oil/Emitter glass/M1/*Cuv*')

glass1 = OneLayerSystem(710e-6)
# load files
glass1.loadData(refn, samn)

#plot raw data
#glass1.plotTimeDomainData()

#crop data to desired range
glass1.cropTimeData(200e-12)
glass1.plotTimeDomainData()

#calculate frequency domain data with some frequency resolution via zeropadding
glass1.fbins = 10e9 
glass1.calculateTransferFunction()
glass1.plotFrequencyDomainData()
glass1.plotTransferFunction()

#change phase interpolation domain
#glass1.changePhaseInterpolationDomain(fmin,fmax)
#change calculation domain
#glass1.changeCalculationDomain(0.3e12,0.8e12)
#verify changes
#glass1.plotFrequencyDomainData()

glass1.estimateNoEchos()    
dopt, ds, tvs = glass1.calculateBestThickness()
glass1.plotTotalVariation(ds,tvs)

glass1.calculateN(glass1.dopt)
glass1.plotRefractiveIndex(includeApproximation=True)
glass1.calculateUncertaintyH()
glass1.calculateUncertaintyOpticalConstants()
glass1.plotRefractiveIndexUnc()
glass1.plotAlpha()