#!/usr/bin/env python3
import os
import numpy as np
from scipy.optimize import curve_fit
import json
import matplotlib.pyplot as plt
import pyhf
from pyhf.contrib.viz import brazil



def Expo (x, N, L):
  y = N * np.exp(-1*L*x)
  return y

def Polynomial2 (x, a, b, c):
  y = a*x*x + b*x + c
  return y

def Polynomial3 (x, a, b, c, d):
  y = a*x*x*x + b*x*x + c*x +d 
  return y

def Gaussian(x, mean, width, constant):
  y = abs(constant)* np.exp(-1*((x-abs(mean))**2)/(2*(width**2)))
  return y

# set these if you want to force your bkg to stay fixed
fixedExpoN = 999
fixedExpoL = 999

def GausPlusFixedExpo(x, mean, width, constant):

  return Gaussian(x, mean, width, constant) + Expo (x, fixedExpoN, fixedExpoL)

startingValues = {
Expo: [700, 0.001],
Gaussian: [800, 30, 1000],
Polynomial2: [0.5, 0.5, 0.5],
Polynomial3: [0.5, 0.5, 0.5, 0.5],
GausPlusFixedExpo :[ 800, 30, 500],

}




#params, errs = curve_fit(Expo, bin_centers, bin_heights, p0=[mean, width, constant, N, L], bounds= ([300, 0, 0, 0, 0  ], [500, 100, np.inf, np.inf, np.inf ]), maxfev=50000)

allfuncs =[]

def do_fit(function, bin_edges, bin_heights, uncertainties=None):
  bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2
  
  # remove zero bins
  new_bin_centers = []
  new_bin_heights = []
  for i, bh in enumerate(bin_heights):
    if bh:
      new_bin_centers+=[bin_centers[i]]
      new_bin_heights+=[bin_heights[i]]
  params, errs = curve_fit(function,  new_bin_centers, new_bin_heights, maxfev=50000, p0 =startingValues[function],  sigma=np.array(new_bin_heights)**0.5 , absolute_sigma=True)
  return bin_centers, params,  np.sqrt(np.diag(errs))

def integral(func, x, xRange, params):
    global allfuncs
    bw = abs(x[1]-x[0])
    bw=1
    y = func(x, *params)
    allfuncs+=[y]
    x = x[ np.logical_and(x > xRange[0], x < xRange[1])]
    y = func(x, *params)
    return (y*bw).sum()


def integral_with_errors(func, x, xRange, params, errs):  
 aucs = []
 for i in range (1000):
    theseParams = []
    for ip, p in enumerate(params):
      theseParams += [ p + np.random.normal(0, errs[ip]) ]
    aucs += [ integral(func, x, xRange, theseParams) ]
 aucs = np.array(aucs)
 return aucs.mean(), aucs.std()


def cls_limit_calculator(sig_pred, bkg_pred, bkg_pred_errs, obs_data, plot_dir="."):
  os.system("mkdir -p %s" % plot_dir) 

  model = pyhf.simplemodels.uncorrelated_background(
      signal=sig_pred, bkg=bkg_pred, bkg_uncertainty=bkg_pred_errs
  )
  
  observations = obs_data + model.config.auxdata
  
  CLs_obs, CLs_exp = pyhf.infer.hypotest(
      1.0,  # null hypothesis
      observations,
      model,
      test_stat="qtilde",
      return_expected_set=True,
  )
  
  print(f"Nominal signal (μ = 1)      Observed CLs: {CLs_obs:.6f}")
  for expected_value, n_sigma in zip(CLs_exp, np.arange(-2, 3)):
      print(f"Nominal signal (μ = 1) Expected CLs({n_sigma:2d} σ): {expected_value:.6f}")
  
  
  
  poi_values = np.linspace(0.1, 5, 50)
  obs_limit, exp_limits, (scan, results) = pyhf.infer.intervals.upper_limits.upper_limit(
      observations, model, poi_values, level=0.05, return_results=True
  )
  print(f"Upper limit (obs): μ = {obs_limit:.4f}")
  print(f"Upper limit (exp): μ = {exp_limits[2]:.4f}")
  
  
  fig, ax = plt.subplots()
  fig.set_size_inches(10.5, 7)
  ax.set_title("Hypothesis Tests")
  
  artists = brazil.plot_results(poi_values, results, ax=ax)
  
  
  fig.savefig("%s/brazil.pdf"% plot_dir)
  
