# -*-coding:Utf-8 -*

"""
  plot_cloud.py
  Author : Alexis Petit, IMCCE, IFAC-CNR
  Date : 2019 02 12
  Plot informations about the cloud of space debris.

  Python >= 3.7 fixes and portability by Daniel Kastinen
"""
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm

earth_eq_radius = 6378136.6        #[m]
earth_mu        = 0.3986004415E15  #[km3/s2] JGM-3

def plot(outputs_folder, plots_folder):
  if not plots_folder.is_dir():
    plots_folder.mkdir(parents=True)

  f = open(outputs_folder / 'cloud.txt','r')
  data = f.readlines()
  f.close()

  fragments = []
  headers = data[0].strip().split()
  for row in data[1:]:
    items = row.strip().split()
    fragments.append(items)
  fragments = pd.DataFrame(fragments,columns=headers)
  selected_headers = ['Julian_Day','a[m]','e','i[deg]','RAAN[deg]','Omega[deg]','MA[deg]','BC[m2/kg]','Mass[kg]','Size[m]']
  fragments[selected_headers] = fragments[selected_headers].astype(float)

  fragments['apogee']  = fragments['a[m]']*(1+fragments['e'])
  fragments['perigee'] = fragments['a[m]']*(1-fragments['e'])
  fragments['period']  = 2*np.pi*np.sqrt(fragments['a[m]']**3/earth_mu)/3600.  

  bc_frag_list = fragments['BC[m2/kg]'].values
  log_am_frag  = map(lambda x : np.log10(float(x)/2.2),list(bc_frag_list))
  mass_frag_list = fragments['Mass[kg]'].values
  log_mass_frag  = map(lambda x : np.log10(float(x)),list(mass_frag_list))
  size_frag_list = fragments['Size[m]'].values
  log_size_frag  = map(lambda x : np.log10(float(x)),list(size_frag_list))

  total_mass = sum(mass_frag_list)

  fig = plt.figure(figsize=(9,7))
  ax1 = fig.add_subplot(1,1,1)
  ax1.spines["top"].set_visible(False)
  ax1.spines["right"].set_visible(False)
  ax1.set_xlabel('Log10(Size[m])',fontsize=14)
  ax1.set_ylabel('Number of debris',fontsize=14)  
  plt.hist(list(log_size_frag),bins=100,label='NBM ('+str(len(bc_frag_list))+')')
  plt.savefig(plots_folder / 'size')
  plt.close()
    
  fig = plt.figure(figsize=(9,7))
  fig.suptitle('Total mass of the fragments: {0:.2f} kg'.format(total_mass))
  ax1 = fig.add_subplot(1,1,1)
  ax1.spines["top"].set_visible(False)
  ax1.spines["right"].set_visible(False)
  ax1.set_xlabel('Log10(Mass[kg])',fontsize=14)
  ax1.set_ylabel('Number of debris',fontsize=14)  
  plt.hist(list(log_mass_frag),bins=100,label='NBM ('+str(len(bc_frag_list))+')')
  plt.savefig(plots_folder / 'mass')
  plt.close()
    
  fig = plt.figure(figsize=(9,7))
  ax1 = fig.add_subplot(1,1,1)
  ax1.spines["top"].set_visible(False)
  ax1.spines["right"].set_visible(False)
  ax1.set_xlabel('Log10(A/M)',fontsize=14)
  ax1.set_ylabel('Number of debris',fontsize=14)  
  plt.hist(list(log_am_frag),bins=100,label='NBM ('+str(len(bc_frag_list))+')')
  plt.savefig(plots_folder / 'am')
  plt.close()

  fig, ax1= plt.subplots(figsize=(8,6))
  ax1.spines["top"].set_visible(False)
  ax1.spines["right"].set_visible(False)
  ax1.set_ylabel('Apogee/Perigee - Re [km]',fontsize=14)
  ax1.set_xlabel('Period [hour]',fontsize=14)
  ax1.plot(fragments['period'],(fragments['apogee']-earth_eq_radius)/1000.,'o',ms=3,c='b',alpha=0.2,label='Apogee')
  ax1.plot(fragments['period'],(fragments['perigee']-earth_eq_radius)/1000.,'o',ms=3,c='r',alpha=0.2,label='Perigee')
  plt.legend(loc='upper left')
  plt.savefig(plots_folder / 'gabbard')
  plt.close()

  f = open(outputs_folder / 'cloud_cart.txt','r')
  data = f.readlines()
  f.close()

  f = open(outputs_folder / 'cloud_dv.txt','r')
  data = f.readlines()
  f.close()

  dv = [] 
  headers = data[0].strip().split()
  for row in data[1:]:
    items = row.strip().split()
    dv.append(items)
  dv = pd.DataFrame(dv,columns=headers)
  dv[headers] = dv[headers].astype(float)
  dv['DV[m/s]'] = np.sqrt(dv['DVx[m/s]']**2+dv['DVy[m/s]']**2+dv['DVz[m/s]']**2)
  dv['LOG10(A/M[m2/kg])'] = np.log10(dv['A/M[m2/kg]'])

  flag = dv['Size[m]']>0.112
  dv_11 = dv[flag]

  fig, ax1= plt.subplots(figsize=(8,6))
  ax1.spines["top"].set_visible(False)
  ax1.spines["right"].set_visible(False)
  ax1.set_xlabel('LOG10(A/M) (m2/kg)',fontsize=14)
  ax1.set_ylabel('Proportion of fragments',fontsize=14)
  n, bins, patches = plt.hist(dv_11['LOG10(A/M[m2/kg])'],bins=30,density=True)

  (mu, sigma) = norm.fit(dv_11['LOG10(A/M[m2/kg])'])
  y = norm.pdf( bins, mu, sigma)
  l = plt.plot(bins, y, 'r--', linewidth=2)

  plt.savefig(plots_folder / 'histo_am_11_2_to_35')
  plt.close()

  fig, ax1= plt.subplots(figsize=(8,6))
  ax1.spines["top"].set_visible(False)
  ax1.spines["right"].set_visible(False)
  ax1.set_xlabel('DV (m/s)',fontsize=14)
  ax1.set_ylabel('A/M (m2/kg)',fontsize=14)
  ax1.plot(dv['DV[m/s]'],dv['LOG10(A/M[m2/kg])'],'o',ms=3,c='b')
  plt.savefig(plots_folder / 'dv_vs_am')
  plt.close() 

  fig, ax1= plt.subplots(figsize=(8,6))
  ax1.spines["top"].set_visible(False)
  ax1.spines["right"].set_visible(False)
  ax1.set_xlabel('Size (m)',fontsize=14)
  ax1.set_ylabel('A/M (m2/kg)',fontsize=14)
  ax1.set_xscale("log",nonpositive='clip')
  ax1.set_yscale("log",nonpositive='clip')
  ax1.plot(dv['Size[m]'],dv['A/M[m2/kg]'],'o',ms=3,c='b')
  plt.savefig(plots_folder / 'am_vs_size')
  plt.close() 

  f = open(outputs_folder / 'size_proba_law.txt','r')
  data = f.readlines()
  f.close()

  proba = []
  for row in data:
    row = [float(item) for item in row.split()]
    proba.append(row)
  f.close()
  headers = ["Size","Proba"]
  cumu = pd.DataFrame(proba,columns=headers)

  fig, ax1= plt.subplots(figsize=(8,6))
  ax1.spines["top"].set_visible(False)
  ax1.spines["right"].set_visible(False)
  ax1.set_xlabel('Log10(Size[m])',fontsize=14)
  ax1.set_ylabel('Probability',fontsize=14)
  ax1.plot(cumu['Size'],cumu['Proba'])
  plt.savefig(plots_folder / 'size_probability')
  plt.close()

  f = open(outputs_folder / 'size_cumulative_distri.txt','r')
  data = f.readlines()
  f.close()

  cumu = []
  for row in data:
    row = [float(item) for item in row.split()]
    cumu.append(row)
  f.close()
  headers = ["Size","Area"]
  cumu = pd.DataFrame(cumu,columns=headers)

  fig, ax1= plt.subplots(figsize=(8,6))
  ax1.spines["top"].set_visible(False)
  ax1.spines["right"].set_visible(False)
  ax1.set_xlabel('Log10(Size[m])',fontsize=14)
  ax1.set_ylabel('Cumulative distribution function',fontsize=14)
  ax1.plot(cumu['Size'],cumu['Area'])
  plt.savefig(plots_folder / 'size_cumulative_distribution')
  plt.close()

  f = open(outputs_folder / 'am_proba_law.txt','r')
  data = f.readlines()
  f.close()

  proba = []
  for row in data:
    row = [float(item) for item in row.split()]
    row[0] = int(row[0])
    proba.append(row)
  f.close()
  headers = ["Fragment","A/M[m2/kg]","Proba","Area"]
  cumu = pd.DataFrame(proba,columns=headers)

  #Fragment 1
  cumu = cumu[cumu["Fragment"]==1]

  fig, ax1= plt.subplots(figsize=(8,6))
  ax1.spines["top"].set_visible(False)
  ax1.spines["right"].set_visible(False)
  ax1.set_xlabel('Log10(Size[m])',fontsize=14)
  ax1.set_ylabel('Probability',fontsize=14)
  ax1.plot(cumu['A/M[m2/kg]'],cumu['Proba'])
  plt.savefig(plots_folder / 'am_probability_fragment_1')
  plt.close()

  fig, ax1= plt.subplots(figsize=(8,6))
  ax1.spines["top"].set_visible(False)
  ax1.spines["right"].set_visible(False)
  ax1.set_xlabel('Log10(Size[m])',fontsize=14)
  ax1.set_ylabel('Probability',fontsize=14)
  ax1.plot(cumu['A/M[m2/kg]'],cumu['Area'])
  plt.savefig(plots_folder / 'am_cumulative_distribution_fragment_1')
  plt.close()
