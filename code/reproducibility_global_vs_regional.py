#!/usr/bin/env python
# coding: utf-8

# ## This Jupyter notebook reproduces the results reported by Bayona et al. (2022b) on the porspective comparison between global and regional time-independent seismicity models for California, New Zealand, and Italy. 

# ### Authors: Toño Bayona, Bill Savran, Pablo Iturrieta, and Asim Khawaja.

# #### Last Update: September 28, 2022.

# In[1]:


import os
import numpy as np
import numpy
import matplotlib
import scipy.stats
import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import csep
from csep.core import regions
from csep.utils import time_utils, comcat
from csep.core import poisson_evaluations as poisson
from csep.utils import datasets, time_utils, plots
from matplotlib.lines import Line2D
from csep.core.forecasts import GriddedForecast, GriddedDataSet
from csep.models import EvaluationResult
from cartopy.io import img_tiles
from csep.utils import readers
from csep.core.regions import CartesianGrid2D
import time, datetime
import warnings
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as pyplot
import pickle


# In[2]:


warnings.filterwarnings('ignore')


# In[ ]:


t0 = time.time()


# In[ ]:


print ('Importing earthquake forecasts for prospective evaluation...')


# ### FORECASTS

# #### Global

# In[3]:


area_fnameW = './forecasts/area.dat'
GEAR1_fnameW = './forecasts/GEAR1.dat'


# In[4]:


bulk_dataW = np.loadtxt(GEAR1_fnameW, skiprows=1, delimiter=',')
bulk_areaW = np.loadtxt(area_fnameW, skiprows=1, delimiter=',')


# In[16]:


lonsW = bulk_dataW[:,0] 
latsW = bulk_dataW[:,1]
m595 = bulk_dataW[:,2] 
m605 = bulk_dataW[:,3] 
m615 = bulk_dataW[:,4] 
m625 = bulk_dataW[:,5] 
m635 = bulk_dataW[:,6] 
m645 = bulk_dataW[:,7] 
m655 = bulk_dataW[:,8] 
m665 = bulk_dataW[:,9] 
m675 = bulk_dataW[:,10] 
m685 = bulk_dataW[:,11] 
m695 = bulk_dataW[:,12] 
m705 = bulk_dataW[:,13] 
m715 = bulk_dataW[:,14] 
m725 = bulk_dataW[:,15] 
m735 = bulk_dataW[:,16] 
m745 = bulk_dataW[:,17] 
m755 = bulk_dataW[:,18] 
m765 = bulk_dataW[:,19] 
m775 = bulk_dataW[:,20] 
m785 = bulk_dataW[:,21] 
m795 = bulk_dataW[:,22] 
m805 = bulk_dataW[:,23] 
m815 = bulk_dataW[:,24] 
m825 = bulk_dataW[:,25] 
m835 = bulk_dataW[:,26] 
m845 = bulk_dataW[:,27] 
m855 = bulk_dataW[:,28] 
m865 = bulk_dataW[:,29] 
m875 = bulk_dataW[:,30] 
m885 = bulk_dataW[:,31] 
m895 = bulk_dataW[:,32]


# In[17]:


GEAR1 = pd.DataFrame()
GEAR1['longitude'] = lonsW
GEAR1['latitude'] = latsW
GEAR1['m595'] = m595
GEAR1['m605'] = m605 
GEAR1['m615'] = m615 
GEAR1['m625'] = m625 
GEAR1['m635'] = m635 
GEAR1['m645'] = m645 
GEAR1['m655'] = m655 
GEAR1['m665'] = m665 
GEAR1['m675'] = m675 
GEAR1['m685'] = m685 
GEAR1['m695'] = m695 
GEAR1['m705'] = m705 
GEAR1['m715'] = m715 
GEAR1['m725'] = m725 
GEAR1['m735'] = m735 
GEAR1['m745'] = m745 
GEAR1['m755'] = m755 
GEAR1['m765'] = m765 
GEAR1['m775'] = m775 
GEAR1['m785'] = m785 
GEAR1['m795'] = m795 
GEAR1['m805'] = m805 
GEAR1['m815'] = m815 
GEAR1['m825'] = m825 
GEAR1['m835'] = m835 
GEAR1['m845'] = m845 
GEAR1['m855'] = m855 
GEAR1['m865'] = m865 
GEAR1['m875'] = m875 
GEAR1['m885'] = m885 
GEAR1['m895'] = m895


# #### California

# In[18]:


GEAR1_Clon = GEAR1[(GEAR1['longitude'] > -128.0) & (GEAR1['longitude'] < -110.0)]
GEAR1_Clat = GEAR1_Clon[(GEAR1_Clon['latitude'] > 31.0) & (GEAR1_Clon['latitude'] < 45.0)]


# In[19]:


GEAR1_Clat.to_csv('./forecasts/GEAR1_around_California.dat')


# In[20]:


cell_areaW = bulk_areaW[:,2]


# In[21]:


area = pd.DataFrame()
area['longitude'] = lonsW
area['latitude'] = latsW
area['area'] = cell_areaW


# In[22]:


area_Clon = area[(area['longitude'] > -128.0) & (area['longitude'] < -110.0)]
area_Clat = area_Clon[(area_Clon['latitude'] > 31.0) & (area_Clon['latitude'] < 45.0)]


# In[23]:


area_Clat.to_csv('./data/areas_around_California.dat')


# In[24]:


area_fnameC = './data/areas_around_California.dat'
fore_fnameC = './forecasts/GEAR1_around_California.dat'


# In[25]:


bulk_dataC = np.loadtxt(fore_fnameC, skiprows=1, delimiter=',')
bulk_areaC = np.loadtxt(area_fnameC, skiprows=1, delimiter=',')


# In[26]:


cell_areaC = bulk_areaC[:,3]

dh = 0.1
offset = dh / 2
lonsC = bulk_dataC[:,1] - offset
latsC = bulk_dataC[:,2] - offset 


# In[27]:


m595C = bulk_dataC[:,3] 
m605C = bulk_dataC[:,4] 
m615C = bulk_dataC[:,5] 
m625C = bulk_dataC[:,6] 
m635C = bulk_dataC[:,7] 
m645C = bulk_dataC[:,8] 
m655C = bulk_dataC[:,9] 
m665C = bulk_dataC[:,10] 
m675C = bulk_dataC[:,11] 
m685C = bulk_dataC[:,12] 
m695C = bulk_dataC[:,13] 
m705C = bulk_dataC[:,14] 
m715C = bulk_dataC[:,15] 
m725C = bulk_dataC[:,16] 
m735C = bulk_dataC[:,17] 
m745C = bulk_dataC[:,18] 
m755C = bulk_dataC[:,19] 
m765C = bulk_dataC[:,20] 
m775C = bulk_dataC[:,21] 
m785C = bulk_dataC[:,22] 
m795C = bulk_dataC[:,23] 
m805C = bulk_dataC[:,24] 
m815C = bulk_dataC[:,25] 
m825C = bulk_dataC[:,26] 
m835C = bulk_dataC[:,27] 
m845C = bulk_dataC[:,28] 
m855C = bulk_dataC[:,29] 
m865C = bulk_dataC[:,30] 
m875C = bulk_dataC[:,31] 
m885C = bulk_dataC[:,32] 
m895C = bulk_dataC[:,33] 


# In[28]:


b_California = 1.0


# In[29]:


GEAR1_C = pd.DataFrame() 
GEAR1_C['longitude'] = np.round(lonsC,1)
GEAR1_C['latitude'] = np.round(latsC,1)
GEAR1_C['m495'] = ((m595C * 5.0) / (10**(-b_California *(5.95 - 4.95)))) 
GEAR1_C['m505'] = ((m595C * 5.0) / (10**(-b_California *(5.95 - 5.05)))) 
GEAR1_C['m515'] = ((m595C * 5.0) / (10**(-b_California *(5.95 - 5.15)))) 
GEAR1_C['m525'] = ((m595C * 5.0) / (10**(-b_California *(5.95 - 5.25)))) 
GEAR1_C['m535'] = ((m595C * 5.0) / (10**(-b_California *(5.95 - 5.35)))) 
GEAR1_C['m545'] = ((m595C * 5.0) / (10**(-b_California *(5.95 - 5.45)))) 
GEAR1_C['m555'] = ((m595C * 5.0) / (10**(-b_California *(5.95 - 5.55)))) 
GEAR1_C['m565'] = ((m595C * 5.0) / (10**(-b_California *(5.95 - 5.65)))) 
GEAR1_C['m575'] = ((m595C * 5.0) / (10**(-b_California *(5.95 - 5.75)))) 
GEAR1_C['m585'] = ((m595C * 5.0) / (10**(-b_California *(5.95 - 5.85)))) 
GEAR1_C['m595'] = m595C * 5.0
GEAR1_C['m605'] = m605C * 5.0 
GEAR1_C['m615'] = m615C * 5.0 
GEAR1_C['m625'] = m625C * 5.0 
GEAR1_C['m635'] = m635C * 5.0 
GEAR1_C['m645'] = m645C * 5.0 
GEAR1_C['m655'] = m655C * 5.0 
GEAR1_C['m665'] = m665C * 5.0 
GEAR1_C['m675'] = m675C * 5.0 
GEAR1_C['m685'] = m685C * 5.0 
GEAR1_C['m695'] = m695C * 5.0 
GEAR1_C['m705'] = m705C * 5.0 
GEAR1_C['m715'] = m715C * 5.0 
GEAR1_C['m725'] = m725C * 5.0 
GEAR1_C['m735'] = m735C * 5.0
GEAR1_C['m745'] = m745C * 5.0 
GEAR1_C['m755'] = m755C * 5.0
GEAR1_C['m765'] = m765C * 5.0 
GEAR1_C['m775'] = m775C * 5.0 
GEAR1_C['m785'] = m785C * 5.0 
GEAR1_C['m795'] = m795C * 5.0 
GEAR1_C['m805'] = m805C * 5.0 
GEAR1_C['m815'] = m815C * 5.0 
GEAR1_C['m825'] = m825C * 5.0 
GEAR1_C['m835'] = m835C * 5.0 
GEAR1_C['m845'] = m845C * 5.0 
GEAR1_C['m855'] = m855C * 5.0 
GEAR1_C['m865'] = m865C * 5.0 
GEAR1_C['m875'] = m875C * 5.0 
GEAR1_C['m885'] = m885C * 5.0 
GEAR1_C['m895'] = m895C * 5.0 


# In[30]:


mw_min = 4.95
mw_max = 8.95
dmw = 0.1
mws = np.arange(mw_min, mw_max+dmw/2, dmw)


# In[31]:


start_date = time_utils.strptime_to_utc_datetime('2014-01-01 00:00:00.0')
end_date = time_utils.strptime_to_utc_datetime('2022-01-01 11:59:59.0')
duration = (end_date - start_date) # in days
duration =  round(duration.days / 365.25,2) # in years
ofp = 5.0 # original forecast period (in years)
factor = duration / ofp # scaling factor
seed = 123456


# In[32]:


HKJ = './forecasts/helmstetter_et_al.hkj.aftershock-fromXML.dat'
EBEL_ET_AL = './forecasts/ebel.aftershock.corrected-fromXML.dat'
NEOKINEMA = './forecasts/bird_liu.neokinema-fromXML.dat'
PI =  './forecasts/holliday.pi-fromXML.dat'


# In[33]:


HKJ_f = csep.load_gridded_forecast(HKJ, start_date=start_date, end_date=end_date, name='HKJ').scale(factor)
EBEL_ET_AL_f = csep.load_gridded_forecast(EBEL_ET_AL, start_date=start_date, end_date=end_date, name='EBEL').scale(factor)
NEOKINEMA_f = csep.load_gridded_forecast(NEOKINEMA, start_date=start_date, end_date=end_date, name='NEOKINEMA').scale(factor)
PI_f = csep.load_gridded_forecast(PI, start_date=start_date, end_date=end_date, name='PI').scale(factor)


# In[34]:


GEAR1_PI =  './forecasts/bird-et-al.gear1_in_pi.corrected-fromXML.dat' # GEAR1 in PI region
GEAR1_EBEL = './forecasts/bird-et-al.gear1_in_ebel.corrected-fromXML.dat' # GEAR1 in EBEL region


# In[35]:


GEAR1_PI_f = csep.load_gridded_forecast(GEAR1_PI, start_date=start_date, end_date=end_date, name='GEAR1').scale(factor) # GEAR1 in PI
GEAR1_EBEL_f = csep.load_gridded_forecast(GEAR1_EBEL, start_date=start_date, end_date=end_date, name='GEAR1').scale(factor) # GEAR1 in EBEL


# In[36]:


California = pd.DataFrame()
California['longitude'] = HKJ_f.get_longitudes()
California['latitude'] = HKJ_f.get_latitudes()


# In[37]:


GEAR1_CSEPC = pd.merge(California, GEAR1_C, how="inner", on=['longitude', 'latitude'])


# In[38]:


GEAR1_CSEPC.to_csv('./forecasts/GEAR1495_California.dat')


# In[39]:


CCSEP_area = pd.DataFrame()
CCSEP_area['longitude'] = np.round(lonsC,1)
CCSEP_area['latitude'] = np.round(latsC,1)
CCSEP_area['area'] = cell_areaC 


# In[40]:


GEAR1_CSEPCa = pd.merge(California, CCSEP_area, how="inner", on=['longitude', 'latitude'])


# In[41]:


GEAR1_CSEPCa.to_csv('./data/GEAR1a_California.dat')


# In[42]:


area_fname_C = './data/GEAR1a_California.dat'
fname_C = './forecasts/GEAR1495_California.dat'


# In[43]:


def read_GEAR1_format(filename, area_filename, magnitudes):
    """filename"""
    # long, lat, >=4.95, >=5.0, ..., >= 8.95
    t0 = time.time()
    bulk_data = np.loadtxt(filename, skiprows=1, delimiter=',')
    
    # construction region information
    lons = bulk_data[:,1]
    lats = bulk_data[:,2]
    coords = np.column_stack([lons, lats])
    
    # coordinates are given as midpoints origin should be in the 'lower left' corner
    r = CartesianGrid2D.from_origins(coords, magnitudes=magnitudes)
    
    # shape: (num_space_bins, num_mag_bins)
    bulk_data_no_coords = bulk_data[:, 3:]
    
    # tono's format provides cumulative rates per meter**2
    incremental_yrly_density = np.diff(np.fliplr(bulk_data_no_coords))
    
    # computing the differences, but returning array with the same shape
    incremental_yrly_density = np.column_stack([np.fliplr(incremental_yrly_density), bulk_data_no_coords[:,-1]])
    
    # read in area to denormalize back onto csep grid
    area = np.loadtxt(area_filename, skiprows=1, delimiter=',')

    # allows us to use broadcasting
    m2_per_cell = np.reshape(area[:,-1], (len(area[:,1]), 1))
    incremental_yrly_rates = incremental_yrly_density * m2_per_cell
    
    return incremental_yrly_rates, r, magnitudes


# In[44]:


GEAR1C_f = GriddedForecast.from_custom(read_GEAR1_format, name='GEAR1', func_args=(fname_C, area_fname_C, mws)).scale(factor)


# #### New Zealand

# In[46]:


GEAR1_NZlon = GEAR1[(GEAR1['longitude'] > 164.5) & (GEAR1['longitude'] < 179.5)]
GEAR1_NZlat = GEAR1_NZlon[(GEAR1_NZlon['latitude'] > -48.5) & (GEAR1_NZlon['latitude'] < -33.5)]


# In[47]:


GEAR1_NZlat.to_csv('./forecasts/GEAR1_around_NZ.dat')


# In[48]:


area_NZlon = area[(area['longitude'] > 164.5) & (area['longitude'] < 179.5)]
area_NZlat = area_NZlon[(area_NZlon['latitude'] > -48.5) & (area_NZlon['latitude'] < -33.5)]


# In[49]:


area_NZlat.to_csv('./data/areas_around_NZ.dat')


# In[50]:


area_fnameNZ = './data/areas_around_NZ.dat'
fore_fnameNZ = './forecasts/GEAR1_around_NZ.dat'


# In[51]:


bulk_dataNZ = np.loadtxt(fore_fnameNZ, skiprows=1, delimiter=',')
bulk_areaNZ = np.loadtxt(area_fnameNZ, skiprows=1, delimiter=',')


# In[52]:


cell_areaNZ = bulk_areaNZ[:,3]

lonsNZ = bulk_dataNZ[:,1] - offset
latsNZ = bulk_dataNZ[:,2] - offset


# In[53]:


m595NZ = bulk_dataNZ[:,3] 
m605NZ = bulk_dataNZ[:,4] 
m615NZ = bulk_dataNZ[:,5] 
m625NZ = bulk_dataNZ[:,6] 
m635NZ = bulk_dataNZ[:,7] 
m645NZ = bulk_dataNZ[:,8] 
m655NZ = bulk_dataNZ[:,9] 
m665NZ = bulk_dataNZ[:,10] 
m675NZ = bulk_dataNZ[:,11] 
m685NZ = bulk_dataNZ[:,12] 
m695NZ = bulk_dataNZ[:,13] 
m705NZ = bulk_dataNZ[:,14] 
m715NZ = bulk_dataNZ[:,15] 
m725NZ = bulk_dataNZ[:,16] 
m735NZ = bulk_dataNZ[:,17] 
m745NZ = bulk_dataNZ[:,18] 
m755NZ = bulk_dataNZ[:,19] 
m765NZ = bulk_dataNZ[:,20] 
m775NZ = bulk_dataNZ[:,21] 
m785NZ = bulk_dataNZ[:,22] 
m795NZ = bulk_dataNZ[:,23] 
m805NZ = bulk_dataNZ[:,24] 
m815NZ = bulk_dataNZ[:,25] 
m825NZ = bulk_dataNZ[:,26] 
m835NZ = bulk_dataNZ[:,27] 
m845NZ = bulk_dataNZ[:,28] 
m855NZ = bulk_dataNZ[:,29] 
m865NZ = bulk_dataNZ[:,30] 
m875NZ = bulk_dataNZ[:,31] 
m885NZ = bulk_dataNZ[:,32] 
m895NZ = bulk_dataNZ[:,33] 


# In[55]:


b_NZ = 1.0


# In[56]:


GEAR1_NZ = pd.DataFrame() 
GEAR1_NZ['longitude'] = np.round(lonsNZ,1)
GEAR1_NZ['latitude'] = np.round(latsNZ,1)
GEAR1_NZ['m495'] = ((m595NZ * 5.0) / (10**(-b_NZ *(5.95 - 4.95))))
GEAR1_NZ['m505'] = ((m595NZ * 5.0) / (10**(-b_NZ *(5.95 - 5.05))))
GEAR1_NZ['m515'] = ((m595NZ * 5.0) / (10**(-b_NZ *(5.95 - 5.15))))
GEAR1_NZ['m525'] = ((m595NZ * 5.0) / (10**(-b_NZ *(5.95 - 5.25))))
GEAR1_NZ['m535'] = ((m595NZ * 5.0) / (10**(-b_NZ *(5.95 - 5.35))))
GEAR1_NZ['m545'] = ((m595NZ * 5.0) / (10**(-b_NZ *(5.95 - 5.45))))
GEAR1_NZ['m555'] = ((m595NZ * 5.0) / (10**(-b_NZ *(5.95 - 5.55))))
GEAR1_NZ['m565'] = ((m595NZ * 5.0) / (10**(-b_NZ *(5.95 - 5.65))))
GEAR1_NZ['m575'] = ((m595NZ * 5.0) / (10**(-b_NZ *(5.95 - 5.75))))
GEAR1_NZ['m585'] = ((m595NZ * 5.0) / (10**(-b_NZ *(5.95 - 5.85))))
GEAR1_NZ['m595'] = m595NZ * 5.0
GEAR1_NZ['m605'] = m605NZ * 5.0
GEAR1_NZ['m615'] = m615NZ * 5.0
GEAR1_NZ['m625'] = m625NZ * 5.0
GEAR1_NZ['m635'] = m635NZ * 5.0
GEAR1_NZ['m645'] = m645NZ * 5.0
GEAR1_NZ['m655'] = m655NZ * 5.0
GEAR1_NZ['m665'] = m665NZ * 5.0
GEAR1_NZ['m675'] = m675NZ * 5.0
GEAR1_NZ['m685'] = m685NZ * 5.0
GEAR1_NZ['m695'] = m695NZ * 5.0
GEAR1_NZ['m705'] = m705NZ * 5.0
GEAR1_NZ['m715'] = m715NZ * 5.0
GEAR1_NZ['m725'] = m725NZ * 5.0
GEAR1_NZ['m735'] = m735NZ * 5.0
GEAR1_NZ['m745'] = m745NZ * 5.0
GEAR1_NZ['m755'] = m755NZ * 5.0
GEAR1_NZ['m765'] = m765NZ * 5.0
GEAR1_NZ['m775'] = m775NZ * 5.0
GEAR1_NZ['m785'] = m785NZ * 5.0
GEAR1_NZ['m795'] = m795NZ * 5.0
GEAR1_NZ['m805'] = m805NZ * 5.0
GEAR1_NZ['m815'] = m815NZ * 5.0
GEAR1_NZ['m825'] = m825NZ * 5.0
GEAR1_NZ['m835'] = m835NZ * 5.0
GEAR1_NZ['m845'] = m845NZ * 5.0
GEAR1_NZ['m855'] = m855NZ * 5.0
GEAR1_NZ['m865'] = m865NZ * 5.0
GEAR1_NZ['m875'] = m875NZ * 5.0
GEAR1_NZ['m885'] = m885NZ * 5.0
GEAR1_NZ['m895'] = m895NZ * 5.0


# In[57]:


NZHM = './forecasts/NZHM_5year_rates-fromXML.dat'
PPE = './forecasts/nz5yrppe_c.dat'
SUP = './forecasts/nz5yrsup_c.dat'


# In[58]:


NZHM_f = csep.load_gridded_forecast(NZHM, start_date=start_date, end_date=end_date, name='NZHM').scale(factor)
PPE_f = csep.load_gridded_forecast(PPE, start_date=start_date, end_date=end_date, name='PPE').scale(factor)
SUP_f = csep.load_gridded_forecast(SUP, start_date=start_date, end_date=end_date, name='SUP').scale(factor)


# In[59]:


NZ = pd.DataFrame()
NZ['longitude'] = NZHM_f.get_longitudes()
NZ['latitude'] = NZHM_f.get_latitudes()


# In[60]:


GEAR1_CSEPNZ = pd.merge(NZ, GEAR1_NZ, how="inner", on=['longitude', 'latitude'])


# In[61]:


GEAR1_CSEPNZ.to_csv('./forecasts/GEAR1495_NZ.dat')


# In[62]:


NZCSEP_area = pd.DataFrame()
NZCSEP_area['longitude'] = np.round(lonsNZ,1)
NZCSEP_area['latitude'] = np.round(latsNZ,1)
NZCSEP_area['area'] = cell_areaNZ 


# In[63]:


GEAR1_CSEPNZa = pd.merge(NZ, NZCSEP_area, how="inner", on=['longitude', 'latitude'])


# In[64]:


GEAR1_CSEPNZa.to_csv('./data/GEAR1a_NZ.dat')


# In[65]:


area_fname_NZ = './data/GEAR1a_NZ.dat'
fname_NZ = './forecasts/GEAR1495_NZ.dat'
fname2_NZ = './forecasts/KJSS495_NZ.dat'


# In[66]:


GEAR1NZ_f = GriddedForecast.from_custom(read_GEAR1_format, name='GEAR1', func_args=(fname_NZ, area_fname_NZ, mws)).scale(factor)
KJSSNZ_f = GriddedForecast.from_custom(read_GEAR1_format, name='KJSS', func_args=(fname2_NZ, area_fname_NZ, mws)).scale(factor)


# #### Italy

# In[67]:


GEAR1_Ilon = GEAR1[(GEAR1['longitude'] > 3.0) & (GEAR1['longitude'] < 22.0)]
GEAR1_Ilat = GEAR1_Ilon[(GEAR1_Ilon['latitude'] > 35.0) & (GEAR1_Ilon['latitude'] < 48.0)]


# In[68]:


GEAR1_Ilat.to_csv('./data/GEAR1_around_Italy.dat')


# In[69]:


area_Ilon = area[(area['longitude'] > 3.0) & (area['longitude'] < 22.0)]
area_Ilat = area_Ilon[(area_Ilon['latitude'] > 35.0) & (area_Ilon['latitude'] < 48.0)]


# In[70]:


area_Ilat.to_csv('./data/areas_around_Italy.dat')


# In[71]:


area_fnameI = './data/areas_around_Italy.dat'
fore_fnameI = './forecasts/GEAR1_around_Italy.dat'


# In[72]:


bulk_dataI = np.loadtxt(fore_fnameI, skiprows=1, delimiter=',')
bulk_areaI = np.loadtxt(area_fnameI, skiprows=1, delimiter=',')


# In[73]:


cell_areaI = bulk_areaI[:,3]

lonsI = bulk_dataI[:,1] - offset
latsI = bulk_dataI[:,2] - offset


# In[74]:


m595I = bulk_dataI[:,3]
m605I = bulk_dataI[:,4] 
m615I = bulk_dataI[:,5] 
m625I = bulk_dataI[:,6] 
m635I = bulk_dataI[:,7] 
m645I = bulk_dataI[:,8] 
m655I = bulk_dataI[:,9] 
m665I = bulk_dataI[:,10] 
m675I = bulk_dataI[:,11] 
m685I = bulk_dataI[:,12] 
m695I = bulk_dataI[:,13] 
m705I = bulk_dataI[:,14] 
m715I = bulk_dataI[:,15] 
m725I = bulk_dataI[:,16] 
m735I = bulk_dataI[:,17] 
m745I = bulk_dataI[:,18] 
m755I = bulk_dataI[:,19] 
m765I = bulk_dataI[:,20] 
m775I = bulk_dataI[:,21] 
m785I = bulk_dataI[:,22] 
m795I = bulk_dataI[:,23] 
m805I = bulk_dataI[:,24] 
m815I = bulk_dataI[:,25] 
m825I = bulk_dataI[:,26] 
m835I = bulk_dataI[:,27] 
m845I = bulk_dataI[:,28] 
m855I = bulk_dataI[:,29] 
m865I = bulk_dataI[:,30] 
m875I = bulk_dataI[:,31] 
m885I = bulk_dataI[:,32] 
m895I = bulk_dataI[:,33] 


# In[75]:


b_Italy = 1.0


# In[76]:


GEAR1_I = pd.DataFrame() 
GEAR1_I['longitude'] = np.round(lonsI,1)
GEAR1_I['latitude'] = np.round(latsI,1)
GEAR1_I['m495'] = ((m595I * 5.0) / (10**(-b_Italy *(5.95 - 4.95)))) 
GEAR1_I['m505'] = ((m595I * 5.0) / (10**(-b_Italy *(5.95 - 5.05)))) 
GEAR1_I['m515'] = ((m595I * 5.0) / (10**(-b_Italy *(5.95 - 5.15)))) 
GEAR1_I['m525'] = ((m595I * 5.0) / (10**(-b_Italy *(5.95 - 5.25)))) 
GEAR1_I['m535'] = ((m595I * 5.0) / (10**(-b_Italy *(5.95 - 5.35)))) 
GEAR1_I['m545'] = ((m595I * 5.0) / (10**(-b_Italy *(5.95 - 5.45)))) 
GEAR1_I['m555'] = ((m595I * 5.0) / (10**(-b_Italy *(5.95 - 5.55)))) 
GEAR1_I['m565'] = ((m595I * 5.0) / (10**(-b_Italy *(5.95 - 5.65)))) 
GEAR1_I['m575'] = ((m595I * 5.0) / (10**(-b_Italy *(5.95 - 5.75)))) 
GEAR1_I['m585'] = ((m595I * 5.0) / (10**(-b_Italy *(5.95 - 5.85)))) 
GEAR1_I['m595'] = m595I * 5.0
GEAR1_I['m605'] = m605I * 5.0 
GEAR1_I['m615'] = m615I * 5.0 
GEAR1_I['m625'] = m625I * 5.0 
GEAR1_I['m635'] = m635I * 5.0 
GEAR1_I['m645'] = m645I * 5.0 
GEAR1_I['m655'] = m655I * 5.0 
GEAR1_I['m665'] = m665I * 5.0 
GEAR1_I['m675'] = m675I * 5.0 
GEAR1_I['m685'] = m685I * 5.0 
GEAR1_I['m695'] = m695I * 5.0 
GEAR1_I['m705'] = m705I * 5.0 
GEAR1_I['m715'] = m715I * 5.0 
GEAR1_I['m725'] = m725I * 5.0 
GEAR1_I['m735'] = m735I * 5.0 
GEAR1_I['m745'] = m745I * 5.0 
GEAR1_I['m755'] = m755I * 5.0 
GEAR1_I['m765'] = m765I * 5.0 
GEAR1_I['m775'] = m775I * 5.0 
GEAR1_I['m785'] = m785I * 5.0 
GEAR1_I['m795'] = m795I * 5.0 
GEAR1_I['m805'] = m805I * 5.0 
GEAR1_I['m815'] = m815I * 5.0 
GEAR1_I['m825'] = m825I * 5.0 
GEAR1_I['m835'] = m835I * 5.0 
GEAR1_I['m845'] = m845I * 5.0 
GEAR1_I['m855'] = m855I * 5.0 
GEAR1_I['m865'] = m865I * 5.0 
GEAR1_I['m875'] = m875I * 5.0 
GEAR1_I['m885'] = m885I * 5.0 
GEAR1_I['m895'] = m895I * 5.0


# In[77]:


ALM = './forecasts/gulia-wiemer.ALM.italy.5yr.2010-01-01.dat'
HALM = './forecasts/gulia-wiemer.HALM.italy.5yr.2010-01-01.dat'
ALM_IT = './forecasts/schorlemmer-wiemer.ALM_IT.italy.5yr.2010-01-01.dat'
MPS04_AFTER = './forecasts/meletti.MPS04after.italy.5yr.2010-01-01.dat'
HAZGRIDX = './forecasts/akinci-lombardi.HAZGRIDX.italy.5yr.2010-01-01.dat'
HZATI = './forecasts/chan.HZA_TI.italy.5yr.2010-01-01.dat'
RI = './forecasts/nanjo.RI.italy.5yr.2010-01-01.dat'
HRSS_CSI = './forecasts/werner.HiResSmoSeis-m1.italy.5yr.2010-01-01.dat'
HRSS_HYBRID = './forecasts/werner.HiResSmoSeis-m2.italy.5yr.2010-01-01.dat'
TRIPLES_CPTI = './forecasts/zechar.TripleS-CPTI.italy.5yr.2010-01-01.dat'
TRIPLES_CSI = './forecasts/zechar.TripleS-CSI.italy.5yr.2010-01-01.dat'
TRIPLES_HYBRID = './forecasts/zechar.TripleS-hybrid.italy.5yr.2010-01-01.dat'


# In[78]:


ALM_f = csep.load_gridded_forecast(ALM, start_date=start_date, end_date=end_date, swap_latlon=True, name='ALM').scale(factor)
HALM_f = csep.load_gridded_forecast(HALM, start_date=start_date,  end_date=end_date, swap_latlon=True, name='HALM').scale(factor)
ALM_IT_f = csep.load_gridded_forecast(ALM_IT, start_date=start_date, end_date=end_date, swap_latlon=True, name='ALM-IT').scale(factor)
MPS04_AFTER_f = csep.load_gridded_forecast(MPS04_AFTER, start_date=start_date, end_date=end_date, swap_latlon=True, name='MPS04-AFTER').scale(factor)
HAZGRIDX_f = csep.load_gridded_forecast(HAZGRIDX, start_date=start_date, end_date=end_date, swap_latlon=True, name='HAZGRIDX').scale(factor)
HZATI_f = csep.load_gridded_forecast(HZATI, start_date=start_date, end_date=end_date, swap_latlon=True, name='HZATI').scale(factor)
RI_f = csep.load_gridded_forecast(RI, start_date=start_date, end_date=end_date, swap_latlon=True, name='RI').scale(factor)
HRSS_CSI_f = csep.load_gridded_forecast(HRSS_CSI, start_date=start_date, end_date=end_date, swap_latlon=True, name='HRSS-CSI').scale(factor)
HRSS_HYBRID_f = csep.load_gridded_forecast(HRSS_HYBRID, start_date=start_date, end_date=end_date, swap_latlon=True, name='HRSS-HYBRID').scale(factor)
TRIPLES_CPTI_f = csep.load_gridded_forecast(TRIPLES_CPTI, start_date=start_date, end_date=end_date, swap_latlon=True, name='TRIPLE_S-CPTI').scale(factor)
TRIPLES_CSI_f = csep.load_gridded_forecast(TRIPLES_CSI, start_date=start_date, end_date=end_date, swap_latlon=True, name='TRIPLE_S-CSI').scale(factor)
TRIPLES_HYBRID_f = csep.load_gridded_forecast(TRIPLES_HYBRID, start_date=start_date, end_date=end_date, swap_latlon=True, name='TRIPLE_S-HYBRID').scale(factor)


# In[79]:


Italy = pd.DataFrame()
Italy['longitude'] = HRSS_CSI_f.get_longitudes()
Italy['latitude'] = HRSS_CSI_f.get_latitudes()


# In[80]:


GEAR1_CSEPI = pd.merge(Italy, GEAR1_I, how="inner", on=['longitude', 'latitude'])


# In[81]:


GEAR1_CSEPI.to_csv('./forecasts/GEAR1495_Italy.dat')


# In[82]:


ICSEP_area = pd.DataFrame()
ICSEP_area['longitude'] = np.round(lonsI,1)
ICSEP_area['latitude'] = np.round(latsI,1)
ICSEP_area['area'] = cell_areaI 


# In[83]:


GEAR1_CSEPIa = pd.merge(Italy, ICSEP_area, how="inner", on=['longitude', 'latitude'])


# In[84]:


GEAR1_CSEPIa.to_csv('./data/GEAR1495a_Italy.dat')


# In[85]:


area_fname_I = './data/GEAR1495a_Italy.dat'
fname_I = './forecasts/GEAR1495_Italy.dat'


# In[86]:


GEAR1I_f = GriddedForecast.from_custom(read_GEAR1_format, name='GEAR1', func_args=(fname_I, area_fname_I, mws)).scale(factor * (1./1.602))


# In[87]:


def _get_basemap(basemap):

    if basemap == 'stamen_terrain':
        tiles = img_tiles.Stamen('terrain')
    elif basemap == 'stamen_terrain-background':
        tiles = img_tiles.Stamen('terrain-background')
    elif basemap == 'google-satellite':
        tiles = img_tiles.GoogleTiles(style='satellite')
    elif basemap == 'ESRI_terrain':
        webservice = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/'                  'MapServer/tile/{z}/{y}/{x}.jpg'
        tiles = img_tiles.GoogleTiles(url=webservice)
    elif basemap == 'ESRI_imagery':
        webservice = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/'                  'MapServer/tile/{z}/{y}/{x}.jpg'
        tiles = img_tiles.GoogleTiles(url=webservice)
    elif basemap == 'ESRI_relief':
        webservice = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/'                  'MapServer/tile/{z}/{y}/{x}.jpg'
        tiles = img_tiles.GoogleTiles(url=webservice)
    elif basemap == 'ESRI_topo':
        webservice = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/'                  'MapServer/tile/{z}/{y}/{x}.jpg'
        tiles = img_tiles.GoogleTiles(url=webservice)
    else:
        raise ValueError('Basemap type not valid or not implemented')

    return tiles


# In[ ]:


print ('Plotting global and regional seismicity forecasts on a map (Fig. 1)...')


# In[88]:


fig = plt.figure(figsize=(17,10))

ax_GEAR1C = fig.add_subplot(231, projection=ccrs.PlateCarree(), adjustable='datalim')
ax_GEAR1C.add_feature(cartopy.feature.STATES, facecolor='None', edgecolor='lightgrey', linewidth=1.5)
ax_GEAR1C.add_image(_get_basemap('ESRI_imagery'), 6)

dh = round(GEAR1C_f.region.dh, 5)
gl = ax_GEAR1C.gridlines()
gl.xlines = False
gl.ylines = False
gl.xlabels_top = True
gl.xlabels_bottom = False 
gl.ylabels_right = False
gl.ylabels_left = True
gl.xlabel_style = {'size': 13}
gl.ylabel_style = {'size': 13}
gl.xlocator = mticker.FixedLocator([-124,-119, -114])
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([32, 34, 36, 38, 40, 42])
gl.yformatter = LATITUDE_FORMATTER

ax_GEAR1C.set_ylim(min(GEAR1C_f.get_latitudes())-0.1+dh/2, max(GEAR1C_f.get_latitudes())+0.1+dh/2)
ax_GEAR1C.set_xlim(min(GEAR1C_f.get_longitudes())-0.1+dh/2, max(GEAR1C_f.get_longitudes())+0.1+dh/2)
props = dict(boxstyle='square', facecolor='white', alpha=1)
ax_GEAR1C.text(0.74, 0.95, f'{GEAR1C_f.name}', fontsize=18, transform=ax_GEAR1C.transAxes, verticalalignment='top', bbox=props)


    
scatter = ax_GEAR1C.scatter(GEAR1C_f.get_longitudes()+dh/2, GEAR1C_f.get_latitudes()+dh/2,  
                         c = np.log10(GEAR1C_f.spatial_counts()), cmap='inferno', vmin=-4.5, vmax=-0.5,
                         s=8, marker='s', alpha =1, edgecolor="None", zorder=1)

ptsC = GEAR1C_f.region.tight_bbox()
ax_GEAR1C.plot(ptsC[:,0], ptsC[:,1], lw=1, color='black', transform=ccrs.PlateCarree(), zorder=2)


ax_GEAR1NZ = fig.add_subplot(232, projection=ccrs.PlateCarree(), adjustable='datalim')
ax_GEAR1NZ.add_feature(cartopy.feature.BORDERS, color='lightgrey', linewidth=1.5, zorder=2)
ax_GEAR1NZ.add_feature(cartopy.feature.COASTLINE, edgecolor='lightgrey', linewidth=1.5, zorder=2)
ax_GEAR1NZ.add_image(_get_basemap('ESRI_imagery'), 6)
gl = ax_GEAR1NZ.gridlines()
gl.xlines = False
gl.ylines = False
gl.xlabels_top = True 
gl.xlabels_bottom = False 
gl.ylabels_right = False
gl.ylabels_left = True
gl.xlabel_style = {'size': 13}
gl.ylabel_style = {'size': 13}
gl.xlocator = mticker.FixedLocator([166,170, 174, 178])
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([-35, -38, -41, -44, -47])
gl.yformatter = LATITUDE_FORMATTER

ax_GEAR1NZ.set_ylim(min(GEAR1NZ_f.get_latitudes())-0.1+dh/2, max(GEAR1NZ_f.get_latitudes())+0.1+dh/2)
ax_GEAR1NZ.set_xlim(min(GEAR1NZ_f.get_longitudes())-0.1+dh/2, max(GEAR1NZ_f.get_longitudes())+0.1+dh/2)
ax_GEAR1NZ.text(0.715, 0.95, f'{GEAR1C_f.name}', fontsize=18, transform=ax_GEAR1NZ.transAxes, verticalalignment='top', bbox=props)
    
scatter = ax_GEAR1NZ.scatter(GEAR1NZ_f.get_longitudes()+dh/2, GEAR1NZ_f.get_latitudes()+dh/2,  
                         c = np.log10(GEAR1NZ_f.spatial_counts()), cmap='inferno', vmin=-4.5, vmax=-0.5,
                         s=5, marker='s', alpha =1, edgecolor="None", zorder=1)


ptsNZ = GEAR1NZ_f.region.tight_bbox()
ax_GEAR1NZ.plot(ptsNZ[:,0], ptsNZ[:,1], lw=1, color='black', transform=ccrs.PlateCarree(), zorder=2)


ax_GEAR1I = fig.add_subplot(233, projection=ccrs.PlateCarree(), adjustable='datalim')
ax_GEAR1I.add_feature(cartopy.feature.BORDERS, color='lightgrey', linewidth=1.5, zorder=1)
ax_GEAR1I.add_feature(cartopy.feature.COASTLINE, edgecolor='lightgrey', linewidth=1.5, zorder=1)
ax_GEAR1I.add_image(_get_basemap('ESRI_imagery'), 6)
gl = ax_GEAR1I.gridlines()
gl.xlines = False
gl.ylines = False
gl.xlabels_top = True 
gl.xlabels_bottom = False 
gl.ylabels_right = False
gl.ylabels_left = True
gl.xlabel_style = {'size': 13}
gl.ylabel_style = {'size': 13}
gl.xlocator = mticker.FixedLocator([6, 10, 14, 18])
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([37, 39, 41, 43, 45, 47])
gl.yformatter = LATITUDE_FORMATTER

ax_GEAR1I.set_ylim(min(GEAR1I_f.get_latitudes())-0.1+dh/2, max(GEAR1I_f.get_latitudes())+0.1+dh/2)
ax_GEAR1I.set_xlim(min(GEAR1I_f.get_longitudes())-0.1+dh/2, max(GEAR1I_f.get_longitudes())+0.1+dh/2)
ax_GEAR1I.text(0.74, 0.95, f'{GEAR1I_f.name}', fontsize=18, transform=ax_GEAR1I.transAxes, verticalalignment='top', bbox=props)
    
scatter = ax_GEAR1I.scatter(GEAR1I_f.get_longitudes()+dh/2, GEAR1I_f.get_latitudes()+dh/2,  
                         c = np.log10(GEAR1I_f.spatial_counts()), cmap='inferno', vmin=-4.5, vmax=-0.5,
                         s=8, marker='s', alpha =1, edgecolor="None", zorder=1)



ptsI = GEAR1I_f.region.tight_bbox()
ax_GEAR1I.plot(ptsI[:,0], ptsI[:,1], lw=1, color='black', transform=ccrs.PlateCarree())

ax_HKJ = fig.add_subplot(234, projection=ccrs.PlateCarree(), adjustable='datalim')
ax_HKJ.add_feature(cartopy.feature.STATES, facecolor='None', edgecolor='lightgrey', linewidth=1.5)
ax_HKJ.add_image(_get_basemap('ESRI_imagery'), 6)
gl = ax_HKJ.gridlines()
gl.xlines = False
gl.ylines = False
gl.xlabels_top = True 
gl.xlabels_bottom = False 
gl.ylabels_right = False
gl.ylabels_left = True
gl.xlabel_style = {'size': 13}
gl.ylabel_style = {'size': 13}
gl.xlocator = mticker.FixedLocator([-124,-119, -114])
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([32, 34, 36, 38, 40, 42])
gl.yformatter = LATITUDE_FORMATTER

ax_HKJ.set_ylim(min(HKJ_f.get_latitudes())-0.1+dh/2, max(HKJ_f.get_latitudes())+0.1+dh/2)
ax_HKJ.set_xlim(min(HKJ_f.get_longitudes())-0.1+dh/2, max(HKJ_f.get_longitudes())+0.1+dh/2)
ax_HKJ.text(0.85, 0.95, f'{HKJ_f.name}', fontsize=18, transform=ax_HKJ.transAxes, verticalalignment='top', bbox=props)

scatter = ax_HKJ.scatter(HKJ_f.get_longitudes()+dh/2, HKJ_f.get_latitudes()+dh/2,  
                         c = np.log10(HKJ_f.spatial_counts()), cmap='inferno', vmin=-4.5, vmax=-0.5,
                         s=8, marker='s', alpha =1, edgecolor="None", zorder=1)

ax_HKJ.plot(ptsC[:,0], ptsC[:,1], lw=1, color='black', transform=ccrs.PlateCarree(), zorder=2)



ax_PPE = fig.add_subplot(235, projection=ccrs.PlateCarree(), adjustable='datalim')
ax_PPE.add_feature(cartopy.feature.BORDERS, color='lightgrey', linewidth=1.5, zorder=2)
ax_PPE.add_feature(cartopy.feature.COASTLINE, edgecolor='lightgrey', linewidth=1.5, zorder=2)
ax_PPE.add_image(_get_basemap('ESRI_imagery'), 6)
gl = ax_PPE.gridlines()
gl.xlines = False
gl.ylines = False
gl.xlabels_top = True 
gl.xlabels_bottom = False 
gl.ylabels_right = False
gl.ylabels_left = True
gl.xlabel_style = {'size': 13}
gl.ylabel_style = {'size': 13}
gl.xlocator = mticker.FixedLocator([166,170, 174, 178])
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([-35, -38, -41, -44, -47])
gl.yformatter = LATITUDE_FORMATTER

ax_PPE.set_ylim(min(PPE_f.get_latitudes())-0.1+dh/2, max(PPE_f.get_latitudes())+0.1+dh/2)
ax_PPE.set_xlim(min(PPE_f.get_longitudes())-0.1+dh/2, max(PPE_f.get_longitudes())+0.1+dh/2)
ax_PPE.text(0.815, 0.95, f'{PPE_f.name}', fontsize=18, transform=ax_PPE.transAxes, verticalalignment='top', bbox=props)

    
scatter = ax_PPE.scatter(PPE_f.get_longitudes()+dh/2, PPE_f.get_latitudes()+dh/2,  
                           c = np.log10(PPE_f.spatial_counts()), cmap='inferno', s=5, vmin=-4.5, vmax=-0.5,
                           marker='s', alpha =1, edgecolor="None", zorder=1)

ax_PPE.plot(ptsNZ[:,0], ptsNZ[:,1], lw=1, color='black', transform=ccrs.PlateCarree(), zorder=2)



ax_HRSS_CSI = fig.add_subplot(236, projection=ccrs.PlateCarree(), adjustable='datalim')
ax_HRSS_CSI.add_feature(cartopy.feature.BORDERS, color='lightgrey', linewidth=1.5, zorder=1)
ax_HRSS_CSI.add_feature(cartopy.feature.COASTLINE, edgecolor='lightgrey', linewidth=1.5, zorder=1)
ax_HRSS_CSI.add_image(_get_basemap('ESRI_imagery'), 6)
gl = ax_HRSS_CSI.gridlines()
gl.xlines = False
gl.ylines = False
gl.xlabels_top = True 
gl.xlabels_bottom = False 
gl.ylabels_right = False
gl.ylabels_left = True
gl.xlabel_style = {'size': 13}
gl.ylabel_style = {'size': 13}
gl.xlocator = mticker.FixedLocator([6, 10, 14, 18])
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([37, 39, 41, 43, 45, 47])
gl.yformatter = LATITUDE_FORMATTER

ax_HRSS_CSI.set_ylim(min(HRSS_CSI_f.get_latitudes())-0.1+dh/2, max(HRSS_CSI_f.get_latitudes())+0.1+dh/2)
ax_HRSS_CSI.set_xlim(min(HRSS_CSI_f.get_longitudes())-0.1+dh/2, max(HRSS_CSI_f.get_longitudes())+0.1+dh/2)
ax_HRSS_CSI.text(0.66, 0.95, f'{HRSS_CSI_f.name}', fontsize=18, transform=ax_HRSS_CSI.transAxes, verticalalignment='top', bbox=props)

    
scatter = ax_HRSS_CSI.scatter(HRSS_CSI_f.get_longitudes()+dh/2, HRSS_CSI_f.get_latitudes()+dh/2, 
                              c = np.log10(HRSS_CSI_f.spatial_counts()), cmap='inferno', s=8, vmin=-4.5, vmax=-0.5, 
                              marker='s',alpha =1, edgecolor="None", zorder=1)

ax_HRSS_CSI.plot(ptsI[:,0], ptsI[:,1], lw=1, color='black', transform=ccrs.PlateCarree(), zorder=2)

plt.savefig('./output/TSR_Fig1_top.png', dpi=150, bbox_inches = 'tight')


# In[89]:


fig = plt.figure(figsize=(17,10))

ax_NEOKINEMA = fig.add_subplot(231, projection=ccrs.PlateCarree(), adjustable='datalim')
ax_NEOKINEMA.add_feature(cartopy.feature.STATES, facecolor='None', edgecolor='lightgrey', linewidth=1.5)
ax_NEOKINEMA.add_image(_get_basemap('ESRI_imagery'), 6)
gl = ax_NEOKINEMA.gridlines()
gl.xlines = False
gl.ylines = False
gl.xlabels_top = True 
gl.xlabels_bottom = False 
gl.ylabels_right = False
gl.ylabels_left = True
gl.xlabel_style = {'size': 13}
gl.ylabel_style = {'size': 13}
gl.xlocator = mticker.FixedLocator([-124,-119, -114])
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([32, 34, 36, 38, 40, 42])
gl.yformatter = LATITUDE_FORMATTER

ax_NEOKINEMA.set_ylim(min(NEOKINEMA_f.get_latitudes())-0.1+dh/2, max(NEOKINEMA_f.get_latitudes())+0.1+dh/2)
ax_NEOKINEMA.set_xlim(min(NEOKINEMA_f.get_longitudes())-0.1+dh/2, max(NEOKINEMA_f.get_longitudes())+0.1+dh/2)
ax_NEOKINEMA.text(0.57, 0.95, f'{NEOKINEMA_f.name}', fontsize=18, transform=ax_NEOKINEMA.transAxes, verticalalignment='top', bbox=props)

    
scatter = ax_NEOKINEMA.scatter(NEOKINEMA_f.get_longitudes()+dh/2, NEOKINEMA_f.get_latitudes()+dh/2,  
                         c = np.log10(NEOKINEMA_f.spatial_counts()), cmap='inferno', vmin=-4.5, vmax=-0.5,
                         s=8, marker='s', alpha =1, edgecolor="None", zorder=1)

ax_NEOKINEMA.plot(ptsC[:,0], ptsC[:,1], lw=1, color='black', transform=ccrs.PlateCarree(), zorder=2)




ax_NZHM = fig.add_subplot(232, projection=ccrs.PlateCarree(), adjustable='datalim')
ax_NZHM.add_feature(cartopy.feature.BORDERS, color='lightgrey', linewidth=1.5, zorder=2)
ax_NZHM.add_feature(cartopy.feature.COASTLINE, edgecolor='lightgrey', linewidth=1.5, zorder=2)
ax_NZHM.add_image(_get_basemap('ESRI_imagery'), 6)
gl = ax_NZHM.gridlines()
gl.xlines = False
gl.ylines = False
gl.xlabels_top = True 
gl.xlabels_bottom = False 
gl.ylabels_right = False
gl.ylabels_left = True
gl.xlabel_style = {'size': 13}
gl.ylabel_style = {'size': 13}
gl.xlocator = mticker.FixedLocator([166,170, 174, 178])
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([-35, -38, -41, -44, -47])
gl.yformatter = LATITUDE_FORMATTER

ax_NZHM.set_ylim(min(NZHM_f.get_latitudes())-0.1+dh/2, max(NZHM_f.get_latitudes())+0.1+dh/2)
ax_NZHM.set_xlim(min(NZHM_f.get_longitudes())-0.1+dh/2, max(NZHM_f.get_longitudes())+0.1+dh/2)
ax_NZHM.text(0.74, 0.95, f'{NZHM_f.name}', fontsize=18, transform=ax_NZHM.transAxes, verticalalignment='top', bbox=props)


scatter = ax_NZHM.scatter(NZHM_f.get_longitudes()+dh/2, NZHM_f.get_latitudes()+dh/2,  
                         c = np.log10(NZHM_f.spatial_counts()), cmap='inferno', vmin=-4.5, vmax=-0.5,
                         s=5, marker='s', alpha =1, edgecolor="None", zorder=1)


ax_NZHM.plot(ptsNZ[:,0], ptsNZ[:,1], lw=1, color='black', transform=ccrs.PlateCarree(), zorder=2)



ax_ALM_IT = fig.add_subplot(233, projection=ccrs.PlateCarree(), adjustable='datalim')
ax_ALM_IT.add_feature(cartopy.feature.BORDERS, color='lightgrey', linewidth=1.5, zorder=1)
ax_ALM_IT.add_feature(cartopy.feature.COASTLINE, edgecolor='lightgrey', linewidth=1.5, zorder=1)
ax_ALM_IT.add_image(_get_basemap('ESRI_imagery'), 6)
gl = ax_ALM_IT.gridlines()
gl.xlines = False
gl.ylines = False
gl.xlabels_top = True 
gl.xlabels_bottom = False 
gl.ylabels_right = False
gl.ylabels_left = True
gl.xlabel_style = {'size': 13}
gl.ylabel_style = {'size': 13}
gl.xlocator = mticker.FixedLocator([6, 10, 14, 18])
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([37, 39, 41, 43, 45, 47])
gl.yformatter = LATITUDE_FORMATTER

ax_ALM_IT.set_ylim(min(ALM_IT_f.get_latitudes())-0.1+dh/2, max(ALM_IT_f.get_latitudes())+0.1+dh/2)
ax_ALM_IT.set_xlim(min(ALM_IT_f.get_longitudes())-0.1+dh/2, max(ALM_IT_f.get_longitudes())+0.1+dh/2)
ax_ALM_IT.text(0.74, 0.95, f'{ALM_IT_f.name}', fontsize=18, transform=ax_ALM_IT.transAxes, verticalalignment='top', bbox=props)

    
scatter = ax_ALM_IT.scatter(ALM_IT_f.get_longitudes()+dh/2, ALM_IT_f.get_latitudes()+dh/2,  
                         c = np.log10(ALM_IT_f.spatial_counts()), cmap='inferno', vmin=-4.5, vmax=-0.5,
                         s=8, marker='s', alpha =1, edgecolor="None", zorder=1)

ax_ALM_IT.plot(ptsI[:,0], ptsI[:,1], lw=1, color='black', transform=ccrs.PlateCarree())



ax_EBEL = fig.add_subplot(234, projection=ccrs.PlateCarree(), adjustable='datalim')
ax_EBEL.add_feature(cartopy.feature.STATES, facecolor='None', edgecolor='lightgrey', linewidth=1.5)
ax_EBEL.add_image(_get_basemap('ESRI_imagery'), 6)
gl = ax_EBEL.gridlines()
gl.xlines = False
gl.ylines = False
gl.xlabels_top = True 
gl.xlabels_bottom = False 
gl.ylabels_right = False
gl.ylabels_left = True
gl.xlabel_style = {'size': 13}
gl.ylabel_style = {'size': 13}
gl.xlocator = mticker.FixedLocator([-124,-119, -114])
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([32, 34, 36, 38, 40, 42])
gl.yformatter = LATITUDE_FORMATTER

ax_EBEL.set_ylim(min(EBEL_ET_AL_f.get_latitudes())-0.1+dh/2, max(EBEL_ET_AL_f.get_latitudes())+0.1+dh/2)
ax_EBEL.set_xlim(min(EBEL_ET_AL_f.get_longitudes())-0.1+dh/2, max(EBEL_ET_AL_f.get_longitudes())+0.1+dh/2)
ax_EBEL.text(0.8, 0.95, f'{EBEL_ET_AL_f.name}', fontsize=18, transform=ax_EBEL.transAxes, verticalalignment='top', bbox=props)

    
scatter = ax_EBEL.scatter(EBEL_ET_AL_f.get_longitudes()+dh/2, EBEL_ET_AL_f.get_latitudes()+dh/2, 
                              c = np.log10(EBEL_ET_AL_f.spatial_counts()), cmap='inferno', s=5.1, vmin=-4.5, vmax=-0.5, 
                              marker='s',alpha =1, edgecolor="None", zorder=1)

ax_EBEL.plot(ptsC[:,0], ptsC[:,1], lw=1, color='black', transform=ccrs.PlateCarree(), zorder=2)




ax_SUP = fig.add_subplot(235, projection=ccrs.PlateCarree(), adjustable='datalim')
ax_SUP.add_feature(cartopy.feature.BORDERS, color='lightgrey', linewidth=1.5, zorder=2)
ax_SUP.add_feature(cartopy.feature.COASTLINE, edgecolor='lightgrey', linewidth=1.5, zorder=2)
ax_SUP.add_image(_get_basemap('ESRI_imagery'), 6)
gl = ax_SUP.gridlines()
gl.xlines = False
gl.ylines = False
gl.xlabels_top = True 
gl.xlabels_bottom = False 
gl.ylabels_right = False
gl.ylabels_left = True
gl.xlabel_style = {'size': 13}
gl.ylabel_style = {'size': 13}
gl.xlocator = mticker.FixedLocator([166,170, 174, 178])
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([-35, -38, -41, -44, -47])
gl.yformatter = LATITUDE_FORMATTER

ax_SUP.set_ylim(min(SUP_f.get_latitudes())-0.1+dh/2, max(SUP_f.get_latitudes())+0.1+dh/2)
ax_SUP.set_xlim(min(SUP_f.get_longitudes())-0.1+dh/2, max(SUP_f.get_longitudes())+0.1+dh/2)
ax_SUP.text(0.808, 0.95, f'{SUP_f.name}', fontsize=18, transform=ax_SUP.transAxes, verticalalignment='top', bbox=props)

    
scatter = ax_SUP.scatter(SUP_f.get_longitudes()+dh/2, SUP_f.get_latitudes()+dh/2,  
                           c = np.log10(SUP_f.spatial_counts()), cmap='inferno', s=5, vmin=-5, vmax=0,
                           marker='s', alpha =1, edgecolor="None", zorder=1)

ax_SUP.plot(ptsNZ[:,0], ptsNZ[:,1], lw=1, color='black', transform=ccrs.PlateCarree(), zorder=2)



ax_TRIPLES_CPTI = fig.add_subplot(236, projection=ccrs.PlateCarree(), adjustable='datalim')
ax_TRIPLES_CPTI.add_feature(cartopy.feature.BORDERS, color='lightgrey', linewidth=1.5, zorder=1)
ax_TRIPLES_CPTI.add_feature(cartopy.feature.COASTLINE, edgecolor='lightgrey', linewidth=1.5, zorder=1)
ax_TRIPLES_CPTI.add_image(_get_basemap('ESRI_imagery'), 6)
gl = ax_TRIPLES_CPTI.gridlines()
gl.xlines = False
gl.ylines = False
gl.xlabels_top = True 
gl.xlabels_bottom = False 
gl.ylabels_right = False
gl.ylabels_left = True
gl.xlabel_style = {'size': 13}
gl.ylabel_style = {'size': 13}
gl.xlocator = mticker.FixedLocator([6, 10, 14, 18])
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([37, 39, 41, 43, 45, 47])
gl.yformatter = LATITUDE_FORMATTER

ax_TRIPLES_CPTI.set_ylim(min(TRIPLES_CPTI_f.get_latitudes())-0.1+dh/2, max(TRIPLES_CPTI_f.get_latitudes())+0.1+dh/2)
ax_TRIPLES_CPTI.set_xlim(min(TRIPLES_CPTI_f.get_longitudes())-0.1+dh/2, max(TRIPLES_CPTI_f.get_longitudes())+0.1+dh/2)
ax_TRIPLES_CPTI.text(0.5, 0.95, f'{TRIPLES_CPTI_f.name}', fontsize=18, transform=ax_TRIPLES_CPTI.transAxes, verticalalignment='top', bbox=props)

    
scatter = ax_TRIPLES_CPTI.scatter(TRIPLES_CPTI_f.get_longitudes()+dh/2, TRIPLES_CPTI_f.get_latitudes()+dh/2, 
                              c = np.log10(TRIPLES_CPTI_f.spatial_counts()), cmap='inferno', s=5.1, vmin=-4.5, vmax=-0.5, 
                              marker='s',alpha =1, edgecolor="None", zorder=1)

ax_TRIPLES_CPTI.plot(ptsI[:,0], ptsI[:,1], lw=1, color='black', transform=ccrs.PlateCarree(), zorder=2)

cax = fig.add_axes([0.23, 0.05, 0.555, 0.025]) 
cbar = plt.colorbar(scatter, cax = cax, orientation="horizontal")  
cbar.set_label('Log$_{10}$ of the expected number of M'f'{GEAR1C_f.min_magnitude}+ earthquakes per ' f'{str(dh)}° x {str(dh)}° in 'f'{duration} years', fontsize=18)
cbar.ax.tick_params(labelsize=18)

plt.savefig('./output/TSR_Fig1_bottom.png', dpi=150, bbox_inches = 'tight')


# ### OBSERVATIONS

# In[ ]:


print ('Reading target earthquake catalogs...')


# #### California

# In[99]:


# Earthquake catalog with the whole CSEP-California testing region:
#catalog_Cali1 = csep.query_comcat(start_time=start_date, end_time=end_date, min_magnitude = NEOKINEMA_f.min_magnitude)
#catalog_California = catalog_Cali1.filter_spatial(NEOKINEMA_f.region, in_place=False)


# In[95]:


# The user can download the catalog and then save it:
#with open('ANSS_catalog2021.obj', 'wb') as obj:
 #   pickle.dump(catalog_California, obj)


# In[90]:


with open('./data/ANSS_catalog2021.obj', 'rb') as obj:
    cat = pickle.load(obj)


# In[91]:


# Earthquake catalog within the NEOKINEMA testing region:
catalog_California1 = cat.filter_spatial(GEAR1C_f.region, in_place=False)


# In[92]:


catalog_California1.event_count


# In[93]:


with open('./data/ANSS_catalog2021.obj', 'rb') as obj:
    cat = pickle.load(obj)


# In[94]:


# Earthquake catalog within the PI testing region:
catalog_California2 = cat.filter_spatial(PI_f.region, in_place=False)


# In[95]:


with open('./data/ANSS_catalog2021.obj', 'rb') as obj:
    cat = pickle.load(obj)


# In[96]:


# Earthquake catalog within the EBEL testing region:
catalog_California3 = cat.filter_spatial(EBEL_ET_AL_f.region, in_place=False)


# #### New Zealand

# In[97]:


GeoNet = './data/GeoNet_catalog2021.txt'


# In[98]:


bulk_data = np.loadtxt(GeoNet, skiprows=1, delimiter=' ''\t', dtype='str')
idx = bulk_data[:,0]
origin_time = bulk_data[:,1]
latitude = bulk_data[:,2]
longitude = bulk_data[:,3]
depth = bulk_data[:,13]
magnitude = bulk_data[:,11]


# In[99]:


dt = []
for i in range (len(origin_time)):
    dt.append(str(datetime.datetime.strptime(origin_time[i],'%Y%m%d%H%M%S')))


# In[100]:


dt2 = []

for i in range (len(dt)):
    dt2.append(time_utils.strptime_to_utc_epoch(dt[i]))


# In[101]:


eventlist = np.column_stack([idx, dt2, latitude, longitude, depth, magnitude])


# In[102]:


array_of_tuples = map(tuple, eventlist)
tuple_of_tuples = tuple(array_of_tuples)


# In[103]:


start_date = time_utils.strptime_to_utc_epoch('2014-01-01 00:00:00.0')
end_date = time_utils.strptime_to_utc_epoch('2021-12-31 11:59:59.0')


# In[104]:


catalog_NZ = csep.catalogs.CSEPCatalog(data=tuple_of_tuples)


# In[105]:


catalog_NZ.filter(f'origin_time >= {start_date}')
catalog_NZ.filter(f'origin_time < {end_date}')
catalog_NZ.filter('magnitude >= 4.95')
catalog_NZ.filter('depth < 40.0')
catalog_NZ.filter_spatial(region=NZHM_f.region, update_stats=False, in_place=True)


# In[106]:


catalog_NZ.event_count


# In[107]:


catalog_NZP = csep.catalogs.CSEPCatalog(data=tuple_of_tuples)


# In[108]:


catalog_NZP.filter(f'origin_time >= {start_date}')
catalog_NZP.filter(f'origin_time < {end_date}')
catalog_NZP.filter('magnitude >= 4.95')
catalog_NZP.filter('depth < 40.0')
catalog_NZP.filter_spatial(region=PPE_f.region, update_stats=False, in_place=True)


# #### Italy

# In[109]:


BSI = './data/BSI_catalog2021.dat'


# In[110]:


bulk_data = np.loadtxt(BSI, skiprows=1, delimiter='|', dtype='str')
idx = bulk_data[:,0]
origin_time = bulk_data[:,1]
latitude = bulk_data[:,2]
longitude = bulk_data[:,3]
depth = bulk_data[:,4]
magnitude = bulk_data[:,10]


# In[111]:


dt = []
for i in range (len(origin_time)):
    dt.append(str(datetime.datetime.fromisoformat(origin_time[i])))  


# In[112]:


dt2 = []

for i in range (len(dt)):
    dt2.append(time_utils.strptime_to_utc_epoch(dt[i]))


# In[113]:


eventlist = np.column_stack([idx, dt2, latitude, longitude, depth, magnitude])
array_of_tuples = map(tuple, eventlist)
tuple_of_tuples = tuple(array_of_tuples)


# In[114]:


catalog_Italy = csep.catalogs.CSEPCatalog(data=tuple_of_tuples)


# In[115]:


CSEP_Italy = csep.core.regions.italy_csep_region(dh_scale=1, magnitudes=mws, name='csep-italy')


# In[116]:


catalog_Italy.filter(f'origin_time >= {start_date}')
catalog_Italy.filter(f'origin_time < {end_date}')
catalog_Italy.filter('magnitude >= 4.95')
catalog_Italy.filter('depth < 30.0')
catalog_Italy.filter_spatial(region=GEAR1I_f.region, update_stats=False, in_place=True)


# In[117]:


catalog_Italy.event_count


# In[ ]:


print ('Printing earthquake catalogs on a map (Fig. 2)...')


# In[118]:


fig = plt.figure(figsize=(16.5, 5))

ax_NZ = fig.add_subplot(132, projection=ccrs.PlateCarree(), adjustable='datalim')
ax_NZ.add_feature(cartopy.feature.COASTLINE, edgecolor='grey', linewidth=1)
ax_NZ.add_feature(cartopy.feature.BORDERS, edgecolor='grey', linewidth=1)
ax_NZ.add_image(_get_basemap('ESRI_terrain'), 6)
gl = ax_NZ.gridlines()
gl.xlines = False
gl.ylines = False
gl.xlabels_top = False 
gl.xlabels_bottom = True 
gl.ylabels_right = False
gl.ylabels_left = True
gl.xlabel_style = {'size': 13}
gl.ylabel_style = {'size': 13}
gl.xlocator = mticker.FixedLocator([166,170, 174, 178])
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([-35, -38, -41, -44, -47])
gl.yformatter = LATITUDE_FORMATTER

catalog_sNZ = np.sort(catalog_NZ.data, order=['magnitude']) 

scatter_eNZ = ax_NZ.scatter(catalog_sNZ['longitude'], catalog_sNZ['latitude'], s = 1*2**(catalog_sNZ['magnitude']), 
                edgecolors= 'red', vmin = min(catalog_sNZ['magnitude']), facecolor="None", vmax = max(catalog_sNZ['magnitude']), 
                            alpha =0.8, linewidth=1, marker='s', zorder=2)

handles, labels = scatter_eNZ.legend_elements(prop="sizes", num=5, markerfacecolor="None", markeredgecolor='red', 
                                            alpha=1, zorder=1)

ax_NZ.plot(ptsNZ[:,0], ptsNZ[:,1], lw=1., color='black', transform=ccrs.PlateCarree())

ax_NZ.set_xlim(165.5,179.5)
ax_NZ.set_ylim(-48.2,-33.6)
ax_NZ.text(165.1, -32.6, 'b)', fontsize =22, color='black')

ax_C = fig.add_subplot(131, projection=ccrs.PlateCarree(), adjustable='datalim')
ax_C.add_feature(cartopy.feature.STATES, facecolor='None', edgecolor='grey', linewidth=1)
ax_C.add_image(_get_basemap('ESRI_terrain'), 6)
gl = ax_C.gridlines()
gl.xlines = False
gl.ylines = False
gl.xlabels_top = False 
gl.xlabels_bottom = True 
gl.ylabels_right = False
gl.ylabels_left = True
gl.xlabel_style = {'size': 13}
gl.ylabel_style = {'size': 13}
gl.xlocator = mticker.FixedLocator([-124,-119, -114])
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([32, 34, 36, 38, 40, 42])
gl.yformatter = LATITUDE_FORMATTER

catalog_sC = np.sort(catalog_California1.data, order=['magnitude']) 

scatter_eC = ax_C.scatter(catalog_sC['longitude'], catalog_sC['latitude'], s = 1*2**(catalog_sC['magnitude']), 
                       edgecolors= 'red', vmin = min(catalog_sNZ['magnitude']), facecolor="None", vmax = max(catalog_sNZ['magnitude']), 
                          alpha =0.8, linewidth=1, marker='s', zorder=2)
    


legendC = ax_C.legend(handles, ['5.0', '5.9', '6.9', '7.8'], loc="lower left", edgecolor='black', 
                    labelspacing=1, framealpha=0.5, fontsize=14, facecolor='white')
legendC.set_title('Magnitude',prop={'size':'x-large'})  

ax_C.plot(ptsC[:,0], ptsC[:,1], lw=1., color='black', transform=ccrs.PlateCarree())

ax_C.set_xlim(-125.6,-112.9)
ax_C.set_ylim(31.3,43.2)
ax_C.text(-125.6, 44.5, 'a)', fontsize =22, color='black')




ax_I = fig.add_subplot(133, projection=ccrs.PlateCarree(), adjustable='datalim')
ax_I.add_feature(cartopy.feature.COASTLINE, edgecolor='grey', linewidth=1)
ax_I.add_feature(cartopy.feature.BORDERS, edgecolor='grey', linewidth=1)
ax_I.add_image(_get_basemap('ESRI_terrain'), 6)
gl = ax_I.gridlines()
gl.xlines = False
gl.ylines = False
gl.xlabels_top = False 
gl.xlabels_bottom = True 
gl.ylabels_right = False
gl.ylabels_left = True
gl.xlabel_style = {'size': 13}
gl.ylabel_style = {'size': 13}
gl.xlocator = mticker.FixedLocator([6, 10, 14, 18])
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([36, 38, 40, 42, 44, 46, 48])
gl.yformatter = LATITUDE_FORMATTER

catalog_sI = np.sort(catalog_Italy.data, order=['magnitude']) 

scatter_eI = ax_I.scatter(catalog_sI['longitude'], catalog_sI['latitude'], s = 1*2**(catalog_sI['magnitude']), 
                          edgecolors= 'red', vmin = min(catalog_sNZ['magnitude']), facecolor="None", 
                          vmax = max(catalog_sNZ['magnitude']), alpha =0.8, linewidth=1, marker='s', zorder=2)


ax_I.plot(ptsI[:,0], ptsI[:,1], lw=1., color='black', transform=ccrs.PlateCarree())

ax_I.set_xlim(5.4,19.6)
ax_I.set_ylim(35.7,48)
ax_I.text(5.38, 50, 'c)', fontsize =22, color='black')

plt.savefig('./output/TSR_Fig2.png', dpi=150, bbox_inches = 'tight')


# ### COMPARATIVE EVALUATIONS

# In[ ]:


print ('Running prospective comparisons between global and regional earthquake forecasting models (Fig. 3)...')


# #### California

# In[119]:


ttest_HKJ_GEAR1 = poisson.paired_t_test(HKJ_f, GEAR1C_f, catalog_California1)
ttest_NEOKINEMA_GEAR1 = poisson.paired_t_test(NEOKINEMA_f, GEAR1C_f, catalog_California1)
ttest_PI_GEAR1 = poisson.paired_t_test(PI_f, GEAR1_PI_f, catalog_California2)
ttest_EBEL_GEAR1 = poisson.paired_t_test(EBEL_ET_AL_f, GEAR1_EBEL_f, catalog_California3)


# In[120]:


poisson_TtestsC = [ttest_HKJ_GEAR1, ttest_NEOKINEMA_GEAR1, ttest_PI_GEAR1, ttest_EBEL_GEAR1]


# In[121]:


def plot_comparison_test(results_t, n_models, plot_args=None):
    """Plots list of T-Test or W-Test Results"""
    if plot_args is None:
        plot_args = {}
    title = plot_args.get('title', 'CSEP1 Consistency Test')
    xlabel = plot_args.get('xlabel', 'X')
    ylabel = plot_args.get('ylabel', 'Y')

    fig, ax = plt.subplots(figsize=(n_models,6))
    ax.axhline(y=0, linestyle='--', color='black', linewidth=1)
    ylow = []
    yhigh = []
    for index, result in enumerate(results_t):
        ylow.append(result.observed_statistic - result.test_distribution[0])
        yhigh.append(result.test_distribution[1] - result.observed_statistic)
        if result.observed_statistic - ylow[index] < 0.0 and result.observed_statistic + yhigh[index] < 0.0:
            ax.errorbar(index, result.observed_statistic, yerr=np.array([[ylow[index], yhigh[index]]]).T, color= '#e41a1c', capsize=4) #red
            ax.plot(index, result.observed_statistic, 'ok', marker="o", markersize=15, color='#e41a1c') #red
        elif result.observed_statistic - ylow[index] > 0.0 and result.observed_statistic + yhigh[index] > 0.0:
            ax.errorbar(index, result.observed_statistic, yerr=np.array([[ylow[index], yhigh[index]]]).T, color='#4daf4a', capsize=4)
            ax.plot(index, result.observed_statistic, 'ok', marker="s", markersize=15, color='#4daf4a') #green
        elif result.observed_statistic - ylow[index] <= 0.0 and result.observed_statistic + yhigh[index] >= 0.0:
            ax.errorbar(index, result.observed_statistic, yerr=np.array([[ylow[index], yhigh[index]]]).T, color='#377eb8', capsize=4)
            ax.plot(index, result.observed_statistic, 'ok', marker="^", markersize=15, color='#377eb8') #blue   
        elif result.observed_statistic - ylow[index] >= 0.0 and result.observed_statistic + yhigh[index] <= 0.0:
            ax.errorbar(index, result.observed_statistic, yerr=np.array([[ylow[index], yhigh[index]]]).T, color= '#377eb8', capsize=4)
            ax.plot(index, result.observed_statistic, 'ok', marker="^", markersize=15, color='#377eb8') #blue
        
    
    ax.set_xticklabels([res.sim_name[0] for res in results_t], fontsize=25)
    ax.set_yticklabels([-3, -2, -1, 0, 1, 2, 3],fontsize=22, rotation=90)
    ax.set_xticks(np.arange(len(results_t)))
    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    return ax


# In[122]:


plt.figure()

ax = plot_comparison_test(poisson_TtestsC, 4, plot_args={'xlabel':'', 
                                                       'ylabel': 'Information Gain per Earthquake'})

a= ax.set_xticklabels(['HKJ','NEOKINEMA', 'PI', 'EBEL'], rotation = 90, horizontalalignment='center')

ax.yaxis.tick_right()
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")
ax.yaxis.set_label_position("right")
ax.set_title('')
ax.set_ylim(-3,3)
ax.set_xlim(-0.5,3.5)
xTickPos = ax.get_xticks()
ax.bar(xTickPos, height=10, width=1, bottom=min(ax.get_yticks()), align='center', alpha=0.2, color=['w', 'gray'], zorder=0 )#, data=None, **kwargs)
plt.savefig('./output/TSR_Fig3_California.png', dpi=150, bbox_inches = 'tight')


# #### New Zealand

# In[123]:


ttest_NZHM_GEAR1 = poisson.paired_t_test(NZHM_f, GEAR1NZ_f, catalog_NZ)
ttest_PPE_GEAR1 = poisson.paired_t_test(PPE_f, GEAR1NZ_f, catalog_NZ)
ttest_SUP_GEAR1 = poisson.paired_t_test(SUP_f, GEAR1NZ_f, catalog_NZ)


# In[124]:


poisson_TtestsNZ = [ttest_NZHM_GEAR1, ttest_PPE_GEAR1, ttest_SUP_GEAR1]


# In[125]:


plt.figure()

ax_ct = plot_comparison_test(poisson_TtestsNZ, 3, plot_args={'xlabel':'', 
                                                       'ylabel': 'Information Gain per Earthquake'})

a= ax_ct.set_xticklabels(['NZHM', 'PPE', 'SUP'],
                         rotation = 90, horizontalalignment='center')
    
ax_ct.yaxis.tick_right()
ax_ct.xaxis.tick_top()
ax_ct.xaxis.set_label_position("bottom")
ax_ct.yaxis.set_label_position("right")
ax_ct.set_title('')
ax_ct.set_ylim(-3,3)

xTickPos = ax_ct.get_xticks()
ax_ct.bar(xTickPos, height=10, width=1, bottom=min(ax_ct.get_yticks()), align='center', alpha=0.2, color=['w', 'gray'], zorder=0 )#, data=None, **kwargs)
plt.savefig('./output/TSR_Fig3_NZ.png', dpi=150, bbox_inches = 'tight')


# #### Italy

# In[126]:


ttest_ALM_GEAR1 = poisson.paired_t_test(ALM_f, GEAR1I_f, catalog_Italy)
ttest_HALM_GEAR1 = poisson.paired_t_test(HALM_f, GEAR1I_f, catalog_Italy)
ttest_ALM_IT_GEAR1 = poisson.paired_t_test(ALM_IT_f, GEAR1I_f, catalog_Italy)
ttest_MPS04_AFTER_GEAR1 = poisson.paired_t_test(MPS04_AFTER_f, GEAR1I_f, catalog_Italy)
ttest_HAZGRIDX_GEAR1 = poisson.paired_t_test(HAZGRIDX_f, GEAR1I_f, catalog_Italy)
ttest_HZATI_GEAR1 = poisson.paired_t_test(HZATI_f, GEAR1I_f, catalog_Italy)
ttest_RI_GEAR1 = poisson.paired_t_test(RI_f, GEAR1I_f, catalog_Italy)
ttest_HRSS_CSI_GEAR1 = poisson.paired_t_test(HRSS_CSI_f, GEAR1I_f, catalog_Italy)
ttest_HRSS_HYBRID_GEAR1 = poisson.paired_t_test(HRSS_HYBRID_f, GEAR1I_f, catalog_Italy)
ttest_TRIPLES_CPTI_GEAR1 = poisson.paired_t_test(TRIPLES_CPTI_f, GEAR1I_f, catalog_Italy)
ttest_TRIPLES_CSI_GEAR1 = poisson.paired_t_test(TRIPLES_CSI_f, GEAR1I_f, catalog_Italy)
ttest_TRIPLES_HYBRID_GEAR1 = poisson.paired_t_test(TRIPLES_HYBRID_f, GEAR1I_f, catalog_Italy)


# In[127]:


poisson_TtestsI = [ttest_ALM_GEAR1, ttest_HALM_GEAR1, ttest_ALM_IT_GEAR1,  ttest_MPS04_AFTER_GEAR1, 
                   ttest_HAZGRIDX_GEAR1, ttest_HZATI_GEAR1, ttest_RI_GEAR1, ttest_HRSS_CSI_GEAR1, 
                   ttest_HRSS_HYBRID_GEAR1, ttest_TRIPLES_CPTI_GEAR1, ttest_TRIPLES_CSI_GEAR1, ttest_TRIPLES_HYBRID_GEAR1]


# In[128]:


plt.figure()


ax_ct = plot_comparison_test(poisson_TtestsI, 12, plot_args={'xlabel':'', 
                                                       'ylabel': 'Information Gain per Earthquake', 'fontsize': 18})

a= ax_ct.set_xticklabels(['ALM', 'HALM', 'ALM-IT', 'MPS04-AFTER', 'HAZGRIDX', 'HZATI', 'RI', 'HRSS-CSI',
                         'HRSS-HYBRID', 'TRIPLE_S-CPTI', 'TRIPLE_S-CSI', 'TRIPLE_S-HYBRID'],
                         rotation = 90, horizontalalignment='center')
    
ax_ct.yaxis.tick_right()
ax_ct.xaxis.tick_bottom()
ax_ct.xaxis.set_label_position("bottom")
ax_ct.yaxis.set_label_position("right")
ax_ct.set_title('')
ax_ct.set_ylim(-3,3)
ax_ct.set_xlim(-0.5,11.5)

xTickPos = ax_ct.get_xticks()
ax_ct.bar(xTickPos, height=10, width=1, bottom=min(ax_ct.get_yticks()), align='center', alpha=0.2, color=['w', 'gray'], zorder=0 )#, data=None, **kwargs)
plt.savefig('./output/TSR_Fig3_Italy.png', dpi=150, bbox_inches = 'tight')


# In[130]:


fig = plt.figure(figsize=(15,10))

ax = fig.add_subplot(111, projection=ccrs.Robinson(central_longitude=180))
ax.set_global()
ax.add_feature(cartopy.feature.COASTLINE, linewidth=1, color='white')
gl = ax.gridlines(draw_labels=True, alpha=0.5, color='black', zorder=1)
gl.xlines = False
gl.ylines = False
gl.xlabels_top = False 
gl.xlabels_bottom = False 
gl.ylabels_right = False
gl.ylabels_left = False
gl.xlocator = mticker.FixedLocator([-160, -120, -80, -40, 40, 80, 120, 160])
gl.xformatter = LONGITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75])
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}


scatter = plt.scatter(GEAR1.longitude, GEAR1.latitude, c = np.log10(GEAR1.m595), cmap='inferno', marker='s', 
                      s=5, transform=ccrs.PlateCarree(), zorder=1, vmin=-14.4, vmax=-9.6)

cax = fig.add_axes([0.235, 0.16, 0.555, 0.03]) #left/right #up/down #length #height
cbar = plt.colorbar(scatter, cax = cax, orientation="horizontal")  
cbar.set_label(r'Log$_\mathrm{10}$ of the expected number of M5.95+ earthquakes per m$^2$ per year', fontsize=18)
cbar.ax.tick_params(labelsize=14)
plt.savefig('./output/TSR_Fig3_world.png', dpi=150, bbox_inches = 'tight')


# ### SPATIAL ANALYSIS (Poisson distribution)

# In[ ]:


print ('Analizing the spatial dimension of the forecasts (Fig. 4)...')


# #### California

# In[131]:


stest_GEAR1C = poisson.spatial_test(GEAR1C_f, catalog_California1, seed=seed)
stest_HKJ = poisson.spatial_test(HKJ_f, catalog_California1, seed=seed)
stest_NEOKINEMA = poisson.spatial_test(NEOKINEMA_f, catalog_California1, seed=seed)
stest_PI = poisson.spatial_test(PI_f, catalog_California1, seed=seed)
stest_EBEL = poisson.spatial_test(EBEL_ET_AL_f, catalog_California1, seed=seed)


# In[132]:


jPOLLs_C = [stest_GEAR1C.observed_statistic, stest_HKJ.observed_statistic,stest_NEOKINEMA.observed_statistic, 
            stest_PI.observed_statistic, stest_EBEL.observed_statistic]

jPOLLs_Cc = []

for i in range(len(jPOLLs_C)):
    jPOLLs_Cc.append(jPOLLs_C[i] / catalog_California1.event_count)


# In[133]:


def compute_gini(forecast):
    
    rate = forecast.spatial_counts()
    I = np.argsort(rate)
    fore_norm_sorted = np.cumsum(rate[I]) / np.sum(rate)
    
    """
    Calculates the Gini coefficient of a numpy array,
    based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    """
    
    fore_norm_sorted = fore_norm_sorted.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(fore_norm_sorted) < 0:
        fore_norm_sorted -= np.amin(fore_norm_sorted) #values cannot be negative
    fore_norm_sorted += 0.0000001 #values cannot be 0
    fore_norm_sorted = np.sort(fore_norm_sorted) #values must be sorted
    index = np.arange(1,fore_norm_sorted.shape[0]+1) #index per array element
    n = fore_norm_sorted.shape[0] #number of array elements
    g = ((np.sum((2 * index - n  - 1) * fore_norm_sorted)) / (n * np.sum(fore_norm_sorted))) #Gini coefficient
    return g


# In[134]:


gini_GEAR1C = compute_gini(GEAR1C_f)
gini_HKJ = compute_gini(HKJ_f)
gini_NEOKINEMA = compute_gini(NEOKINEMA_f)
gini_PI = compute_gini(PI_f)
gini_EBEL_ET_AL = compute_gini(EBEL_ET_AL_f)


# In[135]:


GINIS_C = [gini_HKJ, gini_NEOKINEMA, gini_PI, gini_EBEL_ET_AL]
GINIS_C2 = [gini_GEAR1C, gini_HKJ, gini_NEOKINEMA, gini_PI, gini_EBEL_ET_AL]


# #### New Zealand

# In[136]:


stest_GEAR1NZ = poisson.spatial_test(GEAR1NZ_f, catalog_NZ, seed=seed)
stest_NZHM = poisson.spatial_test(NZHM_f, catalog_NZ, seed=seed)
stest_PPE = poisson.spatial_test(PPE_f, catalog_NZ, seed=seed)
stest_SUP = poisson.spatial_test(SUP_f, catalog_NZ,seed=seed)


# In[137]:


jPOLLs_NZ = [stest_GEAR1NZ.observed_statistic, stest_NZHM.observed_statistic, stest_PPE.observed_statistic,
             stest_SUP.observed_statistic]

jPOLLs_NZc = []

for i in range(len(jPOLLs_NZ)):
    jPOLLs_NZc.append(jPOLLs_NZ[i] / catalog_NZ.event_count)


# In[138]:


gini_GEAR1NZ = compute_gini(GEAR1NZ_f)
gini_NZHM = compute_gini(NZHM_f)
gini_PPE = compute_gini(PPE_f)
gini_SUP = compute_gini(SUP_f)


# In[139]:


GINIS_NZ = [gini_NZHM, gini_PPE, gini_SUP]
GINIS_NZ2 = [gini_GEAR1NZ, gini_NZHM, gini_PPE, gini_SUP]


# #### Italy

# In[140]:


stest_GEAR1I = poisson.spatial_test(GEAR1I_f, catalog_Italy, seed=seed)
stest_ALM = poisson.spatial_test(ALM_f, catalog_Italy, seed=seed)
stest_HALM = poisson.spatial_test(HALM_f, catalog_Italy, seed=seed)
stest_ALM_IT = poisson.spatial_test(ALM_IT_f, catalog_Italy, seed=seed)
stest_MPS04_AFTER = poisson.spatial_test(MPS04_AFTER_f, catalog_Italy, seed=seed)
stest_HAZGRIDX = poisson.spatial_test(HAZGRIDX_f, catalog_Italy, seed=seed)
stest_HZATI = poisson.spatial_test(HZATI_f, catalog_Italy, seed=seed)
stest_RI = poisson.spatial_test(RI_f, catalog_Italy, seed=seed)
stest_HRSS_CSI = poisson.spatial_test(HRSS_CSI_f, catalog_Italy, seed=seed)
stest_HRSS_HYBRID = poisson.spatial_test(HRSS_HYBRID_f, catalog_Italy, seed=seed)
stest_TRIPLES_CPTI = poisson.spatial_test(TRIPLES_CPTI_f, catalog_Italy, seed=seed)
stest_TRIPLES_CSI = poisson.spatial_test(TRIPLES_CSI_f, catalog_Italy, seed=seed) 
stest_TRIPLES_HYBRID = poisson.spatial_test(TRIPLES_HYBRID_f, catalog_Italy, seed=seed)


# In[141]:


jPOLLs_I = [stest_GEAR1I.observed_statistic, stest_ALM.observed_statistic, stest_HALM.observed_statistic, 
            stest_ALM_IT.observed_statistic, stest_MPS04_AFTER.observed_statistic, stest_HAZGRIDX.observed_statistic, 
            stest_HZATI.observed_statistic, stest_RI.observed_statistic, stest_HRSS_CSI.observed_statistic,
            stest_HRSS_HYBRID.observed_statistic, stest_TRIPLES_CPTI.observed_statistic,
            stest_TRIPLES_CSI.observed_statistic, stest_TRIPLES_HYBRID.observed_statistic]

jPOLLs_Ic = []

for i in range(len(jPOLLs_I)):
    jPOLLs_Ic.append(jPOLLs_I[i] / catalog_Italy.event_count)


# In[142]:


gini_GEAR1I = compute_gini(GEAR1I_f)
gini_ALM = compute_gini(ALM_f)
gini_HALM = compute_gini(HALM_f)
gini_ALM_IT = compute_gini(ALM_IT_f)
gini_MPS04_AFTER = compute_gini(MPS04_AFTER_f)
gini_HAZGRIDX = compute_gini(HAZGRIDX_f)
gini_HZATI = compute_gini(HZATI_f)
gini_RI = compute_gini(RI_f)
gini_HRSS_CSI = compute_gini(HRSS_CSI_f)
gini_HRSS_HYBRID = compute_gini(HRSS_HYBRID_f)
gini_TRIPLES_CPTI = compute_gini(TRIPLES_CPTI_f)
gini_TRIPLES_CSI = compute_gini(TRIPLES_CSI_f)
gini_TRIPLES_HYBRID = compute_gini(TRIPLES_HYBRID_f)


# In[143]:


GINIS_I = [gini_ALM, gini_HALM, gini_ALM_IT, gini_MPS04_AFTER, gini_HAZGRIDX, gini_HZATI, gini_RI, 
         gini_HRSS_CSI, gini_HRSS_HYBRID, gini_TRIPLES_CPTI, gini_TRIPLES_CSI, gini_TRIPLES_HYBRID]
GINIS_I2 = [gini_GEAR1I, gini_ALM, gini_HALM, gini_ALM_IT, gini_MPS04_AFTER, gini_HAZGRIDX, gini_HZATI, gini_RI, 
         gini_HRSS_CSI, gini_HRSS_HYBRID, gini_TRIPLES_CPTI, gini_TRIPLES_CSI, gini_TRIPLES_HYBRID]


# In[145]:


fig, ax = plt.subplots(figsize=(8,8))

ax.scatter(GINIS_C2, jPOLLs_Cc, s=150, color='#7fc97f', edgecolor='k', label= 'California', marker="h", zorder=2)
ax.scatter(GINIS_NZ2, jPOLLs_NZc, s=150, color='#beaed4', edgecolor='k', label= 'New Zealand', marker="X", zorder=2)
ax.scatter(GINIS_I2, jPOLLs_Ic, s=150, color='#fdc086', edgecolor='k', label = 'Italy', marker="P", zorder=2)
ax.grid(linestyle='--', linewidth='0.8', color='r')

models_C = ['GEAR1', 'HKJ', 'NEOKINEMA', 'PI', 'EBEL']
for i in range(len(models_C)):
    ax.text(GINIS_C2[i] + 0.02, jPOLLs_Cc[i] - 0.01, f'{models_C[i]}', fontsize=12, rotation=0, zorder=1)
    
models_NZ = ['GEAR1', 'NZHM', 'PPE', 'SUP']
for i in range(len(models_NZ)):
     ax.text(GINIS_NZ2[i] - 0.04, jPOLLs_NZc[i] + 0.1, f'{models_NZ[i]}', fontsize=12, rotation=0, zorder=1)
        
models_I = ['GEAR1', 'ALM', 'HALM', 'ALM-IT', 'MPSO4-AFTER', 'HAZGRIDX', 'HZATI', 'RI',
           'HRSS-CSI', 'HRSS-HYBRID', 'TRIPLES-CPTI', 'TRIPLES-CSI', 'TRIPLES-HYBRID']
for i in range(len(models_I)):
     ax.text(GINIS_I2[i] - 0.1, jPOLLs_Ic[i] -0.015, f'{models_I[i]}', fontsize=12, rotation=0, zorder=1)         

x = ['0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']    
y=['','-8.0', '-7.0','-6.0', '-5.0', '4.0']
 
ax.set_xlim(0.3, 1.0)
ax.set_ylim(-8.5, -3.5)
ax.set_xlabel('Gini coefficient', fontsize=15)
ax.set_ylabel('Spatial Joint Likelihood Score per Earthquake', fontsize=15)

ax.set_xticklabels(x, fontsize=13)
ax.set_yticklabels(y, fontsize=13) 
ax.text(0.3, -3.3, 'a)', fontsize=18)
plt.savefig('./output/TSR_Fig4a.png', dpi=150, bbox_inches = 'tight')


# ### SPATIAL ANALYSIS (Binomial distribution)

# In[83]:


def _simulate_catalog(num_events, sampling_weights, sim_fore, random_numbers=None):

    # generate uniformly distributed random numbers in [0,1), this
    if random_numbers is None:
        random_numbers = np.random.rand(num_events)
    else:
        # TODO: ensure that random numbers are all between 0 and 1.
        pass

    # reset simulation array to zero, but don't reallocate
    sim_fore.fill(0)

    # find insertion points using binary search inserting to satisfy a[i-1] <= v < a[i]
    pnts = np.searchsorted(sampling_weights, random_numbers, side='right')

    # create simulated catalog by adding to the original locations
    np.add.at(sim_fore, pnts, 1)
    assert sim_fore.sum() == num_events, "simulated the wrong number of events!"

    return sim_fore


# In[146]:


def _simulate_catalog(num_events, sampling_weights, sim_fore, random_numbers=None):
    #Asim -- Modified this code to generate simulations in a way that every cell gets one earthquake.
    # generate uniformly distributed random numbers in [0,1), this
    if random_numbers is None:
        random_numbers = numpy.random.rand(num_events)
    else:
        # TODO: ensure that random numbers are all between 0 and 1.
        pass

    # reset simulation array to zero, but don't reallocate
    sim_fore.fill(0)
    
    # ---- Asim changes
#    # find insertion points using binary search inserting to satisfy a[i-1] <= v < a[i]
#    pnts = numpy.searchsorted(sampling_weights, random_numbers, side='right')
#
#    # create simulated catalog by adding to the original locations
#    numpy.add.at(sim_fore, pnts, 1)
#    assert sim_fore.sum() == num_events, "simulated the wrong number of events!"
    
    #-- Change the simulation code in such a way that every cells grid only one earthquake.
    eqs = 0
    while eqs < num_events:
            random_num = numpy.random.uniform(0,1)
            loc = numpy.searchsorted(sampling_weights, random_num)
            if sim_fore[loc] == 0:
                numpy.add.at(sim_fore, loc, 1)
                eqs = eqs+1
    
    return sim_fore


# In[147]:


def binomial_joint_log_likelihood_ndarray(forecast, catalog):
    """
    Computes Bernoulli log-likelihood scores, assuming that earthquakes follow a binomial distribution.
    
    Args:
        forecast:   Forecast of a Model (Gridded) (Numpy Array)
                    A forecast has to be in terms of Average Number of Events in Each Bin
                    It can be anything greater than zero
        catalog:    Observed (Gridded) seismicity (Numpy Array):
                    An Observation has to be Number of Events in Each Bin
                    It has to be a either zero or positive integer only (No Floating Point)
    """
    #First, we mask the forecast in cells where we could find log=0.0 singularities:
    forecast_masked = np.ma.masked_where(forecast.ravel() <= 0.0, forecast.ravel()) 
    
    #Then, we compute the log-likelihood of observing one or more events given a Poisson distribution, i.e., 1 - Pr(0) 
    target_idx = np.nonzero(catalog.ravel())
    y = np.zeros(forecast_masked.ravel().shape)
    y[target_idx[0]] = 1
    first_term = y * (np.log(1.0 - np.exp(-forecast_masked.ravel())))
    
    #Also, we estimate the log-likelihood in cells no events are observed:
    second_term = (1-y) * (-forecast_masked.ravel().data)
    #Finally, we sum both terms to compute the joint log-likelihood score:
    return sum(first_term.data + second_term.data)


# In[148]:


def _binomial_likelihood_test(forecast_data, observed_data, num_simulations=1000, random_numbers=None, 
                              seed=None, use_observed_counts=True, verbose=True, normalize_likelihood=False):
    """
    Computes binary conditional-likelihood test from CSEP using an efficient simulation based approach.
    Args:
        forecast_data (numpy.ndarray): nd array where [:, -1] are the magnitude bins.
        observed_data (numpy.ndarray): same format as observation.
        num_simulations: default number of simulations to use for likelihood based simulations
        seed: used for reproducibility of the prng
        random_numbers (numpy.ndarray): can supply an explicit list of random numbers, primarily used for software testing
        use_observed_counts (bool): if true, will simulate catalogs using the observed events, if false will draw from poisson 
        distribution
    """
    
    # Array-masking that avoids log singularities:
    forecast_masked = np.ma.masked_where(forecast_data.ravel() <= 0.0, forecast_data.ravel()) 
    
    # set seed for the likelihood test
    if seed is not None:
        np.random.seed(seed)

    # used to determine where simulated earthquake should be placed, by definition of cumsum these are sorted
    sampling_weights = np.cumsum(forecast_masked.data) / np.sum(forecast_masked.data)

    # data structures to store results
    sim_fore = np.zeros(sampling_weights.shape)
    simulated_ll = []
    n_obs = len(np.unique(np.nonzero(observed_data.ravel())))
    n_fore =np.sum(forecast_masked.data)
    expected_forecast_count = int(n_obs) 
    
    if use_observed_counts and normalize_likelihood:
        scale = n_obs / n_fore
        expected_forecast_count = int(n_obs)
        forecast_data = scale * forecast_masked.data
        

    # main simulation step in this loop
    for idx in range(num_simulations):
        if use_observed_counts:
            num_events_to_simulate = int(n_obs)
        else:
            num_events_to_simulate = int(np.random.poisson(expected_forecast_count))
    
        if random_numbers is None:
            sim_fore = _simulate_catalog(num_events_to_simulate, sampling_weights, sim_fore)
        else:
            sim_fore = _simulate_catalog(num_events_to_simulate, sampling_weights, sim_fore,
                                         random_numbers=random_numbers[idx,:])

    
        # compute joint log-likelihood
        current_ll = binomial_joint_log_likelihood_ndarray(forecast_data, sim_fore)
        
        # append to list of simulated log-likelihoods
        simulated_ll.append(current_ll)

        # just be verbose
        if verbose:
            if (idx + 1) % 100 == 0:
                print(f'... {idx + 1} catalogs simulated.')
                
                target_idx = np.nonzero(catalog.ravel())

    # observed joint log-likelihood
    obs_ll = binomial_joint_log_likelihood_ndarray(forecast_data, observed_data)
        
    # quantile score
    qs = np.sum(simulated_ll <= obs_ll) / num_simulations

    # float, float, list
    return qs, obs_ll, simulated_ll


# In[149]:


def binary_spatial_test(gridded_forecast, observed_catalog, num_simulations=1000, seed=None, random_numbers=None, verbose=False):
    """
    Performs the binary spatial test on the Forecast using the Observed Catalogs.
    Note: The forecast and the observations should be scaled to the same time period before calling this function. This increases
    transparency as no assumptions are being made about the length of the forecasts. This is particularly important for
    gridded forecasts that supply their forecasts as rates.
    Args:
        gridded_forecast: csep.core.forecasts.GriddedForecast
        observed_catalog: csep.core.catalogs.Catalog
        num_simulations (int): number of simulations used to compute the quantile score
        seed (int): used fore reproducibility, and testing
        random_numbers (numpy.ndarray): random numbers used to override the random number generation. injection point for testing.
    Returns:
        evaluation_result: csep.core.evaluations.EvaluationResult
    """

    # grid catalog onto spatial grid
    gridded_catalog_data = observed_catalog.spatial_counts()

    # simply call likelihood test on catalog data and forecast
    qs, obs_ll, simulated_ll = _binomial_likelihood_test(gridded_forecast.spatial_counts(), gridded_catalog_data,
                                                        num_simulations=num_simulations,
                                                        seed=seed,
                                                        random_numbers=random_numbers,
                                                        use_observed_counts=True,
                                                        verbose=verbose, normalize_likelihood=True)

    
# populate result data structure
    result = EvaluationResult()
    result.test_distribution = simulated_ll
    result.name = 'Binary S-Test'
    result.observed_statistic = obs_ll
    result.quantile = qs
    result.sim_name = gridded_forecast.name
    result.obs_name = observed_catalog.name
    result.status = 'normal'
    try:
        result.min_mw = np.min(gridded_forecast.magnitudes)
    except AttributeError:
        result.min_mw = -1
    return result


# #### California

# In[150]:


sbtest_GEAR1C = binary_spatial_test(GEAR1C_f, catalog_California1, seed=seed)
sbtest_HKJ = binary_spatial_test(HKJ_f, catalog_California1, seed=seed)
sbtest_NEOKINEMA = binary_spatial_test(NEOKINEMA_f, catalog_California1, seed=seed)
sbtest_PI = binary_spatial_test(PI_f, catalog_California1, seed=seed)
sbtest_EBEL = binary_spatial_test(EBEL_ET_AL_f, catalog_California1, seed=seed)


# In[151]:


jBILLs_C = [sbtest_GEAR1C.observed_statistic, sbtest_HKJ.observed_statistic,sbtest_NEOKINEMA.observed_statistic, 
            sbtest_PI.observed_statistic, sbtest_EBEL.observed_statistic]

jBILLs_Cc = []

for i in range(len(jBILLs_C)):
    jBILLs_Cc.append(jBILLs_C[i] / catalog_California1.event_count)


# #### New Zealand

# In[152]:


sbtest_GEAR1NZ = binary_spatial_test(GEAR1NZ_f, catalog_NZ, seed=seed)
sbtest_NZHM = binary_spatial_test(NZHM_f, catalog_NZ, seed=seed)
sbtest_PPE = binary_spatial_test(PPE_f, catalog_NZ, seed=seed)
sbtest_SUP = binary_spatial_test(SUP_f, catalog_NZ, seed=seed)


# In[153]:


jBILLs_NZ = [sbtest_GEAR1NZ.observed_statistic, sbtest_NZHM.observed_statistic, sbtest_PPE.observed_statistic,
             sbtest_SUP.observed_statistic]

jBILLs_NZc = []

for i in range(len(jBILLs_NZ)):
    jBILLs_NZc.append(jBILLs_NZ[i] / catalog_NZ.event_count)


# #### Italy

# In[154]:


sbtest_GEAR1I = binary_spatial_test(GEAR1I_f, catalog_Italy, seed=seed)
sbtest_ALM = binary_spatial_test(ALM_f, catalog_Italy, seed=seed)
sbtest_HALM = binary_spatial_test(HALM_f, catalog_Italy, seed=seed)
sbtest_ALM_IT = binary_spatial_test(ALM_IT_f, catalog_Italy, seed=seed)
sbtest_MPS04_AFTER = binary_spatial_test(MPS04_AFTER_f, catalog_Italy, seed=seed)
sbtest_HAZGRIDX = binary_spatial_test(HAZGRIDX_f, catalog_Italy, seed=seed)
sbtest_HZATI = binary_spatial_test(HZATI_f, catalog_Italy, seed=seed)
sbtest_RI = binary_spatial_test(RI_f, catalog_Italy, seed=seed)
sbtest_HRSS_CSI = binary_spatial_test(HRSS_CSI_f, catalog_Italy, seed=seed)
sbtest_HRSS_HYBRID = binary_spatial_test(HRSS_HYBRID_f, catalog_Italy, seed=seed)
sbtest_TRIPLES_CPTI = binary_spatial_test(TRIPLES_CPTI_f, catalog_Italy, seed=seed)
sbtest_TRIPLES_CSI = binary_spatial_test(TRIPLES_CSI_f, catalog_Italy, seed=seed) 
sbtest_TRIPLES_HYBRID = binary_spatial_test(TRIPLES_HYBRID_f, catalog_Italy, seed=seed)


# In[155]:


jBILLs_I = [sbtest_GEAR1I.observed_statistic, sbtest_ALM.observed_statistic, sbtest_HALM.observed_statistic, 
            sbtest_ALM_IT.observed_statistic, sbtest_MPS04_AFTER.observed_statistic, sbtest_HAZGRIDX.observed_statistic, 
            sbtest_HZATI.observed_statistic, sbtest_RI.observed_statistic, sbtest_HRSS_CSI.observed_statistic,
            sbtest_HRSS_HYBRID.observed_statistic, sbtest_TRIPLES_CPTI.observed_statistic,
            sbtest_TRIPLES_CSI.observed_statistic, sbtest_TRIPLES_HYBRID.observed_statistic]

jBILLs_Ic = []

for i in range(len(jBILLs_I)):
    jBILLs_Ic.append(jBILLs_I[i] / catalog_Italy.event_count)


# In[156]:


fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(GINIS_C2, jBILLs_Cc, s=150, color='#7fc97f', edgecolor='k', label= 'California', marker="h")
ax.scatter(GINIS_NZ2, jBILLs_NZc, s=150, color='#beaed4', edgecolor='k', label= 'New Zealand', marker="X")
ax.scatter(GINIS_I2, jBILLs_Ic, s=150, color='#fdc086', edgecolor='k', label = 'Italy', marker="P")
ax.legend(loc=4, fontsize=14, edgecolor='k')
ax.grid(linestyle='--', linewidth='0.8', color='r')
    
x = ['0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']    
y=['','-8.0', '-7.0', '-6.0','-5.0','-4.0']

models_C = ['GEAR1', 'HKJ', 'NEOKINEMA', 'PI', 'EBEL']
for i in range(len(models_C)):
    ax.text(GINIS_C2[i] + 0.02, jBILLs_Cc[i] - 0.01, f'{models_C[i]}', fontsize=12, rotation=0, zorder=1)
    
models_NZ = ['GEAR1', 'NZHM', 'PPE', 'SUP']
for i in range(len(models_NZ)):
     ax.text(GINIS_NZ2[i] - 0.04, jBILLs_NZc[i] + 0.1, f'{models_NZ[i]}', fontsize=12, rotation=0, zorder=1)
        
models_I = ['GEAR1', 'ALM', 'HALM', 'ALM-IT', 'MPSO4-AFTER', 'HAZGRIDX', 'HZATI', 'RI',
            'HRSS-CSI', 'HRSS-HYBRID', 'TRIPLES-CPTI', 'TRIPLES-CSI', 'TRIPLES-HYBRID']
for i in range(len(models_I)):
     ax.text(GINIS_I2[i] - 0.1, jBILLs_Ic[i] -0.015, f'{models_I[i]}', fontsize=12, rotation=0, zorder=1) 
               
ax.set_xlim(0.3, 1.0)
ax.set_ylim(-8.5, -3.5)
ax.set_xlabel('Gini coefficient', fontsize=15)
ax.set_ylabel('Spatial Joint Likelihood Score per Active Cell', fontsize=15)
ax.yaxis.set_label_position('right')
ax.yaxis.tick_right() 
ax.text(0.3, -3.3, 'b)', fontsize=18)
ax.set_xticklabels(x, fontsize=13)
ax.set_yticklabels(y, fontsize=13) 
plt.savefig('./output/TSR_Fig4b.png', dpi=150, bbox_inches = 'tight')


# ### SUPPLEMENTAL MATERIAL

# In[ ]:


print ('Creating supplemental material to the manuscript...')


# In[157]:


def _plot_diff_spatial_likelihood(forecast1, forecast2, catalog, diff_LL, markersizem, marker_sizedLL, bg, lf):
    'This function plots differences between log-likehood scores obtained by two earthquake forecasts,' 
    'given an observed catalog.'
    
    'Arguments:'
    'forecast1: gridded forecast 1'
    'forecast2: gridded forecast 2'
    'catalog: observed catalog'
    'diff_LL: array containing differences in log-likelihood scores, computed in each grid cell'
    'marker_sizem: numerical value controlling the size of the earthquake markers according to magnitude.'
    'In this example, a value of 3.5 is used.'
    'marker_sizedLL: numerical value controlling the size of (delta) log-likelihood markers. In this case study,'
    'a value of 10 is employed.'
    'bg: Size of gray grid cells used to highlight the testing region.'
    'lf: Probability function used to compute log-likelihood scores; P=Poisson, B=binary.'
    
    # We define some plot parameters for the figure:
    ax_dLL = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax_dLL.add_image(_get_basemap('ESRI_imagery'), 6)
    ax_dLL.set_facecolor('lightgrey')
    dh = round(forecast1.region.dh, 5)
    gldLL = ax_dLL.gridlines(draw_labels=True, alpha=0)
    gldLL.xlines = False
    gldLL.ylines = False
    gldLL.ylabels_right = False
    gldLL.xlabel_style = {'size': 13}
    gldLL.ylabel_style = {'size': 13}
    ax_dLL.set_xlim(min(forecast1.get_longitudes())-0.1+dh/2, max(forecast1.get_longitudes())+0.1+dh/2)
    ax_dLL.set_ylim(min(forecast1.get_latitudes())-0.1+dh/2, max(forecast1.get_latitudes())+0.1+dh/2)
    
    
    # We highlight the testing area for visualization purposes:
    ptsNZ = GEAR1NZ_f.region.tight_bbox()
    ax_dLL.plot(ptsNZ[:,0], ptsNZ[:,1], lw=1, color='black', transform=ccrs.PlateCarree(), zorder=2)
    
    
    ax_dLL.plot(ptsNZ[:,0], ptsNZ[:,1], lw=1., color='black', transform=ccrs.PlateCarree(), zorder=1) 
    
    # Here, we sort the catalog to plot earthquakes according to magnitudes:
    catalog_s = np.sort(catalog.data, order=['magnitude']) 
    
    # We plot the earthquake catalog:
    scatter_e = ax_dLL.scatter(catalog_s['longitude'], catalog_s['latitude'], 
                              s = markersizem*2**(catalog_s['magnitude']), 
                edgecolors= 'white', vmin = min(catalog_s['magnitude']), facecolor="None",
                vmax = max(catalog_s['magnitude']), alpha =1, linewidth=1, marker='s', zorder=2)
    
    handles, labels = scatter_e.legend_elements(prop="sizes", num=5, markerfacecolor="None", 
                                                  markeredgecolor='white', alpha=1)

    legend2 = ax_dLL.legend(handles, ['5.0', '5.9', '6.9', '7.8'], loc="lower right", 
                        edgecolor='black', labelspacing=1, framealpha=0.5, fontsize=14, facecolor='white')
    legend2.set_title('Magnitude',prop={'size':'x-large'})  
    
    # We plot the differences in log-likelihood scores:
    scatter_dLL = ax_dLL.scatter(forecast1.get_longitudes() + dh/2, forecast1.get_latitudes() + dh/2, 
                       c=diff_LL, cmap='seismic_r', s= np.abs(marker_sizedLL * diff_LL), vmin=-5, vmax=5, marker='s', zorder=2)

    cax = fig.add_axes([ax_dLL.get_position().x1 + 0.01, ax_dLL.get_position().y0, 0.025, ax_dLL.get_position().height])
    cbar = fig.colorbar(scatter_dLL, cax=cax)
    
    if lf=='P':
        cbar.set_label('Residuals between spatial likelihood scores obtained by 'f'{forecast1.name} and ' f'{forecast2.name}', fontsize=15.5)
    elif lf=='B':
        cbar.set_label('Residuals between BILLs obtained by 'f'{forecast1.name} and ' f'{forecast2.name}', fontsize=15.5)
        
    cbar.ax.tick_params(labelsize='x-large')
    
    return ax_dLL


# In[158]:


BILL_GEAR1 = poisson.binary_spatial_likelihood(GEAR1NZ_f, catalog_NZ)
BILL_KJSS = poisson.binary_spatial_likelihood(KJSSNZ_f, catalog_NZ)
BILL_NZHM = poisson.binary_spatial_likelihood(NZHM_f, catalog_NZ)


# In[159]:


diff_GEAR1_NZHM = BILL_GEAR1 - BILL_NZHM
diff_KJSS_NZHM = BILL_KJSS - BILL_NZHM


# In[160]:


fig = plt.figure(figsize=(10,20))
ax_dBILL_GEAR1_NZHM = _plot_diff_spatial_likelihood(GEAR1NZ_f, NZHM_f, catalog_NZ, diff_GEAR1_NZHM, 3.5, 10, 20, 'P')
ax_dBILL_GEAR1_NZHM.text(165.6, -33., 'a)', fontsize =22, color='black')
plt.savefig('./output/TSR_S1a.png', dpi=150, bbox_inches = 'tight')


# In[161]:


fig = plt.figure(figsize=(10,20))
ax_dBILL_KJSS1_NZHM = _plot_diff_spatial_likelihood(KJSSNZ_f, NZHM_f, catalog_NZ, diff_KJSS_NZHM, 3.5, 10, 20, 'P')
ax_dBILL_KJSS1_NZHM.text(165.6, -33., 'b)', fontsize =22, color='black')
plt.savefig('./output/TSR_S1b.png', dpi=150, bbox_inches = 'tight')


# ### CONSISTENCY TESTS

# #### CALIFORNIA

# #### Poisson Number Test

# In[162]:


ntest_GEAR1 = poisson.number_test(GEAR1C_f, catalog_California1)
ntest_HKJ = poisson.number_test(HKJ_f, catalog_California1)
ntest_EBEL = poisson.number_test(EBEL_ET_AL_f, catalog_California1)
ntest_NEOKINEMA = poisson.number_test(NEOKINEMA_f, catalog_California1)
ntest_PI = poisson.number_test(PI_f, catalog_California1)


# In[163]:


def _get_marker_style(obs_stat, p, one_sided_lower):
    """Returns matplotlib marker style as fmt string"""
    if obs_stat < p[0] or obs_stat > p[1]:
        # red circle
        fmt = 'ko'
    else:
        # green square
        fmt = 'ko'
    if one_sided_lower:
        if obs_stat < p[0]:
            fmt = 'ko'
        else:
            fmt = 'ko'
    return fmt


# In[164]:


def _get_axis_limits(pnts, border=0.05):
    """Returns a tuple of x_min and x_max given points on plot."""
    x_min = numpy.min(pnts)
    x_max = numpy.max(pnts)
    xd = (x_max - x_min)*border
    return (x_min-xd, x_max+xd)


# In[165]:


def plot_consistency_test(eval_results, normalize=False, one_sided_lower=True, plot_args=None, variance=None):
    """ Plots results from CSEP1 tests following the CSEP1 convention.
    Note: All of the evaluations should be from the same type of evaluation, otherwise the results will not be
          comparable on the same figure.
    Args:
        eval_results (list): Contains the tests results :class:`csep.core.evaluations.EvaluationResult` (see note above)
        normalize (bool): select this if the forecast likelihood should be normalized by the observed likelihood. useful
                          for plotting simulation based simulation tests.
        one_sided_lower (bool): select this if the plot should be for a one sided test
        plot_args(dict): optional argument containing a dictionary of plotting arguments, with keys as strings and items as described below
    Optional plotting arguments:
        * figsize: (:class:`list`/:class:`tuple`) - default: [6.4, 4.8]
        * title: (:class:`str`) - default: name of the first evaluation result type
        * title_fontsize: (:class:`float`) Fontsize of the plot title - default: 10
        * xlabel: (:class:`str`) - default: 'X'
        * xlabel_fontsize: (:class:`float`) - default: 10
        * xticks_fontsize: (:class:`float`) - default: 10
        * ylabel_fontsize: (:class:`float`) - default: 10
        * color: (:class:`float`/:class:`None`) If None, sets it to red/green according to :func:`_get_marker_style` - default: 'black'
        * linewidth: (:class:`float`) - default: 1.5
        * capsize: (:class:`float`) - default: 4
        * hbars:  (:class:`bool`)  Flag to draw horizontal bars for each model - default: True
        * tight_layout: (:class:`bool`) Set matplotlib.figure.tight_layout to remove excess blank space in the plot - default: True
    Returns:
        ax (:class:`matplotlib.pyplot.axes` object)
    """


    try:
        results = list(eval_results)
    except TypeError:
        results = [eval_results]
    results.reverse()
    # Parse plot arguments. More can be added here
    if plot_args is None:
        plot_args = {}
    figsize= plot_args.get('figsize', (7,8))
    xlabel = plot_args.get('xlabel', 'X')
    xlabel_fontsize = plot_args.get('xlabel_fontsize', None)
    xticks_fontsize = plot_args.get('xticks_fontsize', None)
    ylabel_fontsize = plot_args.get('ylabel_fontsize', None)
    color = plot_args.get('color', 'black')
    linewidth = plot_args.get('linewidth', None)
    capsize = plot_args.get('capsize', 4)
    hbars = plot_args.get('hbars', True)
    tight_layout = plot_args.get('tight_layout', True)
    percentile = plot_args.get('percentile', 95)

    fig, ax = pyplot.subplots(figsize=figsize)
    xlims = []

    for index, res in enumerate(results):
        # handle analytical distributions first, they are all in the form ['name', parameters].
        if res.test_distribution[0] == 'poisson':
            plow = scipy.stats.poisson.ppf((1 - percentile/100.)/2., res.test_distribution[1])
            phigh = scipy.stats.poisson.ppf(1 - (1 - percentile/100.)/2., res.test_distribution[1])
            observed_statistic = res.observed_statistic

        elif res.test_distribution[0] == 'negative_binomial':
            var = variance
            observed_statistic = res.observed_statistic
            mean = res.test_distribution[1]
            upsilon = 1.0 - ((var - mean) / var)
            tau = (mean**2 /(var - mean))
            phigh = scipy.stats.nbinom.ppf((1 - percentile/100.)/2., tau, upsilon)
            plow = scipy.stats.nbinom.ppf(1 - (1 - percentile/100.)/2., tau, upsilon)

        # empirical distributions
        else:
            if normalize:
                test_distribution = numpy.array(res.test_distribution) - res.observed_statistic
                observed_statistic = 0
            else:
                test_distribution = numpy.array(res.test_distribution)
                observed_statistic = res.observed_statistic
            # compute distribution depending on type of test
            if one_sided_lower:
                plow = numpy.percentile(test_distribution, 5)
                phigh = numpy.percentile(test_distribution, 100)
            else:
                plow = numpy.percentile(test_distribution, 2.5)
                phigh = numpy.percentile(test_distribution, 97.5)

        if not numpy.isinf(observed_statistic): # Check if test result does not diverges
            low = observed_statistic - plow
            high = phigh - observed_statistic
            ax.errorbar(observed_statistic, index, xerr=numpy.array([[low, high]]).T,
                        fmt=_get_marker_style(observed_statistic, (plow, phigh), one_sided_lower),
                        capsize=4, linewidth=linewidth, ecolor=color, markersize = 10, zorder=1)
            # determine the limits to use
            xlims.append((plow, phigh, observed_statistic))
            # we want to only extent the distribution where it falls outside of it in the acceptable tail
            if one_sided_lower:
                if observed_statistic >= plow and phigh < observed_statistic:
                    # draw dashed line to infinity
                    xt = numpy.linspace(phigh, 99999, 100)
                    yt = numpy.ones(100) * index
                    ax.plot(xt, yt, linestyle='--', linewidth=linewidth, color=color)

        else:
            print('Observed statistic diverges for forecast %s, index %i.'
                  ' Check for zero-valued bins within the forecast'% (res.sim_name, index))
            ax.barh(index, 99999, left=-10000, height=1, color=['red'], alpha=0.5)


    try:
        ax.set_xlim(*_get_axis_limits(xlims))
    except ValueError:
        raise ValueError('All EvaluationResults have infinite observed_statistics')
    ax.set_yticks(numpy.arange(len(results)))
    ax.set_yticklabels([res.sim_name for res in results], fontsize=14)
    ax.set_ylim([-0.5, len(results)-0.5])
    if hbars:
        yTickPos = ax.get_yticks()
        if len(yTickPos) >= 2:
            ax.barh(yTickPos, numpy.array([99999] * len(yTickPos)), left=-10000,
                    height=(yTickPos[1] - yTickPos[0]), color=['w', 'gray'], alpha=0.2, zorder=0)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.tick_params(axis='x', labelsize=13)
    if tight_layout:
        ax.figure.tight_layout()
        fig.tight_layout()
    return ax


# In[166]:


def plot_pvalues_and_intervals(test_results, ax, var=None):
    """ Plots p-values and intervals for a list of Poisson or NBD test results
    Args:
        test_results (list): list of EvaluationResults for N-test. All tests should use the same distribution
                             (ie Poisson or NBD).
        ax (matplotlib.axes.Axes.axis): axes to use for plot. create using matplotlib
        var (float): variance of the NBD distribution. Must be used for NBD plots.
    Returns:
        ax (matplotlib.axes.Axes.axis): axes handle containing this plot
    Raises:
        ValueError: throws error if NBD tests are supplied without a variance
    """

    variance = var
    percentile = 97.5
    p_values = []

    # Differentiate between N-tests and other consistency tests
    if test_results[0].name == 'NBD N-Test' or test_results[0].name == 'Poisson N-Test':
        legend_elements = [matplotlib.lines.Line2D([0], [0], marker='o', color='red', lw=0, label=r'p < 10e-5', markersize=10, markeredgecolor='k'),
                           matplotlib.lines.Line2D([0], [0], marker='o', color='#FF7F50', lw=0, label=r'10e-5 $\leq$ p < 10e-4', markersize=10, markeredgecolor='k'),
                           matplotlib.lines.Line2D([0], [0], marker='o', color='gold', lw=0, label=r'10e-4 $\leq$ p < 10e-3', markersize=10, markeredgecolor='k'),
                           matplotlib.lines.Line2D([0], [0], marker='o', color='white', lw=0, label=r'10e-3 $\leq$ p < 0.0125', markersize=10, markeredgecolor='k'),
                           matplotlib.lines.Line2D([0], [0], marker='o', color='skyblue', lw=0, label=r'0.0125 $\leq$ p < 0.025', markersize=10, markeredgecolor='k'),
                           matplotlib.lines.Line2D([0], [0], marker='o', color='blue', lw=0, label=r'p $\geq$ 0.025', markersize=10, markeredgecolor='k')]
        ax.legend(handles=legend_elements, loc=4, fontsize=13, edgecolor='k')
        # Act on Negative binomial tests
        if test_results[0].name == 'NBD N-Test':
            if var is None:
                raise ValueError("var must not be None if N-tests use the NBD distribution.")

            for i in range(len(test_results)):
                mean = test_results[i].test_distribution[1]
                upsilon = 1.0 - ((variance - mean) / variance)
                tau = (mean**2 /(variance - mean))
                phigh97 = scipy.stats.nbinom.ppf((1 - percentile/100.)/2., tau, upsilon)
                plow97 = scipy.stats.nbinom.ppf(1 - (1 - percentile/100.)/2., tau, upsilon)
                low97 = test_results[i].observed_statistic - plow97
                high97 = phigh97 - test_results[i].observed_statistic
                ax.errorbar(test_results[i].observed_statistic, (len(test_results)-1) - i, xerr=numpy.array([[low97, high97]]).T, capsize=4, 
                            color='slategray', alpha=1.0, zorder=0)
                p_values.append(test_results[i].quantile[1] * 2.0) # Calculated p-values according to Meletti et al., (2021)

                if p_values[i] < 10e-5:
                    ax.plot(test_results[i].observed_statistic, (len(test_results)-1) - i, marker='o', color='red', markersize=8, zorder=2)
                if p_values[i] >= 10e-5 and p_values[i] < 10e-4:
                    ax.plot(test_results[i].observed_statistic, (len(test_results)-1) - i, marker='o', color='#FF7F50', markersize=8, zorder=2)
                if p_values[i] >= 10e-4 and p_values[i] < 10e-3:
                    ax.plot(test_results[i].observed_statistic, (len(test_results)-1) - i, marker='o', color='gold', markersize=8, zorder=2)
                if p_values[i] >= 10e-3 and p_values[i] < 0.0125:
                    ax.plot(test_results[i].observed_statistic, (len(test_results)-1) - i, marker='o', color='white', markersize=8, zorder=2)
                if p_values[i] >= 0.0125 and p_values[i] < 0.025:
                    ax.plot(test_results[i].observed_statistic, (len(test_results)-1) - i, marker='o', color='skyblue', markersize=8, zorder=2)
                if p_values[i] >= 0.025:
                    ax.plot(test_results[i].observed_statistic, (len(test_results)-1) - i, marker='o', color='blue', markersize=8, zorder=2)
        # Act on Poisson N-test
        if test_results[0].name == 'Poisson N-Test':
            for i in range(len(test_results)):
                plow97 = scipy.stats.poisson.ppf((1 - percentile/100.)/2., test_results[i].test_distribution[1])
                phigh97 = scipy.stats.poisson.ppf(1 - (1 - percentile/100.)/2., test_results[i].test_distribution[1])
                low97 = test_results[i].observed_statistic - plow97
                high97 = phigh97 - test_results[i].observed_statistic
                ax.errorbar(test_results[i].observed_statistic, (len(test_results)-1) - i, xerr=numpy.array([[low97, high97]]).T, capsize=4, 
                            color='slategray', alpha=1.0, zorder=0)
                p_values.append(test_results[i].quantile[1] * 2.0)
                if p_values[i] < 10e-5:
                    ax.plot(test_results[i].observed_statistic, (len(test_results)-1) - i, marker='o', color='red', markersize=8, zorder=2)
                elif p_values[i] >= 10e-5 and p_values[i] < 10e-4:
                    ax.plot(test_results[i].observed_statistic, (len(test_results)-1) - i, marker='o', color='#FF7F50', markersize=8, zorder=2)
                elif p_values[i] >= 10e-4 and p_values[i] < 10e-3:
                    ax.plot(test_results[i].observed_statistic, (len(test_results)-1) - i, marker='o', color='gold', markersize=8, zorder=2)
                elif p_values[i] >= 10e-3 and p_values[i] < 0.0125:
                    ax.plot(test_results[i].observed_statistic, (len(test_results)-1) - i, marker='o', color='white', markersize=8, zorder=2)
                elif p_values[i] >= 0.0125 and p_values[i] < 0.025:
                    ax.plot(test_results[i].observed_statistic, (len(test_results)-1) - i, marker='o', color='skyblue', markersize=8, zorder=2)
                elif p_values[i] >= 0.025:
                    ax.plot(test_results[i].observed_statistic, (len(test_results)-1) - i, marker='o', color='blue', markersize=8, zorder=2)
    # Operate on all other consistency tests
    else:
        for i in range(len(test_results)):
            plow97 = numpy.percentile(test_results[i].test_distribution, 2.5)
            phigh97 = numpy.percentile(test_results[i].test_distribution, 100)
            low97 = test_results[i].observed_statistic - plow97
            high97 = phigh97 - test_results[i].observed_statistic  
            ax.errorbar(test_results[i].observed_statistic, (len(test_results)-1) -i, xerr=numpy.array([[low97, high97]]).T, capsize=4, 
                        color='slategray', alpha=1.0, zorder=0)
            p_values.append(test_results[i].quantile)

            if p_values[i] < 10e-5:
                ax.plot(test_results[i].observed_statistic, (len(test_results)-1) - i, marker='o', color='red', markersize=8, zorder=2)
            elif p_values[i] >= 10e-5 and p_values[i] < 10e-4:
                ax.plot(test_results[i].observed_statistic, (len(test_results)-1) - i, marker='o', color='#FF7F50', markersize=8, zorder=2)
            elif p_values[i] >= 10e-4 and p_values[i] < 10e-3:
                ax.plot(test_results[i].observed_statistic, (len(test_results)-1) - i, marker='o', color='gold', markersize=8, zorder=2)
            elif p_values[i] >= 10e-3  and p_values[i] < 0.025:
                ax.plot(test_results[i].observed_statistic, (len(test_results)-1) - i, marker='o', color='white', markersize=8, zorder=2)
            elif p_values[i] >= 0.025 and p_values[i] < 0.05:
                ax.plot(test_results[i].observed_statistic, (len(test_results)-1) - i, marker='o', color='skyblue', markersize=8, zorder=2)
            elif p_values[i] >= 0.05:
                ax.plot(test_results[i].observed_statistic, (len(test_results)-1) - i, marker='o', color='blue', markersize=8, zorder=2)

        legend_elements = [
            matplotlib.lines.Line2D([0], [0], marker='o', color='red', lw=0, label=r'p < 10e-5', markersize=10, markeredgecolor='k'),
            matplotlib.lines.Line2D([0], [0], marker='o', color='#FF7F50', lw=0, label=r'10e-5 $\leq$ p < 10e-4', markersize=10, markeredgecolor='k'),
            matplotlib.lines.Line2D([0], [0], marker='o', color='gold', lw=0, label=r'10e-4 $\leq$ p < 10e-3', markersize=10, markeredgecolor='k'),
            matplotlib.lines.Line2D([0], [0], marker='o', color='white', lw=0, label=r'10e-3 $\leq$ p < 0.025', markersize=10, markeredgecolor='k'),
            matplotlib.lines.Line2D([0], [0], marker='o', color='skyblue', lw=0, label=r'0.025 $\leq$ p < 0.05', markersize=10, markeredgecolor='k'),
            matplotlib.lines.Line2D([0], [0], marker='o', color='blue', lw=0, label=r'p $\geq$ 0.05', markersize=10, markeredgecolor='k')]

        ax.legend(handles=legend_elements, loc=4, fontsize=13, edgecolor='k') 

    return ax        


# In[167]:


plt.figure()
    
poisson_Ntests_C = [ntest_GEAR1, ntest_HKJ, ntest_EBEL, ntest_NEOKINEMA, ntest_PI]

ax = plot_consistency_test(poisson_Ntests_C, one_sided_lower=False, plot_args={'xlabel': 'Number of earthquakes'})

ax = plot_pvalues_and_intervals(poisson_Ntests_C, ax)          

ax.text(0, 4.7, 'a)', fontsize =20, color='black')
ax.set_xlim(0,120)
ax.set_title('')
plt.savefig('./output/TSR_poisson_Ntests_California.png', dpi=150, bbox_inches = 'tight')


# #### Negative Binomial Number Test

# In[168]:


rates = pd.read_csv('./data/BND_database1934_8yr.txt', sep='\t', skiprows=0) 
var_ANSS = rates.EQs.var()


# In[169]:


def _nbd_number_test_ndarray(fore_cnt, obs_cnt, variance, epsilon=1e-6):
    """ 
    Computes delta1 and delta2 values from the Negative Binomial (NBD) number test.

    Args:
        fore_cnt (float): parameter of negative binomial distribution coming from expected value of the forecast
        obs_cnt (float): count of earthquakes observed during the testing period.
        variance (float): variance parameter of negative binomial distribution coming from historical catalog. 
        A variance value of approximately 23541 has been calculated using M5.95+ earthquakes observed worldwide from 1982 to 2013.
        epsilon (float): tolerance level to satisfy the requirements of two-sided p-value

    Returns
        result (tuple): (delta1, delta2)
    """
    var = variance
    mean = fore_cnt
    upsilon = 1.0 - ((var - mean) / var)
    tau = (mean**2 /(var - mean))
    
    delta1 = 1.0 - scipy.stats.nbinom.cdf(obs_cnt - epsilon, tau, upsilon, loc=0)
    delta2 = scipy.stats.nbinom.cdf(obs_cnt + epsilon, tau, upsilon, loc=0)

    return delta1, delta2


# In[170]:


def negative_binomial_number_test(gridded_forecast, observed_catalog, variance):
    """
    Computes "negative binomial N-Test" on a gridded forecast.

    Computes Number (N) test for Observed and Forecasts. Both data sets are expected to be in terms of event counts.
    We find the Total number of events in Observed Catalog and Forecasted Catalogs. Which are then employed to compute the 
    probablities of
    (i) At least no. of events (delta 1)
    (ii) At most no. of events (delta 2) assuming the negative binomial distribution.

    Args:
        gridded_forecast:   Forecast of a Model (Gridded) (Numpy Array)
                    A forecast has to be in terms of Average Number of Events in Each Bin
                    It can be anything greater than zero
        observed_catalog:   Observed (Gridded) seismicity (Numpy Array):
                    An Observation has to be Number of Events in Each Bin
                    It has to be a either zero or positive integer only (No Floating Point)
        variance:   Variance parameter of negative binomial distribution obtained from historical catalog.            

    Returns:
        out (tuple): (delta_1, delta_2)
    """
    result = EvaluationResult()

    # observed count
    obs_cnt = observed_catalog.event_count

    # forecasts provide the expeceted number of events during the time horizon of the forecast
    fore_cnt = gridded_forecast.event_count

    epsilon = 1e-6

    # stores the actual result of the number test
    delta1, delta2 = _nbd_number_test_ndarray(fore_cnt, obs_cnt, variance, epsilon=epsilon)
    
    # store results
    result.test_distribution = ('negative_binomial', fore_cnt)
    result.name = 'NBD N-Test'
    result.observed_statistic = obs_cnt
    result.quantile = (delta1, delta2)
    result.sim_name = gridded_forecast.name
    result.obs_name = observed_catalog.name
    result.status = 'normal'
    result.min_mw = numpy.min(gridded_forecast.magnitudes)

    return result


# In[171]:


ntest_GEAR1_nbd = negative_binomial_number_test(GEAR1C_f, catalog_California1, var_ANSS)
ntest_HKJ_nbd = negative_binomial_number_test(HKJ_f, catalog_California1, var_ANSS)
ntest_EBEL_nbd = negative_binomial_number_test(EBEL_ET_AL_f, catalog_California1, var_ANSS)
ntest_NEOKINEMA_nbd = negative_binomial_number_test(NEOKINEMA_f, catalog_California1, var_ANSS)
ntest_PI_nbd = negative_binomial_number_test(PI_f, catalog_California1, var_ANSS)


# In[172]:


plt.figure()

nbd_Ntests_C = [ntest_GEAR1_nbd, ntest_HKJ_nbd, ntest_EBEL_nbd, ntest_NEOKINEMA_nbd, ntest_PI_nbd]

ax = plot_consistency_test(nbd_Ntests_C, one_sided_lower=False, variance=var_ANSS, plot_args={'xlabel': 'Number of earthquakes'})

ax = plot_pvalues_and_intervals(nbd_Ntests_C, ax, var=var_ANSS) 

ax.text(0, 4.7, 'b)', fontsize =20, color='black')
ax.set_xlim(0,120)
ax.set_title('')
ax.yaxis.tick_right()
plt.savefig('./output/TSR_NBD_Ntests_California.png', dpi=150, bbox_inches = 'tight')


# #### Poisson Spatial Test

# In[173]:


plt.figure()
    
poisson_Stests_C = [stest_GEAR1C, stest_HKJ, stest_EBEL, stest_NEOKINEMA, stest_PI]

ax = plot_consistency_test(poisson_Stests_C, one_sided_lower=True, plot_args={'xlabel': 'Log-likelihood (space)'})

ax = plot_pvalues_and_intervals(poisson_Stests_C, ax)     

ax.text(-300, 4.7, 'a)', fontsize =20, color='black')
ax.set_xlim(-300,-100)
ax.set_title('')

plt.savefig('./output/TSR_poisson_Stests_California.png', dpi=150, bbox_inches = 'tight')


# In[174]:


plt.figure()
    
binary_Stests_C = [sbtest_GEAR1C, sbtest_HKJ, sbtest_EBEL, sbtest_NEOKINEMA, sbtest_PI]

ax = plot_consistency_test(binary_Stests_C, one_sided_lower=True, plot_args={'xlabel': 'Log-likelihood (space)'})

ax = plot_pvalues_and_intervals(binary_Stests_C, ax)     

ax.text(-300.0, 4.7, 'b)', fontsize =20, color='black')
ax.set_xlim(-300,-100)
ax.set_title('')

plt.savefig('./output/TSR_binary_Stests_California.png', dpi=150, bbox_inches = 'tight')


# #### Poisson Conditional Likelihood Test

# In[175]:


cltest_GEAR1 = poisson.conditional_likelihood_test(GEAR1C_f, catalog_California1, seed=seed)
cltest_HKJ = poisson.conditional_likelihood_test(HKJ_f, catalog_California1, seed=seed)
cltest_EBEL = poisson.conditional_likelihood_test(EBEL_ET_AL_f, catalog_California1, seed=seed)
cltest_NEOKINEMA = poisson.conditional_likelihood_test(NEOKINEMA_f, catalog_California1, seed=seed)
cltest_PI = poisson.conditional_likelihood_test(PI_f, catalog_California1, seed=seed)


# In[176]:


plt.figure()

poisson_CLtests_C = [cltest_GEAR1, cltest_HKJ, cltest_EBEL, cltest_NEOKINEMA, cltest_PI]

ax = plot_consistency_test(poisson_CLtests_C, one_sided_lower=True, plot_args={'xlabel': 'Log-likelihood'})

ax = plot_pvalues_and_intervals(poisson_CLtests_C, ax) 

ax.text(-420, 4.7, 'a)', fontsize =20, color='black')
ax.set_xlim(-420, -180)
ax.set_title('')
plt.savefig('./output/TSR_poisson_cLtests_California.png', dpi=150, bbox_inches = 'tight')


# #### Binary Conditional Likelihood Test

# In[177]:


def binomial_conditional_likelihood_test(gridded_forecast, observed_catalog, num_simulations=1000, seed=None, random_numbers=None, verbose=False):
    """
    Performs the binary conditional likelihood test on Gridded Forecast using an Observed Catalog.

    Normalizes the forecast so the forecasted rate are consistent with the observations. This modification
    eliminates the strong impact differences in the number distribution have on the forecasted rates.

    Note: The forecast and the observations should be scaled to the same time period before calling this function. This increases
    transparency as no assumptions are being made about the length of the forecasts. This is particularly important for
    gridded forecasts that supply their forecasts as rates.

    Args:
        gridded_forecast: csep.core.forecasts.GriddedForecast
        observed_catalog: csep.core.catalogs.Catalog
        num_simulations (int): number of simulations used to compute the quantile score
        seed (int): used fore reproducibility, and testing
        random_numbers (numpy.ndarray): random numbers used to override the random number generation. injection point for testing.

    Returns:
        evaluation_result: csep.core.evaluations.EvaluationResult
    """
        
    # grid catalog onto spatial grid
    try:
        _ = observed_catalog.region.magnitudes
    except CSEPCatalogException:
        observed_catalog.region = gridded_forecast.region
    gridded_catalog_data = observed_catalog.spatial_magnitude_counts()

    # simply call likelihood test on catalog data and forecast
    qs, obs_ll, simulated_ll = _binomial_likelihood_test(gridded_forecast.data, gridded_catalog_data,
                                                        num_simulations=num_simulations, seed=seed, random_numbers=random_numbers,
                                                        use_observed_counts=True,
                                                        verbose=verbose, normalize_likelihood=False)

    # populate result data structure
    result = EvaluationResult()
    result.test_distribution = simulated_ll
    result.name = 'Binary CL-Test'
    result.observed_statistic = obs_ll
    result.quantile = qs
    result.sim_name = gridded_forecast.name
    result.obs_name = observed_catalog.name
    result.status = 'normal'
    result.min_mw = numpy.min(gridded_forecast.magnitudes)

    return result


# In[178]:


clbtest_GEAR1 = binomial_conditional_likelihood_test(GEAR1C_f, catalog_California1, seed=seed)
clbtest_HKJ = binomial_conditional_likelihood_test(HKJ_f, catalog_California1, seed=seed)
clbtest_EBEL = binomial_conditional_likelihood_test(EBEL_ET_AL_f, catalog_California1, seed=seed)
clbtest_NEOKINEMA = binomial_conditional_likelihood_test(NEOKINEMA_f, catalog_California1, seed=seed)
clbtest_PI = binomial_conditional_likelihood_test(PI_f, catalog_California1, seed=seed)


# In[179]:


plt.figure()

binary_CLtests_C = [clbtest_GEAR1, clbtest_HKJ, clbtest_EBEL, clbtest_NEOKINEMA, clbtest_PI]

ax = plot_consistency_test(binary_CLtests_C, one_sided_lower=True, plot_args={'xlabel': 'Log-likelihood'})

ax = plot_pvalues_and_intervals(binary_CLtests_C, ax) 

ax.yaxis.tick_right()
ax.set_xlim(-420, -180)
ax.set_title('')
ax.text(-420, 4.7, 'b)', fontsize =20, color='black')
plt.savefig('./output/TSR_binary_cLtests_California.png', dpi=150, bbox_inches = 'tight')


# ### NEW ZEALAND

# #### Poisson Number Test

# In[180]:


ntest_GEAR1_NZ = poisson.number_test(GEAR1NZ_f, catalog_NZ)
ntest_NZHM = poisson.number_test(NZHM_f, catalog_NZ)
ntest_PPE = poisson.number_test(PPE_f, catalog_NZ)
ntest_SUP = poisson.number_test(SUP_f, catalog_NZ)


# In[181]:


plt.figure()
    
poisson_Ntests_NZ = [ntest_GEAR1_NZ, ntest_NZHM, ntest_PPE, ntest_SUP]

ax = plot_consistency_test(poisson_Ntests_NZ, one_sided_lower=False, plot_args={'xlabel': 'Number of earthquakes'})

ax = plot_pvalues_and_intervals(poisson_Ntests_NZ, ax)          

ax.text(0, 3.7, 'a)', fontsize =20, color='black')
ax.set_xlim(0,100)
ax.set_title('')
plt.savefig('./output/TSR_poisson_Ntests_NZ.png', dpi=150, bbox_inches = 'tight')


# #### NBD Number Test

# In[182]:


rates = pd.read_csv('./data/BND_database1942_8yr.txt', sep='\t', skiprows=0) 
var_GeoNet = rates.EQs.var()


# In[183]:


ntest_GEAR1NZ_nbd = negative_binomial_number_test(GEAR1NZ_f, catalog_NZ, var_GeoNet)
ntest_NZHM_nbd = negative_binomial_number_test(NZHM_f, catalog_NZ, var_GeoNet)
ntest_PPE_nbd = negative_binomial_number_test(PPE_f, catalog_NZ, var_GeoNet)
ntest_SUP_nbd = negative_binomial_number_test(SUP_f, catalog_NZ, var_GeoNet)


# In[184]:


plt.figure()

nbd_Ntests_NZ = [ntest_GEAR1NZ_nbd, ntest_NZHM_nbd, ntest_PPE_nbd, ntest_SUP_nbd]

ax = plot_consistency_test(nbd_Ntests_NZ, one_sided_lower=False, variance=var_GeoNet, plot_args={'xlabel': 'Number of earthquakes'})

ax = plot_pvalues_and_intervals(nbd_Ntests_NZ, ax, var=var_GeoNet) 

ax.text(0, 3.7, 'b)', fontsize =20, color='black')
ax.set_xlim(0,100)
ax.set_title('')
ax.yaxis.tick_right()
plt.savefig('./output/TSR_NBD_Ntests_NZ.png', dpi=150, bbox_inches = 'tight')


# #### Poisson Spatial Test

# In[185]:


stest_GEAR1NZ = poisson.spatial_test(GEAR1NZ_f, catalog_NZ, seed=seed)
stest_NZHM = poisson.spatial_test(NZHM_f, catalog_NZ, seed=seed)
stest_PPE = poisson.spatial_test(PPE_f, catalog_NZ, seed=seed)
stest_SUP = poisson.spatial_test(SUP_f, catalog_NZ,seed=seed)


# In[186]:


plt.figure()
    
poisson_Stests_NZ = [stest_GEAR1NZ, stest_NZHM, stest_PPE, stest_SUP]

ax = plot_consistency_test(poisson_Stests_NZ, plot_args={'xlabel': 'Log-likelihood (space)'})

ax = plot_pvalues_and_intervals(poisson_Stests_NZ, ax)     

ax.text(-285, 3.7, 'a)', fontsize =20, color='black')
ax.set_xlim(-285,-175)
ax.set_title('')

plt.savefig('./output/TSR_poisson_Stests_NZ.png', dpi=150, bbox_inches = 'tight')


# #### Binary Spatial Test

# In[187]:


plt.figure()

binary_Stests_NZ = [sbtest_GEAR1NZ, sbtest_NZHM, sbtest_PPE, sbtest_SUP]

ax = plot_consistency_test(binary_Stests_NZ, one_sided_lower=True, plot_args={'xlabel': 'Log-likelihood (space)'})

ax = plot_pvalues_and_intervals(binary_Stests_NZ, ax) 

ax.yaxis.tick_right()  
ax.text(-285, 3.7, 'b)', fontsize =20, color='black')
ax.set_xlim(-285,-175)
ax.set_title('')
#plt.savefig('./output/TSR_binary_Stests_NZ.png', dpi=150, bbox_inches = 'tight')


# #### Poisson Conditional Likelihood Test

# In[188]:


cltest_GEAR1NZ = poisson.conditional_likelihood_test(GEAR1NZ_f, catalog_NZ, seed=seed)
cltest_NZHM = poisson.conditional_likelihood_test(NZHM_f, catalog_NZ, seed=seed)
cltest_PPE = poisson.conditional_likelihood_test(PPE_f, catalog_NZ, seed=seed)
cltest_SUP = poisson.conditional_likelihood_test(SUP_f, catalog_NZ, seed=seed)


# In[189]:


plt.figure()

poisson_CLtests_NZ = [cltest_GEAR1NZ, cltest_NZHM, cltest_PPE, cltest_SUP]

ax = plot_consistency_test(poisson_CLtests_NZ, plot_args={'xlabel': 'Log-likelihood'})

ax = plot_pvalues_and_intervals(poisson_CLtests_NZ, ax) 

ax.text(-420, 3.7, 'a)', fontsize =20, color='black')
ax.set_xlim(-420, -280)
ax.set_title('')
plt.savefig('./output/TSR_poisson_cLtests_NZ.png', dpi=150, bbox_inches = 'tight')


# #### Binary Conditional Likelihood Test

# In[190]:


clbtest_GEAR1NZ = binomial_conditional_likelihood_test(GEAR1NZ_f, catalog_NZ, seed=seed)
clbtest_NZHM = binomial_conditional_likelihood_test(NZHM_f, catalog_NZ, seed=seed)
clbtest_PPE = binomial_conditional_likelihood_test(PPE_f, catalog_NZ, seed=seed)
clbtest_SUP = binomial_conditional_likelihood_test(SUP_f, catalog_NZ, seed=seed)


# In[191]:


plt.figure()

binary_CLtests_NZ = [clbtest_GEAR1NZ, clbtest_NZHM, clbtest_PPE, clbtest_SUP]

ax = plot_consistency_test(binary_CLtests_NZ, plot_args={'xlabel': 'Log-likelihood'})

ax = plot_pvalues_and_intervals(binary_CLtests_NZ, ax) 

ax.yaxis.tick_right()
ax.set_xlim(-420, -280)
ax.text(-415, 3.7, 'b)', fontsize =20, color='black')
ax.set_title('')
plt.savefig('./output/TSR_binary_cLtests_NZ.png', dpi=150, bbox_inches = 'tight')


# #### ITALY

# #### Poisson Number Test

# In[192]:


ntest_GEAR1I = poisson.number_test(GEAR1I_f, catalog_Italy)
ntest_ALM = poisson.number_test(ALM_f, catalog_Italy)
ntest_HALM = poisson.number_test(HALM_f, catalog_Italy)
ntest_ALM_IT = poisson.number_test(ALM_IT_f, catalog_Italy)
ntest_MPS04_AFTER = poisson.number_test(MPS04_AFTER_f, catalog_Italy)
ntest_HAZGRIDX = poisson.number_test(HAZGRIDX_f, catalog_Italy)
ntest_HZATI = poisson.number_test(HZATI_f, catalog_Italy)
ntest_RI = poisson.number_test(RI_f, catalog_Italy)
ntest_HRSS_CSI = poisson.number_test(HRSS_CSI_f, catalog_Italy)
ntest_HRSS_HYBRID = poisson.number_test(HRSS_HYBRID_f, catalog_Italy)
ntest_TRIPLES_CPTI = poisson.number_test(TRIPLES_CPTI_f, catalog_Italy)
ntest_TRIPLES_CSI = poisson.number_test(TRIPLES_CSI_f, catalog_Italy) 
ntest_TRIPLES_HYBRID = poisson.number_test(TRIPLES_HYBRID_f, catalog_Italy)


# In[193]:


plt.figure()
    
poisson_Ntests_I = [ntest_GEAR1I, ntest_ALM, ntest_HALM, ntest_ALM_IT, ntest_MPS04_AFTER, ntest_HAZGRIDX,
                 ntest_HZATI, ntest_RI, ntest_HRSS_CSI, ntest_HRSS_HYBRID, ntest_TRIPLES_CPTI, 
                 ntest_TRIPLES_CSI, ntest_TRIPLES_HYBRID]

ax = plot_consistency_test(poisson_Ntests_I, one_sided_lower=False, plot_args={'xlabel': 'Number of earthquakes'})

ax = plot_pvalues_and_intervals(poisson_Ntests_I, ax)          

ax.text(0, 13, 'a)', fontsize =20, color='black')
ax.set_xlim(0,40)
plt.savefig('./output/TSR_poisson_Ntests_Italy.png', dpi=150, bbox_inches = 'tight')


# #### NBD Number Test

# In[194]:


rates = pd.read_csv('./data/BND_database1990_8yr.txt', sep='\t', skiprows=0) 
var_BSI = rates.EQs.var()


# In[195]:


ntest_GEAR1I_nbd = negative_binomial_number_test(GEAR1I_f, catalog_Italy, var_BSI)
ntest_ALM_nbd = negative_binomial_number_test(ALM_f, catalog_Italy, var_BSI)
ntest_HALM_nbd = negative_binomial_number_test(HALM_f, catalog_Italy, var_BSI)
ntest_ALM_IT_nbd = negative_binomial_number_test(ALM_IT_f, catalog_Italy,var_BSI)
ntest_MPS04_AFTER_nbd = negative_binomial_number_test(MPS04_AFTER_f, catalog_Italy,var_BSI)
ntest_HAZGRIDX_nbd = negative_binomial_number_test(HAZGRIDX_f, catalog_Italy, var_BSI)
ntest_HZATI_nbd = negative_binomial_number_test(HZATI_f, catalog_Italy, var_BSI)
ntest_RI_nbd = negative_binomial_number_test(RI_f, catalog_Italy, var_BSI)
ntest_HRSS_CSI_nbd = negative_binomial_number_test(HRSS_CSI_f, catalog_Italy, var_BSI)
ntest_HRSS_HYBRID_nbd = negative_binomial_number_test(HRSS_HYBRID_f, catalog_Italy, var_BSI)
ntest_TRIPLES_CPTI_nbd = negative_binomial_number_test(TRIPLES_CPTI_f, catalog_Italy, var_BSI)
ntest_TRIPLES_CSI_nbd = negative_binomial_number_test(TRIPLES_CSI_f, catalog_Italy, var_BSI) 
ntest_TRIPLES_HYBRID_nbd = negative_binomial_number_test(TRIPLES_HYBRID_f, catalog_Italy, var_BSI)


# In[196]:


plt.figure()

nbd_Ntests_I = [ntest_GEAR1I_nbd, ntest_ALM_nbd, ntest_HALM_nbd, ntest_ALM_IT_nbd, ntest_MPS04_AFTER_nbd, 
              ntest_HAZGRIDX_nbd, ntest_HZATI_nbd, ntest_RI_nbd, ntest_HRSS_CSI_nbd, ntest_HRSS_HYBRID_nbd,
              ntest_TRIPLES_CPTI_nbd, ntest_TRIPLES_CSI_nbd, ntest_TRIPLES_HYBRID_nbd]

ax = plot_consistency_test(nbd_Ntests_I, one_sided_lower=False, variance=var_BSI, plot_args={'xlabel': 'Number of earthquakes'})

ax = plot_pvalues_and_intervals(nbd_Ntests_I, ax, var=var_BSI) 

ax.yaxis.tick_right()
ax.text(0, 13, 'b)', fontsize =20, color='black')
ax.set_xlim(0,40)
plt.savefig('./output/TSR_NBD_Ntests_Italy.png', dpi=150, bbox_inches = 'tight')


# #### Poisson Spatial Test

# In[197]:


plt.figure()
    
poisson_Stests_I = [stest_GEAR1I, stest_ALM, stest_HALM, stest_ALM_IT, stest_MPS04_AFTER, stest_HAZGRIDX, 
                  stest_HZATI, stest_RI, stest_HRSS_CSI, stest_HRSS_HYBRID, stest_TRIPLES_CPTI, 
                  stest_TRIPLES_CSI, stest_TRIPLES_HYBRID]

ax = plot_consistency_test(poisson_Stests_I, one_sided_lower=True, plot_args={'xlabel': 'Log-likelihood (space)'})

ax = plot_pvalues_and_intervals(poisson_Stests_I, ax)     

ax.text(-100, 13, 'a)', fontsize =20, color='black')
ax.set_xlim(-100,0)
ax.set_title('')

plt.savefig('./output/TSR_poisson_Stest_Italy.png', dpi=150, bbox_inches = 'tight')


# In[198]:


plt.figure()

binary_Stests_I = [sbtest_GEAR1I, sbtest_ALM, sbtest_HALM, sbtest_ALM_IT, sbtest_MPS04_AFTER, 
                sbtest_HAZGRIDX, sbtest_HZATI, sbtest_RI, sbtest_HRSS_CSI, sbtest_HRSS_HYBRID,
                sbtest_TRIPLES_CPTI, sbtest_TRIPLES_CSI, sbtest_TRIPLES_HYBRID]

ax = plot_consistency_test(binary_Stests_I, one_sided_lower=True, plot_args={'xlabel': 'Log-likelihood (space)'})

ax = plot_pvalues_and_intervals(binary_Stests_I, ax) 

ax.yaxis.tick_right()  
ax.set_xlim(-100,0)
ax.set_title('')
ax.text(-100.0, 13, 'b)', fontsize =20, color='black')
#plt.savefig('./output/TSR_binary_Stests_Italy.png', dpi=150, bbox_inches = 'tight')


# #### Poisson Conditional Likelihood Test

# In[199]:


cltest_GEAR1I = poisson.conditional_likelihood_test(GEAR1I_f, catalog_Italy, seed=seed)
cltest_ALM = poisson.conditional_likelihood_test(ALM_f, catalog_Italy, seed=seed)
cltest_HALM = poisson.conditional_likelihood_test(HALM_f, catalog_Italy, seed=seed)
cltest_ALM_IT = poisson.conditional_likelihood_test(ALM_IT_f, catalog_Italy, seed=seed)
cltest_MPS04_AFTER = poisson.conditional_likelihood_test(MPS04_AFTER_f, catalog_Italy, seed=seed)
cltest_HAZGRIDX = poisson.conditional_likelihood_test(HAZGRIDX_f, catalog_Italy, seed=seed)
cltest_HZATI = poisson.conditional_likelihood_test(HZATI_f, catalog_Italy, seed=seed)
cltest_RI = poisson.conditional_likelihood_test(RI_f, catalog_Italy, seed=seed)
cltest_HRSS_CSI = poisson.conditional_likelihood_test(HRSS_CSI_f, catalog_Italy, seed=seed)
cltest_HRSS_HYBRID = poisson.conditional_likelihood_test(HRSS_HYBRID_f, catalog_Italy, seed=seed)
cltest_TRIPLES_CPTI = poisson.conditional_likelihood_test(TRIPLES_CPTI_f, catalog_Italy, seed=seed)
cltest_TRIPLES_CSI = poisson.conditional_likelihood_test(TRIPLES_CSI_f, catalog_Italy, seed=seed) 
cltest_TRIPLES_HYBRID = poisson.conditional_likelihood_test(TRIPLES_HYBRID_f, catalog_Italy, seed=seed)


# In[200]:


plt.figure()

poisson_CLtests_I = [cltest_GEAR1I, cltest_ALM, cltest_HALM, cltest_ALM_IT, cltest_MPS04_AFTER, cltest_HAZGRIDX, 
                  cltest_HZATI, cltest_RI, cltest_HRSS_CSI, cltest_HRSS_HYBRID, cltest_TRIPLES_CPTI, 
                  cltest_TRIPLES_CSI, cltest_TRIPLES_HYBRID]

ax = plot_consistency_test(poisson_CLtests_I, plot_args={'xlabel': 'Log-likelihood'})

ax = plot_pvalues_and_intervals(poisson_CLtests_I, ax) 

ax.text(-130, 13, 'a)', fontsize =20, color='black')
ax.set_xlim(-130, -40)
ax.set_title('')
plt.savefig('./output/TSR_poisson_cLtests_Italy.png', dpi=150, bbox_inches = 'tight')


# #### Binary Conditional Likelihood Test

# In[201]:


clbtest_GEAR1 = binomial_conditional_likelihood_test(GEAR1I_f, catalog_Italy, seed=seed)
clbtest_ALM = binomial_conditional_likelihood_test(ALM_f, catalog_Italy, seed=seed)
clbtest_HALM = binomial_conditional_likelihood_test(HALM_f, catalog_Italy, seed=seed)
clbtest_ALM_IT = binomial_conditional_likelihood_test(ALM_IT_f, catalog_Italy, seed=seed)
clbtest_MPS04_AFTER = binomial_conditional_likelihood_test(MPS04_AFTER_f, catalog_Italy, seed=seed)
clbtest_HAZGRIDX = binomial_conditional_likelihood_test(HAZGRIDX_f, catalog_Italy, seed=seed)
clbtest_HZATI = binomial_conditional_likelihood_test(HZATI_f, catalog_Italy, seed=seed)
clbtest_RI = binomial_conditional_likelihood_test(RI_f, catalog_Italy, seed=seed)
clbtest_HRSS_CSI = binomial_conditional_likelihood_test(HRSS_CSI_f, catalog_Italy, seed=seed)
clbtest_HRSS_HYBRID = binomial_conditional_likelihood_test(HRSS_HYBRID_f, catalog_Italy, seed=seed)
clbtest_TRIPLES_CPTI = binomial_conditional_likelihood_test(TRIPLES_CPTI_f, catalog_Italy, seed=seed)
clbtest_TRIPLES_CSI = binomial_conditional_likelihood_test(TRIPLES_CSI_f, catalog_Italy, seed=seed) 
clbtest_TRIPLES_HYBRID = binomial_conditional_likelihood_test(TRIPLES_HYBRID_f, catalog_Italy, seed=seed)


# In[202]:


plt.figure()

binary_CLtests_I = [clbtest_GEAR1, cltest_ALM, cltest_HALM, cltest_ALM_IT, cltest_MPS04_AFTER, 
                 clbtest_HAZGRIDX, cltest_HZATI, cltest_RI, cltest_HRSS_CSI, cltest_HRSS_HYBRID,
                 clbtest_TRIPLES_CPTI, cltest_TRIPLES_CSI, cltest_TRIPLES_HYBRID]

ax = plot_consistency_test(binary_CLtests_I, plot_args={'xlabel': 'Log-likelihood'})

ax = plot_pvalues_and_intervals(binary_CLtests_I, ax) 

ax.yaxis.tick_right()  
ax.text(-130, 13, 'b)', fontsize =20, color='black')
ax.set_xlim(-130, -40)
ax.set_title('')
plt.savefig('./output/TSR_binary_cLtests_Italy.png', dpi=150, bbox_inches = 'tight')


# In[113]:


print("All done in %s seconds ;o)" %(time.time() -t0))

