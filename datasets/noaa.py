import os
import pandas as pds
import xarray as xr
import json
from datetime import datetime, timedelta
from matplotlib.dates import num2date, date2num
import matplotlib.dates as mdates
import numpy as np
import logging
import h5py
import scipy
import pickle 

def read_archive_nc(path):
    """Read all netcdf files from a local path and convert them to a merged pandas dataframe.

    Args:
        path (str): The local path where the netcdf files are located.

    Returns:
        pandas.DataFrame: A merged pandas dataframe containing all the data from the netcdf files.
    """
    items = os.listdir(path)
    
    pla_list, mag_list = [], []
    for name in items:
        if 'm1m' in name and name.endswith(".nc"):
            mag_list.append(name)
        if 'f1m' in name and name.endswith(".nc"):
            pla_list.append(name)
            
    pla_list.sort()
    mag_list.sort()
    
    pla_keys = ['time', 'proton_density', 'proton_speed', 'proton_temperature']
    mag_keys = ['time', 'bx_gsm', 'by_gsm', 'bz_gsm', 'bt']
    
    pla_sets = [read_single_nc(os.path.join(path, file), pla_keys) for file in pla_list]
    mag_sets = [read_single_nc(os.path.join(path, file), mag_keys) for file in mag_list]
    
    df_pla = pds.concat(pla_sets)
    df_mag = pds.concat(mag_sets)
    df = pds.concat([df_mag,df_pla], axis=1)
    df.to_pickle('datasets/files/rtsw_all.p')
    return df_pla, df_mag, df

def read_rtsw_archive_data(filepath):
    
    """Reads the archived h5 data.

    Parameters
    ==========
    json_path : str
        String of directory containing data files.

    Returns
    =======
    DataFile

    Example
    =======
    >>> read_and_process_json("rtswdata")
    """

    logging.info('Reading all archived json data as pandas DataFrame.')
    
    hf = h5py.File(filepath, 'r')
    keys = ['time', 'bx_gsm', 'by_gsm', 'bz_gsm', 'bt', 'density', 'speed', 'temperature']
    
    df = pds.DataFrame(columns = keys)
    
    for k in keys:
        df[k] = np.array(hf[k])
    
    df['time'] = num2date(df['time'], tz=None) 
    df['time'] = pds.to_datetime(df['time'], format="%Y/%m/%d %H:%M")
    df.set_index('time',  inplace=True)
    df.index.name = None
    df.index = df.index.tz_localize(None)
    
    return df

def read_single_nc(path, keys):
    ds = xr.open_dataset(path)
    df = pds.DataFrame(columns = keys)
    
    for k in keys:
        df[k] = np.array(ds[k])
    
    df.set_index('time',  inplace=True)
    df.index.name = None
    
    return df

def read_noaa_rtsw_json_single(json_file, timef="%Y-%m-%d %H:%M:%S.%f"):
    """Reads NOAA real-time solar wind data JSON files (already downloaded).

    Parameters
    ==========
    json_file : str
        String of direct path to plasma data file.

    Returns
    =======
    rtsw_data : np.array
        Numpy array with JSON keys accessible as keys or under rtsw_data.dtype.names.

    Example
    =======
    >>> json_file = 'data/plasma-7-day_2020_Mar_28_17_00.json'
    >>> pla_data = read_noaa_rtsw_json(json_file)
    """

    # Read JSON file:
    with open(json_file, 'r') as jdata:
        dp = json.loads(jdata.read())
        dpn = [[np.nan if x == None else x for x in d] for d in dp]     # Replace None w NaN
        dtype=[(x, 'float') for x in dp[0]]
        datesp = [datetime.strptime(x[0], timef)  for x in dpn[1:]]
        #convert datetime to matplotlib times
        mdatesp = date2num(datesp)
        dp_ = [tuple([d]+[float(y) for y in x[1:]]) for d, x in zip(mdatesp, dpn[1:])]
        rtsw_data = np.array(dp_, dtype=dtype)

    return rtsw_data
    
    
def read_and_process_json(json_path):
    """Reads the json data.

    Parameters
    ==========
    json_path : str
        String of directory containing data files.

    Returns
    =======
    DataFile

    Example
    =======
    >>> read_and_process_json("rtswdata")
    """

    logging.info('Reading all archived json data as pandas DataFrame.')

    items = os.listdir(json_path)
    pla_list, mag_list = [], []
    for name in items:
        if name.startswith("mag") and name.endswith(".json"):
            mag_list.append(name)
        if name.startswith("pla") and name.endswith(".json"):
            pla_list.append(name)
            
    pla_list.sort()
    mag_list.sort()
    
    pla_keys = ['time_tag', 'density', 'speed', 'temperature']
    mag_keys = ['time_tag', 'bx_gsm', 'by_gsm', 'bz_gsm', 'bt']
    keys = list(set(mag_keys) | set(pla_keys))
    
    df_pla = pds.DataFrame(columns = pla_keys)
    df_mag = pds.DataFrame(columns = mag_keys)

    # READ FILES
    # ----------

    # Go through plasma files:
    for json_file in pla_list[0:100]:
        #try:
        pla_data = read_noaa_rtsw_json_single(os.path.join(json_path, json_file))
        pla_data_df = pds.DataFrame(pla_data)
            
        pla_data_df['time'] = num2date(pla_data_df['time_tag'], tz=None) 
        pla_data_df['time'] = pds.to_datetime(pla_data_df['time'], format="%Y/%m/%d %H:%M")
        pla_data_df.set_index('time',  inplace=True)
        pla_data_df.index.name = None
        pla_data_df.index = pla_data_df.index.tz_localize(None)
        pla_data_df = pla_data_df.resample('1T').mean().dropna()
        df_pla = pds.concat([df_pla, pla_data_df])
        #except:
         #   pass

    # Go through magnetic files:
    for json_file in mag_list[0:100]:
        try:
            mag_data = read_noaa_rtsw_json_single(os.path.join(json_path, json_file))
            mag_data_df = pds.DataFrame(mag_data)
            
            mag_data_df['time'] = num2date(mag_data_df['time_tag'], tz=None) 
            mag_data_df['time'] = pds.to_datetime(mag_data_df['time'], format="%Y/%m/%d %H:%M")
            mag_data_df.set_index('time',  inplace=True)
            mag_data_df.index.name = None
            mag_data_df.index = mag_data_df.index.tz_localize(None)
            mag_data_df = mag_data_df.resample('1T').mean().dropna()
            
            df_mag = pds.concat([df_mag, mag_data_df])
        except:
            pass
    df_pla.drop(columns = ['time_tag'], inplace = True)
    df_mag.drop(columns = ['time_tag'], inplace = True)
    df = pds.concat([df_mag, df_pla])
    return df


def archive_noaa_rtsw_data(json_path, archive_path):
    """Archives the NOAA real-time solar wind data files in hdf5 format.

    Parameters
    ==========
    json_path : str
        String of directory containing plasma data files.
    archive_path : str
        String of directory to save plasma data file.

    Returns
    =======
    True if completed.

    Example
    =======
    >>> archive_noaa_rtsw_data("rtswdata", "archive")
    """

    logging.info('Archive NOAA real time solar wind data as h5 file')

    items = os.listdir(json_path)
    pla_list, mag_list = [], []
    for name in items:
        if name.startswith("mag") and name.endswith(".json"):
            mag_list.append(name)
        if name.startswith("pla") and name.endswith(".json"):
            pla_list.append(name)
    pla_list.sort()
    mag_list.sort()

    pla_keys = ['time_tag', 'density', 'speed', 'temperature']
    mag_keys = ['time_tag', 'bx_gsm', 'by_gsm', 'bz_gsm', 'bt']
    lens = 10000*2*len(mag_list)
    rtsw_pla = np.zeros((lens, len(pla_keys)))
    rtsw_mag = np.zeros((lens, len(mag_keys)))

    # READ FILES
    # ----------

    # Go through plasma files:
    kp = 0
    for json_file in pla_list:
        try:
            pla_data = read_noaa_rtsw_json(os.path.join(json_path, json_file))
            for ip, pkey in enumerate(pla_keys):
                rtsw_pla[kp:kp+np.size(pla_data),ip] = pla_data[pkey]
            kp = kp + np.size(pla_data)
            print("JSON load successful for file {}".format(json_file), end='\r')
        except:
            logging.error("JSON load failed for file {}".format(json_file))
    rtsw_pla_cut = rtsw_pla[0:kp]
    rtsw_pla_cut = rtsw_pla_cut[rtsw_pla_cut[:,0].argsort()] # sort by time
    dum, ind = np.unique(rtsw_pla_cut[:,0], return_index=True)
    rtsw_pla_fin = rtsw_pla_cut[ind] # remove multiples of timesteps

    # Go through magnetic files:
    km = 0
    for json_file in mag_list:
        try:
            mag_data = read_noaa_rtsw_json(os.path.join(json_path, json_file))
            for ip, pkey in enumerate(mag_keys):
                rtsw_mag[km:km+np.size(mag_data),ip] = mag_data[pkey]
            km = km + np.size(mag_data)
            print("JSON load successful for file {}".format(json_file), end='\r')
        except:
            logging.error("JSON load failed for file {}".format(json_file))
    rtsw_mag_cut = rtsw_mag[0:km]
    rtsw_mag_cut = rtsw_mag_cut[rtsw_mag_cut[:,0].argsort()] # sort by time
    dum, ind = np.unique(rtsw_mag_cut[:,0], return_index=True)
    rtsw_mag_fin = rtsw_mag_cut[ind] # remove multiples of timesteps


    # Interpolate onto minute and hour timesteps (since both files are mismatched/missing timesteps):
    first_timestamp_min = num2date(np.max((rtsw_pla_fin[0,0], rtsw_mag_fin[0,0])))
    first_timestamp_min = first_timestamp_min - timedelta(seconds=first_timestamp_min.second)
    last_timestamp_min = num2date(np.min((rtsw_pla_fin[-1,0], rtsw_mag_fin[-1,0])))
    last_timestamp_min = last_timestamp_min - timedelta(seconds=last_timestamp_min.second)
    n_min = int((last_timestamp_min-first_timestamp_min).total_seconds() / 60)

    min_steps = np.array([date2num(first_timestamp_min+timedelta(minutes=n)) for n in range(n_min)])

    # DEFINE HEADER
    # -------------
    metadata = {
        "Description": "Real time solar wind magnetic field and plasma data from NOAA",
        "TimeRange": "{} - {}".format(first_timestamp_min.strftime("%Y-%m-%dT%H:%M"), last_timestamp_min.strftime("%Y-%m-%d %H:%M")),
        "SourceURL": "https://services.swpc.noaa.gov/products/solar-wind/",
        "CompiledBy": "Helio4Cast code, https://github.com/helioforecast/helio4cast",
        "Authors": "C. Moestl (twitter @chrisoutofspace) and R. L. Bailey (GitHub bairaelyn)",
        "FileCreationDate": datetime.utcnow().strftime("%Y-%m-%dT%H:%M")+' UTC',
        "Units": "B-field: nT, Density: cm^-3, Temperature: K, Speed: km s^-1",
        "Notes": "None in data have been replaced with np.NaNs.",
    }

    # WRITE DATA: LAST 100 DAYS
    # -------------------------
    if not os.path.exists(archive_path):
        os.mkdir(archive_path)

    # Write to file (minute timesteps):
    hdf5_file = os.path.join(archive_path, 'rtsw_all.h5')
    hf = h5py.File(hdf5_file, mode='w')

    hf.create_dataset('time', data=min_steps)
    for key in pla_keys[1:]:
        data_interp = np.interp(min_steps, rtsw_pla_fin[:,0], rtsw_pla_fin[:,pla_keys.index(key)])
        hf.create_dataset(key, data=data_interp)
    for key in mag_keys[1:]:
        data_interp = np.interp(min_steps, rtsw_mag_fin[:,0], rtsw_mag_fin[:,mag_keys.index(key)])
        hf.create_dataset(key, data=data_interp)
    metadata['SamplingRate'] = 1./24./60.
    for k, v in metadata.items():
        hf.attrs[k] = v
    hf.close()


    logging.info('Archiving of NOAA data done')

    return True

def read_noaa_rtsw_json(json_file, timef="%Y-%m-%d %H:%M:%S.%f"):
    """Reads NOAA real-time solar wind data JSON files (already downloaded).

    Parameters
    ==========
    json_file : str
        String of direct path to plasma data file.

    Returns
    =======
    rtsw_data : np.array
        Numpy array with JSON keys accessible as keys or under rtsw_data.dtype.names.

    Example
    =======
    >>> json_file = 'data/plasma-7-day_2020_Mar_28_17_00.json'
    >>> pla_data = read_noaa_rtsw_json(json_file)
    """

    # Read JSON file:
    with open(json_file, 'r') as jdata:
        dp = json.loads(jdata.read())
        dpn = [[np.nan if x == None else x for x in d] for d in dp]     # Replace None w NaN
        dtype=[(x, 'float') for x in dp[0]]
        datesp = [datetime.strptime(x[0], timef)  for x in dpn[1:]]
        #convert datetime to matplotlib times
        mdatesp = date2num(datesp)
        dp_ = [tuple([d]+[float(y) for y in x[1:]]) for d, x in zip(mdatesp, dpn[1:])]
        rtsw_data = np.array(dp_, dtype=dtype)

    return rtsw_data



def save_wind_data_ascii():
    
    '''
    description of data sources used in this function
    
    SWE 92 sec
    https://spdf.gsfc.nasa.gov/pub/data/wind/swe/ascii/swe_kp_unspike
    
    MFI 1 min    
    https://spdf.gsfc.nasa.gov/pub/data/wind/mfi/ascii/1min_ascii/
    
    examples:
    
    https://spdf.gsfc.nasa.gov/pub/data/wind/swe/ascii/swe_kp_unspike/wind_kp_unspike1996.txt    
    https://spdf.gsfc.nasa.gov/pub/data/wind/mfi/ascii/1min_ascii/201908_wind_mag_1min.asc
    '''
  
    t_start = datetime(1995,1,1)
    t_end = datetime(2016,1,1)
    
    #create an array with 2 minute resolution between t start and end
    time = [ t_start + timedelta(minutes=n) for n in range(int ((t_end - t_start).days*60*24))]  
    time_mat=mdates.date2num(time)
    read_data_end_year = 2016
    
    
    ##########################################################################
    

    #############mfi
    
    wind_data_path='/nas/helio/data/Wind/mfi_1min_ascii/'

    wind_years_strings=[]
    for j in np.arange(1995,read_data_end_year):
        wind_years_strings.append(str(j))


    #array for 21 years
    win_mag=np.zeros(60*24*365*21,dtype=[('time',object),('bx', float),('by', float),\
                    ('bz', float),('bt', float),('np', float),('vt', float), ('tp', float)])   

    #convert to recarray
    win_mag = win_mag.view(np.recarray)  


    counter=0


    for i in np.arange(0,len(wind_years_strings)):    

        for k in np.arange(1,13):    

            a=str(k).zfill(2) #add leading zeros

            file=wind_data_path+wind_years_strings[i]+a+'_wind_mag_1min.asc' #199504_wind_mag_1min.asc	
            print(file, end='\r')
            #get data from file, no 2021 12 available yet
            if file!='/nas/helio/data/Wind/mfi_1min_ascii/202112_wind_mag_1min.asc':
                mfi_data=np.genfromtxt(file,dtype="i8,i8,i8,i8,i8,i8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,")


            #put data in array
            for p in np.arange(0,len(mfi_data)):
                #time
                win_mag.time[p+counter]=datetime(mfi_data[p][0],mfi_data[p][1],mfi_data[p][2],mfi_data[p][3],mfi_data[p][4],mfi_data[p][5])
                win_mag.bx[p+counter]=mfi_data[p][6]
                
                win_mag.by[p+counter]=mfi_data[p][9]
                win_mag.bz[p+counter]=mfi_data[p][10]
                win_mag.bt[p+counter]=mfi_data[p][11]

            counter=counter+len(mfi_data)    

    #cutoff        
    win_mag2=win_mag[0:counter]    
    win_time2=mdates.date2num(win_mag2.time)

    #set missing data to nan
    win_mag2.bt[np.where(win_mag2.bt == -1e31)]=np.nan  
    win_mag2.bx[np.where(win_mag2.bx == -1e31)]=np.nan  
    win_mag2.by[np.where(win_mag2.by == -1e31)]=np.nan  
    win_mag2.bz[np.where(win_mag2.bz == -1e31)]=np.nan    
    
    
    ############################swe

    
    #array for 21 years
    win_swe=np.zeros(60*24*365*21,dtype=[('time',object),('bx', float),('by', float),\
                    ('bz', float),('bt', float),('np', float),('vt', float),('tp', float)])   

    #convert to recarray
    win_swe = win_swe.view(np.recarray)  

    
    
    counter=0

    wind_data_path='/nas/helio/data/Wind/swe_92sec_ascii/'

    for i in np.arange(0,len(wind_years_strings)):    

            file=wind_data_path+'wind_kp_unspike'+wind_years_strings[i]+'.txt'
            print(file, end='\r')
            swe_data=np.genfromtxt(file,dtype="i8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8,f8")

            firstjan=mdates.date2num(datetime(int(wind_years_strings[i]),1,1))-1

            for p in np.arange(0,len(swe_data)):

                #datenum
                win_swe.time[p+counter]=firstjan+swe_data[p][1]
                win_swe.vt[p+counter]=swe_data[p][2]
                win_swe.tp[p+counter]=swe_data[p][6]
                win_swe.np[p+counter]=swe_data[p][7]            

            counter=counter+len(swe_data) 


   
    win_swe=win_swe[0:counter]    
    win_swe_time=np.array(win_swe.time,dtype='float')


    win_swe.vt[np.where(win_swe.vt == 99999.9)]=np.nan  
    win_swe.tp[np.where(win_swe.tp ==9999999.)]=np.nan  
    win_swe.np[np.where(win_swe.np ==999.99)]=np.nan  


    
    ###################### interpolate
    
     
    #linear interpolation to time_mat times    
    bx = np.interp(time_mat, win_time2, win_mag2.bx )
    by = np.interp(time_mat, win_time2, win_mag2.by )
    bz = np.interp(time_mat, win_time2, win_mag2.bz  )
    bt = np.sqrt(bx**2+by**2+bz**2)
        
    den = np.interp(time_mat, win_swe_time, win_swe.np)
    vt = np.interp(time_mat, win_swe_time,win_swe.vt)
    tp = np.interp(time_mat, win_swe_time,win_swe.tp)
    
    #make array
    win=np.zeros(np.size(bx),dtype=[('time',object),('bx', float),('by', float),\
                ('bz', float),('bt', float),('np', float),('vt', float),('tp', float)])   
       
    #convert to recarray
    win = win.view(np.recarray)  

    #fill with data
    win.time=time
    win.bx=bx
    win.by=by
    win.bz=bz 
    win.bt=bt

    win.np=den
    win.vt=vt
    win.tp=tp
    
    ############ spike removal
            
    #plasma    
    win.np[np.where(win.np> 500)]=1000000
    #get rid of all single spikes with scipy signal find peaks
    peaks, properties = scipy.signal.find_peaks(win.np, height=500,width=(1, 250))
    #go through all of them and set to nan according to widths
    for i in np.arange(len(peaks)):
        #get width of current peak
        width=int(np.ceil(properties['widths']/2)[i])
        #remove data
        win.np[peaks[i]-width-2:peaks[i]+width+2]=np.nan

    win.tp[np.where(win.tp> 1e8)]=1e11
    #get rid of all single spikes with scipy signal find peaks
    peaks, properties = scipy.signal.find_peaks(win.tp, height=1e8,width=(1, 250))
    #go through all of them and set to nan according to widths
    for i in np.arange(len(peaks)):
        #get width of current peak
        width=int(np.ceil(properties['widths']/2)[i])
        #remove data
        win.tp[peaks[i]-width-2:peaks[i]+width+2]=np.nan

    win.vt[np.where(win.vt> 3000)]=1e11
    #get rid of all single spikes with scipy signal find peaks
    peaks, properties = scipy.signal.find_peaks(win.vt, height=1e8,width=(1, 250))
    #go through all of them and set to nan according to widths
    for i in np.arange(len(peaks)):
        #get width of current peak
        width=int(np.ceil(properties['widths']/2)[i])
        #remove data
        win.vt[peaks[i]-width-2:peaks[i]+width+2]=np.nan

        
        
        
    #magnetic field    
    peaks, properties = scipy.signal.find_peaks(win.bt, prominence=30,width=(1, 10))
    #go through all of them and set to nan according to widths
    for i in np.arange(len(peaks)):
        #get width of current peak
        width=int(np.ceil(properties['widths'])[i])
        #remove data
        win.bt[peaks[i]-width-5:peaks[i]+width+5]=np.nan    

    peaks, properties = scipy.signal.find_peaks(abs(win.bx), prominence=30,width=(1, 10))
    for i in np.arange(len(peaks)):
        width=int(np.ceil(properties['widths'])[i])
        win.bx[peaks[i]-width-5:peaks[i]+width+5]=np.nan    

    peaks, properties = scipy.signal.find_peaks(abs(win.by), prominence=30,width=(1, 10))
    for i in np.arange(len(peaks)):
        width=int(np.ceil(properties['widths'])[i])
        win.by[peaks[i]-width-5:peaks[i]+width+5]=np.nan    

    peaks, properties = scipy.signal.find_peaks(abs(win.bz), prominence=30,width=(1, 10))
    for i in np.arange(len(peaks)):
        width=int(np.ceil(properties['widths'])[i])
        win.bz[peaks[i]-width-5:peaks[i]+width+5]=np.nan    



    #manual spike removal for magnetic field
    if t_start < datetime(2018, 7, 19, 16, 25):    
        if t_end > datetime(2018, 7, 19, 16, 25):         

            remove_start=datetime(2018, 7, 19, 16, 25)
            remove_end=datetime(2018, 7, 19, 17, 35)
            remove_start_ind=np.where(remove_start<win.time)[0][0]
            remove_end_ind=np.where(remove_end<win.time)[0][0] 

            win.bt[remove_start_ind:remove_end_ind]=np.nan
            win.bx[remove_start_ind:remove_end_ind]=np.nan
            win.by[remove_start_ind:remove_end_ind]=np.nan
            win.bz[remove_start_ind:remove_end_ind]=np.nan

    if t_start < datetime(2018, 8, 29, 19, 0):    
        if t_end > datetime(2018, 8, 29, 19, 0):         

            remove_start=datetime(2018, 8, 29, 19, 0)
            remove_end=datetime(2018,8, 30, 5, 0)
            remove_start_ind=np.where(remove_start<win.time)[0][0]
            remove_end_ind=np.where(remove_end<win.time)[0][0] 

            win.bt[remove_start_ind:remove_end_ind]=np.nan
            win.bx[remove_start_ind:remove_end_ind]=np.nan
            win.by[remove_start_ind:remove_end_ind]=np.nan
            win.bz[remove_start_ind:remove_end_ind]=np.nan

            
    if t_start < datetime(2019, 8, 8, 22, 45):    
        if t_end > datetime(2019, 8, 8, 22, 45):         

            remove_start=datetime(2019, 8, 8, 22, 45)
            remove_end=datetime(2019,   8, 9, 17, 0)
            remove_start_ind=np.where(remove_start<win.time)[0][0]
            remove_end_ind=np.where(remove_end<win.time)[0][0] 

            win.bt[remove_start_ind:remove_end_ind]=np.nan
            win.bx[remove_start_ind:remove_end_ind]=np.nan
            win.by[remove_start_ind:remove_end_ind]=np.nan
            win.bz[remove_start_ind:remove_end_ind]=np.nan

    if t_start < datetime(2019, 8, 21, 22, 45):    
        if t_end > datetime(2019, 8, 21, 22, 45):         

            remove_start=datetime(2019, 8, 20, 18, 0)
            remove_end=datetime(2019,   8, 21, 12, 0)
            remove_start_ind=np.where(remove_start<win.time)[0][0]
            remove_end_ind=np.where(remove_end<win.time)[0][0] 

            win.bt[remove_start_ind:remove_end_ind]=np.nan
            win.bx[remove_start_ind:remove_end_ind]=np.nan
            win.by[remove_start_ind:remove_end_ind]=np.nan
            win.bz[remove_start_ind:remove_end_ind]=np.nan            

    if t_start < datetime(2019, 8, 21, 22, 45):    
        if t_end > datetime(2019, 8, 21, 22, 45):         

            remove_start=datetime(2019, 8, 22, 1, 0)
            remove_end=datetime(2019,   8, 22, 9, 0)
            remove_start_ind=np.where(remove_start<win.time)[0][0]
            remove_end_ind=np.where(remove_end<win.time)[0][0] 

            win.bt[remove_start_ind:remove_end_ind]=np.nan
            win.bx[remove_start_ind:remove_end_ind]=np.nan
            win.by[remove_start_ind:remove_end_ind]=np.nan
            win.bz[remove_start_ind:remove_end_ind]=np.nan            



    pickle.dump(win, open('datasets/files/wind_all.p', "wb"))
    