"""
TODO in here:
- Update archive_noaa_rtsw_data so it only updates files, doesn't read in all every time.
- Write NOAA RTSW data minute+hourly data into year files.
"""

import os
import sys
from datetime import datetime, timedelta
import h5py
import json
import logging
from matplotlib.dates import num2date, date2num
import matplotlib.pyplot as plt
import numpy as np
import urllib
import urllib.error
import requests

import copy
import heliosat

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
#    PREDSTORM LEGACY
# -------------------------------------------------------------------

# =======================================================================================
# -------------------------------- I. CLASSES ------------------------------------------
# =======================================================================================


class SatData():
    """Data object containing satellite data.

    Init Parameters
    ===============
    --> SatData(input_dict, source=None, header=None)
    input_dict : dict(key: dataarray)
        Dict containing the input data in the form of key: data (in array or list)
        Example: {'time': timearray, 'bx': bxarray}. The available keys for data input
        can be accessed in SatData.default_keys.
    header : dict(headerkey: value)
        Dict containing metadata on the data array provided. Useful data headers are
        provided in SatData.empty_header but this can be expanded as needed.
    source : str
        Provide quick-access name of satellite/data type for source.

    Attributes
    ==========
    .data : np.ndarray
        Array containing measurements/indices. Best accessed using SatData[key].
    .position : np.ndarray
        Array containing position data for satellite.
    .h : dict
        Dict of metadata as defined by input header.
    .state : np.array (dtype=object)
        Array of None, str if defining state of data (e.g. 'quiet', 'cme').
    .vars : list
        List of variables stored in SatData.data.
    .source : str
        Data source name.

    Methods
    =======
    .convert_GSE_to_GSM()
        Coordinate conversion.
    .convert_RTN_to_GSE()
        Coordinate conversion.
    .cut(starttime=None, endtime=None)
        Cuts data to within timerange and returns.
    .get_position(timestamp)
        Returns position of spacecraft at time.
    .get_newell_coupling()
        Calculates Newell coupling indice for data.
    .interp_nans(keys=None)
        Linearly interpolates over nans in data.
    .interp_to_time()
        Linearly interpolates over nans.
    .load_position_data(position_data_file)
        Loads position data from file.
    .make_aurora_power_prediction()
        Calculates aurora power.
    .make_dst_prediction()
        Makes prediction of Dst from data.
    .make_kp_prediction()
        Prediction of kp.
    .make_hourly_data()
        Takes minute resolution data and interpolates to hourly data points.
    .shift_time_to_L1()
        Shifts time to L1 from satellite ahead in sw rotation.

    Examples
    ========
    """

    default_keys = ['time',
                    'speed', 'speedx', 'density', 'temp', 'pdyn',
                    'bx', 'by', 'bz', 'btot',
                    'br', 'bt', 'bn',
                    'dst', 'kp', 'aurora', 'ec', 'ae', 'f10.7']

    empty_header = {'DataSource': '',
                    'SourceURL' : '',
                    'SamplingRate': None,
                    'ReferenceFrame': '',
                    'FileVersion': {},
                    'Instruments': [],
                    'RemovedTimes': [],
                    'PlasmaDataIntegrity': 10
                    }

    def __init__(self, input_dict, source=None, header=None):
        """Create new instance of class."""

        # Check input data
        for k in input_dict.keys():
            if not k in SatData.default_keys: 
                raise NotImplementedError("Key {} not implemented in SatData class!".format(k))
        if 'time' not in input_dict.keys():
            raise Exception("Time variable is required for SatData object!")
        dt = [x for x in SatData.default_keys if x in input_dict.keys()]
        if len(input_dict['time']) == 0:
            logger.warning("SatData.__init__: Inititating empty array! Is the data missing?")
        # Create data array attribute
        data = [input_dict[x] if x in dt else np.zeros(len(input_dict['time'])) for x in SatData.default_keys]
        self.data = np.asarray(data)
        # Create array for state classifiers (currently empty)
        self.state = np.array([None]*len(self.data[0]), dtype='object')
        # Add new attributes to the created instance
        self.source = source
        if header == None:               # Inititalise empty header
            self.h = copy.deepcopy(SatData.empty_header)
        else:
            self.h = header
        self.pos = None
        self.vars = dt
        self.vars.remove('time')

        
########## FUNCTIONS ################        
        
def get_noaa_realtime_data():
    """
    Downloads and returns real-time solar wind data
    7-day data downloaded from http://services.swpc.noaa.gov/products/solar-wind/

    Parameters
    ==========
    None

    Returns
    =======
    sw_data : ps.SatData object
        Object containing NOAA real-time solar wind data under standard keys.
    """

    url_plasma='http://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json'
    url_mag='http://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json'

    # Read plasma data:
    with urllib.request.urlopen(url_plasma) as url:
        dp = json.loads (url.read().decode())
        dpn = [[np.nan if x == None else x for x in d] for d in dp]     # Replace None w NaN
        dtype=[(x, 'float') for x in dp[0]]
        dates = [date2num(datetime.strptime(x[0], "%Y-%m-%d %H:%M:%S.%f")) for x in dpn[1:]]
        dp_ = [tuple([d]+[float(y) for y in x[1:]]) for d, x in zip(dates, dpn[1:])]
        plasma = np.array(dp_, dtype=dtype)
    # Read magnetic field data:
    with urllib.request.urlopen(url_mag) as url:
        dm = json.loads(url.read().decode())
        dmn = [[np.nan if x == None else x for x in d] for d in dm]     # Replace None w NaN
        dtype=[(x, 'float') for x in dmn[0]]
        dates = [date2num(datetime.strptime(x[0], "%Y-%m-%d %H:%M:%S.%f")) for x in dmn[1:]]
        dm_ = [tuple([d]+[float(y) for y in x[1:]]) for d, x in zip(dates, dm[1:])]
        magfield = np.array(dm_, dtype=dtype)

    last_timestep = np.min([magfield['time_tag'][-1], plasma['time_tag'][-1]])
    first_timestep = np.max([magfield['time_tag'][0], plasma['time_tag'][0]])

    nminutes = int((num2date(last_timestep)-num2date(first_timestep)).total_seconds()/60.)
    itime = np.asarray([date2num(num2date(first_timestep) + timedelta(minutes=i)) for i in range(nminutes)], dtype=np.float64)

    rbtot_m = np.interp(itime, magfield['time_tag'], magfield['bt'])
    rbxgsm_m = np.interp(itime, magfield['time_tag'], magfield['bx_gsm'])
    rbygsm_m = np.interp(itime, magfield['time_tag'], magfield['by_gsm'])
    rbzgsm_m = np.interp(itime, magfield['time_tag'], magfield['bz_gsm'])
    rpv_m = np.interp(itime, plasma['time_tag'], plasma['speed'])
    rpn_m = np.interp(itime, plasma['time_tag'], plasma['density'])
    rpt_m = np.interp(itime, plasma['time_tag'], plasma['temperature'])

    # Pack into object
    sw_data = SatData({'time': itime,
                           'btot': rbtot_m, 'bx': rbxgsm_m, 'by': rbygsm_m, 'bz': rbzgsm_m,
                           'speed': rpv_m, 'density': rpn_m, 'temp': rpt_m},
                           source='DSCOVR')
    sw_data.h['DataSource'] = "DSCOVR (NOAA)"
    sw_data.h['SamplingRate'] = 1./24./60.
    sw_data.h['ReferenceFrame'] = 'GSM'
    # Source isn't provided, but can assume DSCOVR:
    DSCOVR_ = heliosat.DSCOVR()
    sw_data.h['HeliosatObject'] = DSCOVR_

    logger.info('get_noaa_realtime_data: NOAA RTSW data read completed.')

    return sw_data


def get_dscovr_data(starttime, endtime, resolution='min', skip_files=True):

    if (datetime.utcnow() - starttime).days < 7.:
        dscovr_data = ps.get_dscovr_realtime_data()
        dscovr_data = dscovr_data.cut(starttime=starttime, endtime=endtime)
        return dscovr_data
    else:
        dscovr_data = get_dscovr_archive_data(starttime, endtime, 
                            resolution=resolution, skip_files=skip_files)
        return dscovr_data
    
def get_dscovr_archive_data(starttime, endtime, resolution='min', skip_files=True):
    """Downloads and reads STEREO-A beacon data from CDF files. Files handling
    is done using heliosat, so files are downloaded to HELIOSAT_DATAPATH.
    Data sourced from: 
    https://www.ngdc.noaa.gov/dscovr/portal/index.html#/download/1542848400000;1554163200000/f1m;m1m

    Parameters
    ==========
    starttime : datetime.datetime
        Datetime object with the required starttime of the input data.
    endtime : datetime.datetime
        Datetime object with the required endtime of the input data.
    resolution : str, (optional, 'min' (default) or 'hour')
        Determines which resolution data should be returned in.
    skip_files : bool (default=True)
        Heliosat get_data_raw var. Skips missing files in download folder.

    Returns
    =======
    dscovr : predstorm.SatData
        Object containing satellite data under keys.

    """

    logger = logging.getLogger(__name__)

    DSCOVR_ = heliosat.DSCOVR()

    logger.info("Reading archived DSCOVR data")

    # Magnetometer data
    magt, magdata = DSCOVR_.get([starttime, endtime], "mag", as_endpoints=True, return_datetimes=True, frame="HEEQ", cached=True)
    #magt = [datetime.utcfromtimestamp(t) for t in magt]
    bx, by, bz = magdata[:,0], magdata[:,1], magdata[:,2]
    missing_value = -99999.
    bx[bx==missing_value] = np.NaN
    by[by==missing_value] = np.NaN
    bz[bz==missing_value] = np.NaN
    btot = np.sqrt(bx**2. + by**2. + bz**2.)

    if len(bx) == 0:
        logger.error("DSCOVR data is missing or masked in time range! Returning empty data object.")
        return SatData({'time': []})

    # Particle data
    try:
        pt, pdata = DSCOVR_.get([starttime, endtime], "proton", as_endpoints=True, return_datetimes=True, frame="HEEQ", cached=True)
    except:
        DSCOVR_._json['keys']['dscovr_plas'] = DSCOVR_._json['dscovr_plas']
        pt, pdata = DSCOVR_.get([starttime, endtime], "proton", as_endpoints=True, return_datetimes=True, frame="HEEQ", cached=True)
    #pt = [datetime.utcfromtimestamp(t) for t in pt]
    density, vtot, temperature = pdata[:,0], pdata[:,1], pdata[:,2]
    density[density==missing_value] = np.NaN
    vtot[vtot==missing_value] = np.NaN
    temperature[temperature==missing_value] = np.NaN

    if resolution == 'hour':
        stime = date2num(starttime) - date2num(starttime)%(1./24.)
        nhours = (endtime.replace(tzinfo=None) - num2date(stime).replace(tzinfo=None)).total_seconds()/60./60.
        tarray = np.array(stime + np.arange(0, nhours)*(1./24.))
    elif resolution == 'min':
        stime = date2num(starttime) - date2num(starttime)%(1./24./60.)
        nmins = (endtime.replace(tzinfo=None) - num2date(stime).replace(tzinfo=None)).total_seconds()/60.
        tarray = np.array(stime + np.arange(0, nmins)*(1./24./60.))

    # Interpolate variables to time:
    bx_int = np.interp(tarray, date2num(magt), bx)
    by_int = np.interp(tarray, date2num(magt), by)
    bz_int = np.interp(tarray, date2num(magt), bz)
    btot_int = np.interp(tarray, date2num(magt), btot)
    density_int = np.interp(tarray, date2num(pt), density)
    vtot_int = np.interp(tarray, date2num(pt), vtot)
    temp_int = np.interp(tarray, date2num(pt), temperature)

    # # Pack into object:
    # dscovr = SatData({'time': tarray,
    #                   'btot': btot_int, 'bx': bx_int, 'by': by_int, 'bz': bz_int,
    #                   'speed': vtot_int, 'density': density_int, 'temp': temp_int},
    #                   source='DSCOVR')
    # dscovr.h['DataSource'] = "DSCOVR Level 1 (NOAA)"
    # dscovr.h['SamplingRate'] = tarray[1] - tarray[0]
    # if heliosat.__version__ >= '0.4.0':
    #     dscovr.h['ReferenceFrame'] = DSCOVR_.spacecraft['data_keys']['dscovr_mag']['version_default']['columns'][0]['frame']
    # else:
    #     dscovr.h['ReferenceFrame'] = DSCOVR_.spacecraft["data"]['mag'].get("frame")
    # dscovr.h['HeliosatObject'] = DSCOVR_

    return tarray, bx_int




# -------------------------------------------------------------------
#    ASWO BASE by Rachel
# -------------------------------------------------------------------


# -------------------------------------------------------------------
#    DOWNLOAD
# -------------------------------------------------------------------

def download_noaa_rtsw_data(save_path):
    """Downloads NOAA real-time solar wind data (plasma and mag).
    Parameters
    ==========
    save_path : str
        String of directory to save files to.
    Returns
    =======
    (get_plas, get_mag) : (bool, bool)
        Both True if both files were successfully downloaded.
    Example
    =======
    >>> pla_success, mag_success = download_noaa_rtsw_data("data")
    """

    plasma = 'http://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json'
    mag = 'http://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json'
    dst = 'http://services.swpc.noaa.gov/products/kyoto-dst.json'

    datestr=str(datetime.utcnow().strftime("%Y-%m-%dT%Hh"))
    logging.info('downloading NOAA real time solar wind plasma and mag for {}'.format(datestr))

    get_plas, get_mag, get_dst = True, True, False

    try:
        urllib.request.urlretrieve(plasma, os.path.join(save_path, 'plasma-7-day_'+datestr+'.json'))
    except urllib.error.URLError as e:
        logging.error(' '+plasma+' '+e.reason)
        get_plas = False

    try:
        urllib.request.urlretrieve(mag, os.path.join(save_path, 'mag-7-day_'+datestr+'.json'))
    except urllib.error.URLError as e:
        logging.error(' '+mag+' '+e.reason)
        get_mag = False

    try:
        urllib.request.urlretrieve(dst, os.path.join(save_path, 'dst-7-day_'+datestr+'.json'))
    except urllib.error.URLError as e:
        logging.error(' '+dst+' '+e.reason)
        get_dst = False

    return get_plas, get_mag


def archive_noaa_rtsw_data(json_path, archive_path, limit_by_ndays=100):
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
    pla_list, mag_list, dst_list = [], [], []
    for name in items:
       if name.startswith("mag") and name.endswith(".json"):
            mag_list.append(name)
       if name.startswith("pla") and name.endswith(".json"):
            pla_list.append(name)
       if name.startswith("dst") and name.endswith(".json"):
            dst_list.append(name)

    pla_keys = ['time_tag', 'density', 'speed', 'temperature']
    mag_keys = ['time_tag', 'bx_gsm', 'by_gsm', 'bz_gsm', 'bt']
    dst_keys = ['time_tag', 'dst']
    rtsw_pla = np.zeros((5000000, len(pla_keys)))
    rtsw_mag = np.zeros((5000000, len(mag_keys)))
    rtsw_dst = np.zeros((5000000, len(dst_keys)))

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
        except:
            logging.error("JSON load failed for file {}".format(json_file))
    rtsw_mag_cut = rtsw_mag[0:km]
    rtsw_mag_cut = rtsw_mag_cut[rtsw_mag_cut[:,0].argsort()] # sort by time
    dum, ind = np.unique(rtsw_mag_cut[:,0], return_index=True)
    rtsw_mag_fin = rtsw_mag_cut[ind] # remove multiples of timesteps

    # Go through Dst files:
    kd = 0
    for json_file in dst_list:
        try:
            dst_data = read_noaa_rtsw_json(os.path.join(json_path, json_file),
                                           timef="%Y-%m-%d %H:%M:%S")
            for ip, pkey in enumerate(dst_keys):
                rtsw_dst[kd:kd+np.size(dst_data),ip] = dst_data[pkey]
            kd = kd + np.size(dst_data)
        except:
            logging.error("JSON load failed for file {}".format(json_file))
    rtsw_dst_cut = rtsw_dst[0:kd]
    rtsw_dst_cut = rtsw_dst_cut[rtsw_dst_cut[:,0].argsort()] # sort by time
    dum, ind = np.unique(rtsw_dst_cut[:,0], return_index=True)
    rtsw_dst_fin = rtsw_dst_cut[ind] # remove multiples of timesteps

    # Interpolate onto minute and hour timesteps (since both files are mismatched/missing timesteps):
    first_timestamp_min = num2date(np.max((rtsw_pla_fin[0,0], rtsw_mag_fin[0,0])))
    first_timestamp_min = first_timestamp_min - timedelta(seconds=first_timestamp_min.second)
    first_timestamp_hour = num2date(np.max((rtsw_pla_fin[0,0], rtsw_mag_fin[0,0], rtsw_dst_fin[0,0])))
    first_timestamp_hour = first_timestamp_hour - timedelta(seconds=first_timestamp_hour.second)
    last_timestamp_min = num2date(np.min((rtsw_pla_fin[-1,0], rtsw_mag_fin[-1,0])))
    last_timestamp_min = last_timestamp_min - timedelta(seconds=last_timestamp_min.second)
    last_timestamp_hour = num2date(np.min((rtsw_pla_fin[-1,0], rtsw_mag_fin[-1,0], rtsw_dst_fin[-1,0])))
    last_timestamp_hour = last_timestamp_hour - timedelta(seconds=last_timestamp_hour.second)
    n_min = int((last_timestamp_min-first_timestamp_min).total_seconds() / 60)
    n_hour = int(np.round(int((last_timestamp_hour-first_timestamp_hour).total_seconds() / 60) / 60, 0))

    min_steps = np.array([date2num(first_timestamp_min+timedelta(minutes=n)) for n in range(n_min)])
    hour_steps = np.array([date2num(first_timestamp_hour+timedelta(hours=n)) for n in range(n_hour)])

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
    past_100days = datetime.utcnow() - timedelta(days=limit_by_ndays)

    if not os.path.exists(archive_path):
        os.mkdir(archive_path)

    # Write to file (minute timesteps):
    min_steps_100 = min_steps[min_steps > date2num(past_100days)]
    hour_steps_100 = hour_steps[hour_steps > date2num(past_100days)]
    hdf5_file = os.path.join(archive_path, 'rtsw_min_last100days.h5')
    hf = h5py.File(hdf5_file, mode='w')

    hf.create_dataset('time', data=min_steps_100)
    for key in pla_keys[1:]:
        data_interp = np.interp(min_steps_100, rtsw_pla_fin[:,0], rtsw_pla_fin[:,pla_keys.index(key)])
        hf.create_dataset(key, data=data_interp)
    for key in mag_keys[1:]:
        data_interp = np.interp(min_steps_100, rtsw_mag_fin[:,0], rtsw_mag_fin[:,mag_keys.index(key)])
        hf.create_dataset(key, data=data_interp)
    metadata['SamplingRate'] = 1./24./60.
    for k, v in metadata.items():
        hf.attrs[k] = v
    hf.close()

    # Write to file (hour timesteps):
    metadata["TimeRange"] = "{} - {}".format(first_timestamp_hour.strftime("%Y-%m-%dT%H:%M"), last_timestamp_hour.strftime("%Y-%m-%d %H:%M"))
    hdf5_file = os.path.join(archive_path, 'rtsw_hour_last100days.h5')
    hf = h5py.File(hdf5_file, mode='w')

    hf.create_dataset('time', data=hour_steps_100)
    for key in pla_keys[1:]:
        data_interp = np.interp(hour_steps_100, rtsw_pla_fin[:,0], rtsw_pla_fin[:,pla_keys.index(key)])
        hf.create_dataset(key, data=data_interp)
    for key in mag_keys[1:]:
        data_interp = np.interp(hour_steps_100, rtsw_mag_fin[:,0], rtsw_mag_fin[:,mag_keys.index(key)])
        hf.create_dataset(key, data=data_interp)
    for key in dst_keys[1:]:
        data_interp = np.interp(hour_steps_100, rtsw_dst_fin[:,0], rtsw_dst_fin[:,dst_keys.index(key)])
        hf.create_dataset(key, data=data_interp)
    metadata['SamplingRate'] = 1./24.
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
