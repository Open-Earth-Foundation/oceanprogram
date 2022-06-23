#!/usr/bin/env python
# coding: utf-8

# Retrieving CMEMS data automatically

"""
This script shows how to retrieve Copernicus Marine Environment Monitoring Service (CMEMS) automatically.

In our case, we want the following environmental variables:
- nitrate concentrations
- phosphate concentrations
- silica concentrations
- chlorophyll concentrations
- the euphotic zone depth

Given that you are a registered CMEMS user, this data can be manually downloaded via CMEMS My Ocean Viewer at http://myocean.marine.copernicus.eu/ using the “Atlantic-Iberian Biscay Irish- Ocean Biogeochemical Analysis and Forecast” product. 

This is the product page: https://resources.marine.copernicus.eu/product-detail/IBI_MULTIYEAR_BGC_005_003/INFORMATION.


Here, we use the `motuclient` to retrieve the data. NOTE: you need a registered username!

See here for Copernicus documentation on motuclient:
https://help.marine.copernicus.eu/en/articles/4796533-what-are-the-motu-client-motuclient-and-python-requirements    
https://help.marine.copernicus.eu/en/articles/4899195-how-to-generate-and-run-a-script-to-download-a-subset-of-a-dataset-from-the-copernicus-marine-data-store
https://help.marine.copernicus.eu/en/articles/5211063-how-to-use-the-motuclient-within-python-environment-e-g-spyder

Script by Margaux Filippi for the Open Earth Foundation based on:
"""
#! /usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Copernicus Marine User Support Team"
__copyright__ = "(C) 2021 E.U. Copernicus Marine Service Information"
__credits__ = ["E.U. Copernicus Marine Service Information"]
__license__ = "MIT License - You must cite this source"
__version__ = "202105"
__maintainer__ = "D. Bazin, E. DiMedio, J. Cedillovalcarce, C. Giordan"
__email__ = "servicedesk dot cmems at mercator hyphen ocean dot eu"



# PARAMETERS
# Enter your parameters here
# product and service ID
service_id = "IBI_MULTIYEAR_BGC_005_003-TDS"
product_id = "cmems_mod_ibi_bgc_my_0.083deg-3D_P1D-m"
# datetimes in yyyy-mm-dd HH:MM:SS format
date_min = "2017-08-03 00:00:00"
date_max = "2017-08-10 23:59:59"
# coordinates
depth_min = 0.5057600140571594
depth_max = 0.5057600140571594
lon_min = -7
lon_max = 3.5
lat_min = 49.5
lat_max = 55.0
# variables
variables = ["no3", "po4", "si", "chl", "zeu"]

#------------------
import os
import motuclient
import getpass

class MotuOptions:
    def __init__(self, attrs: dict):
        super(MotuOptions, self).__setattr__("attrs", attrs)

    def __setattr__(self, k, v):
        self.attrs[k] = v

    def __getattr__(self, k):
        try:
            return self.attrs[k]
        except KeyError:
            return None

def motu_option_parser(script_template, usr, pwd, output_filename):
    dictionary = dict(
        [e.strip().partition(" ")[::2] for e in script_template.split('--')])
    dictionary['variable'] = [value for (var, value) in [e.strip().partition(" ")[::2] for e in script_template.split('--')] if var == 'variable']  # pylint: disable=line-too-long
    for k, v in list(dictionary.items()):
        if v == '<OUTPUT_DIRECTORY>':
            dictionary[k] = '.'
        if v == '<OUTPUT_FILENAME>':
            dictionary[k] = output_filename
        if v == '<USERNAME>':
            dictionary[k] = usr
        if v == '<PASSWORD>':
            dictionary[k] = pwd
        if k in ['longitude-min', 'longitude-max', 'latitude-min', 
                 'latitude-max', 'depth-min', 'depth-max']:
            dictionary[k] = float(v)
        if k in ['date-min', 'date-max']:
            dictionary[k] = v[1:-1]
        dictionary[k.replace('-','_')] = dictionary.pop(k)
    dictionary.pop('python')
    dictionary['auth_mode'] = 'cas'
    
    return dictionary

# make a folder within the current directory to download the output files
out_dir = "automated_CMEMS_downloads"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
print("The downloaded files will be saved in the following directory: " + out_dir)

USERNAME = input('Enter your username: ')
PASSWORD = getpass.getpass('Enter your password: ')
OUTPUT_FILENAME = input('Output filename (e.g. myexample.nc): ')

# this is the template - the actual inputs will be entered subsequently
script_template = 'python -m motuclient --motu https://nrt.cmems-du.eu/motu-web/Motu --service-id GLOBAL_ANALYSIS_FORECAST_PHY_001_024-TDS --product-id global-analysis-forecast-phy-001-024 --longitude-min 100 --longitude-max 155 --latitude-min 10 --latitude-max 40 --date-min "2021-02-21 12:00:00" --date-max "2021-02-27 12:00:00" --depth-min 0.493 --depth-max 0.4942 --variable thetao --out-dir <OUTPUT_DIRECTORY> --out-name <OUTPUT_FILENAME> --user <USERNAME> --pwd <PASSWORD>'

# actual request dictionary
data_request_options_dict_manual = {
    "service_id": service_id,
    "product_id": product_id,
    "date_min": date_min,
    "date_max": date_max,
    "longitude_min": lon_min,
    "longitude_max": lon_max,
    "latitude_min": lat_min,
    "latitude_max": lat_max,
    "depth_min": depth_min,
    "depth_max": depth_max,
    "variable": variables,#["thetao"],
    "motu": "https://my.cmems-du.eu/motu-web/Motu",#"https://nrt.cmems-du.eu/motu-web/Motu",
    "out_dir": out_dir,
    "out_name": OUTPUT_FILENAME,
    "auth_mode": "cas",
    "user": USERNAME,
    "pwd": PASSWORD
}

print("This is the summary of the request:")
print(data_request_options_dict_manual)

data_request_options_dict_automated = motu_option_parser(script_template, USERNAME, PASSWORD, OUTPUT_FILENAME)

motuclient.motu_api.execute_request(MotuOptions(data_request_options_dict_manual))

