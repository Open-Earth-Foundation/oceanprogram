import numpy as np
import xarray as xr

"""
These definitions follow Sarelli et al (2018) [SA18]
http://doi.org/10.1117/12.2326160
"""

def get_overlapping_domain():
    # this function gets the intersection domains, e.g. if
    # the concentrations of nitrate, phosphate, silica or
    # chlorophyll do not overlap.
    # To be filled.
    pass

"""
The calculations below return the indices for nutrients,
chlorophyll and euphotic depth. The index for coastal
eutrophication ICEP then corresponds to the sum of these indices.
"""
def get_Z_eu(Z_SD, linear_fn = True):
    # returns the euphotic zone depth Z_eu [m] from the Secchi depth [m]
    # this follows SA18

    # the linear function was showed to give better results, per SA18
    if linear_fn:
        Z_eu = 1.9322*Z_SD + 2.6629
    else:
        Z_eu = 3.7489*(Z_SD**0.7506)

    return Z_eu


def get_Z_SD(Z_eu, linear_fn = True):
    # invert the equation - this follows SA18

    if linear_fn:
        Z_SD = (Z_eu - 2.6669)/1.9322
    else:
        Z_SD = (Z_eu/3.7489)**(1/0.7506)

    return Z_SD


def get_indN(C_N, C_Si, C_P):
    # get the level 1 index from N, P, Si
    # see Figure 2 workflow in SA18

    # If the concentrations are scalar values, return a scalar
    if (np.isscalar(C_N))&(np.isscalar(C_Si))&(np.isscalar(C_P)):
        # if there is more silica than both phosphorus and nitrogen
        if (C_P<C_Si)&(C_N<C_Si):
            ind_N = 0
        # if there is more phosphorus than silica or more nitrogen than silica
        if (C_P>C_Si)|(C_N>C_Si):
            ind_N = 1
        # if there is both more phosphorus and nitrogen than silica
        if (C_P>C_Si)&(C_N>C_Si):
            ind_N = 2

    # If the concentrations are xarray's (hopefully)
    elif (type(C_N)==type(C_Si)==type(C_P)==xr.core.dataarray.DataArray):

        ind_N = xr.DataArray(coords=C_N.coords, dims=C_N.dims)
        ind_N.data[(C_P.data<C_Si.data)&(C_N.data<C_Si.data)] = 0
        ind_N.data[(C_P.data>C_Si.data)|(C_N.data>C_Si.data)] = 1
        ind_N.data[(C_P.data>C_Si.data)&(C_N.data>C_Si.data)] = 2

    else: # try a simple array
        # instantiate the variable
        domain = get_overlapping_domain()
        ind_N = np.empty(C_N.shape)
        ind_N.fill(np.nan)

        ind_N[(C_P<C_Si)&(C_N<C_Si)] = 0
        ind_N[(C_P>C_Si)|(C_N>C_Si)] = 1
        ind_N[(C_P>C_Si)&(C_N>C_Si)] = 2

    return ind_N


def get_ind_chl(chl):
    # get the level 2 index
    # see Figure 2 workflow in SA18

    # If the concentration is a scalar value
    if np.isscalar(chl):
        if chl <= 2.2:
            ind_chl = 0
        if (chl > 2.2)&(chl <= 3.2):
            ind_chl = 1
        if chl > 3.2:
            ind_chl = 2

    # If the concentration is an xarray (hopefully)
    elif (type(chl)==xr.core.dataarray.DataArray):

        ind_chl = xr.DataArray(coords=chl.coords, dims=chl.dims)
        ind_chl.data[chl.data <= 2.2] = 0
        ind_chl.data[(chl.data > 2.2)&(chl.data<= 3.2)] = 1
        ind_chl.data[chl.data > 3.2] = 2

    else: # try a simple array
        # instantiate the variable
        domain = get_overlapping_domain()
        ind_chl = np.empty(chl.shape)
        ind_chl.fill(np.nan)

        ind_chl[chl <= 2.2] = 0
        ind_chl[(chl > 2.2)&(chl <= 3.2)] = 1
        ind_chl[chl > 3.2] = 2

    return ind_chl

def get_ind_ZSD(Z_eu):
    # get the level 3 index
    # see Figure 2 workflow in SA18

    # If the Z_eu is a scalar value
    if np.isscalar(Z_eu):
        if Z_eu > 6:
            ind_Z_eu = 0
        if (Z_eu > 3)&(Z_eu <= 6):
            ind_Z_eu = 1
        # if there is both more phosphorus and nitrogen than silica
        if Z_eu <= 3:
            ind_Z_eu = 2

    # If the depth is an xarray (hopefully)
    elif (type(Z_eu)==xr.core.dataarray.DataArray):

        ind_Z_eu = xr.DataArray(coords=Z_eu.coords, dims=Z_eu.dims)
        ind_Z_eu.data[Z_eu.data > 6] = 0
        ind_Z_eu.data[(Z_eu.data > 3)&(Z_eu.data <= 6)] = 1
        ind_Z_eu.data[Z_eu.data <= 3] = 2

    else: # try a simple array
        # instantiate the variable
        domain = get_overlapping_domain()
        ind_Z_eu = np.empty(Z_eu.shape)
        ind_Z_eu.fill(np.nan)

        ind_Z_eu[Z_eu > 6] = 0
        ind_Z_eu[(Z_eu > 3)&(Z_eu <= 6)] = 1
        ind_Z_eu[Z_eu <= 3] = 2

    return ind_Z_eu

def get_index(ind_N, ind_chl, ind_Z_eu):
    # calculates ICEP
    return ind_N + ind_chl + ind_Z_eu
