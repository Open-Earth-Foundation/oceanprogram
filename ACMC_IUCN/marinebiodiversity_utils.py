import numpy as np

def get_DistRatioFactor(DistRatio, print_out = True):
    # This function calculates the weighted endemism of species by taking the sum of the
    # ratios of (area conserved / total distribution area)
    #
    # Input arguments:
    #      DistRatio: a series of the area conserved / total distribution area ratio for each species
    #      print_out: a boolean for printing out the factor
    #
    # Output:
    #      factor: the factor of weighted endemism
    
    factor = 1+sum(DistRatio)
    if print_out:
        print("We multiply N credits by " + "{:0.3f}".format(factor))
    return factor


def get_DistRatioFactor_square(DistRatio, print_out = False):
    # This function calculates the weighted endemism of species by taking the sum of the
    # squares of the ratios of (area conserved / total distribution area)
    #
    # Input arguments:
    #      DistRatio: a series of the area conserved / total distribution area ratio for each species
    #      print_out: a boolean for printing out the factor
    #
    # Output:
    #      factor: the factor of weighted endemism
    
    factor = 1+sum(DistRatio**.5)
    if print_out:
        print("We multiply N credits by " + "{:0.3f}".format(factor))
    return factor


def example_credits(area_sqdeg, DistRatio, factorMethod, print_out = True):
    # This function calculates the number of credits obtained over an area
    # by incorporating the weighted endemism of species
    #
    # Input arguments:
    #      area_sqdeg: the area of the conserved zone in square degrees
    #      DistRatio: a series of the area conserved / total distribution area ratio for each species
    #      factorMethod: function to calculate the weighted endemism of species
    #
    # Output:
    #      Ncredits: the number of credits obtained
    
    # assuming the area is in square degrees, taking a rough estimate for now
    Ncredits_base = area_sqdeg*(111.1**2) #1deg = 111.1 km
    
    factor = factorMethod(DistRatio)
    Ncredits = Ncredits_base*factor
    
    if print_out:
        print("Using " + factorMethod.__name__ + ", from " + "{:0.2f}".format(Ncredits_base) + " credits over " + \
              "{:0.2f}".format(area_sqdeg) + "sqdeg," + \
              " we obtain " + "{:0.2f}".format(Ncredits) + " credits from the weighted endemism of " + \
             str(len(DistRatio)) + " local species.\n")
    
    return Ncredits


def get_IUCN50_prob(iucn_status):
    """
    inputs:
        iucn_status: <string/object> containing one of ['Least Concern', 'Vulnerable', 'Data Deficient', 'Near Threatened',
       'Endangered', 'Critically Endangered']
    outputs:
        er: <float> extinction probability as defined by the IUCN50 standard (Davis et al., 2018)
    """
    
    if (iucn_status == 'Least Concern'):
        return 0.0009
    elif (iucn_status == 'Near Threatened'):
        return 0.0071
    elif (iucn_status == 'Vulnerable'):
        return 0.0513
    elif (iucn_status == 'Endangered'):
        return 0.4276
    elif (iucn_status == 'Critically Endangered'):
        return 0.9688
    elif (iucn_status == 'Data Deficient'):
        return 0.5 #arbitrarily assigned, needs to be re-evaluted by first analyzing Bland, Collen, Orme, and Bielby (2015).
    else:
        print("ERROR: unknown IUCN category: ",iucn_status)
        return 0


    
def WEGE_score(df, factorMethod, distRatio_col, erMethod, rl_col):
    """
    inputs:
        df: geopanda/panda series
        factorMethod: function to calculate the weighted endemism of species
        distRatio_col: <string> column name that has the distibution ratio to be used
        erMethod: extinction probability derivation method
        rl_col: <string> column name that has the IUCN redlist category names
    outputs:
        none: a new column called 'WEGE' is written into df
    """
    
    df['WEGE'] = df[distRatio_col].apply(lambda z: factorMethod(z)) * df[rl_col].apply(lambda x: erMethod(x))

    
def WEGE_credits_area(df, area_sqdeg, wege_col):
    """
    inputs:
        df: geopanda/panda series
        area_sqdeg: the area of the conserved zone in square degrees
        wege_col: <string> column name that has the WEGE score per species
        erMethod: extinction probability derivation method
        rl_col: <string> column name that has the IUCN redlist category names
    outputs:
        WEGE_credits: total number of credits for the area under protection
    """

    # assuming the area is in square degrees, taking a rough estimate for now
    Ncredits_base = area_sqdeg*(111.1**2) #1deg = 111.1 km
    WEGE_score = np.sum(df[wege_col])
    
    return WEGE_score*Ncredits_base