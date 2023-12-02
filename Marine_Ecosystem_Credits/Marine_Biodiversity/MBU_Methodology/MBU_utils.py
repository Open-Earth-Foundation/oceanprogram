import os
import glob
import math
import numpy as np
import pandas as pd
from pyobis import occurrences
import shapely
import geopandas as gpd
from geopandas.tools import sjoin
from shapely.geometry import Polygon, Point, box
from shapely.ops import linemerge, unary_union, polygonize

#---------------------------------------------------------------------------------------------------------------------
#High Level Helper Functions
#---------------------------------------------------------------------------------------------------------------------

def get_OBIS(MPA):
    # create a polygon to access the OBIS data
    min_x, min_y, max_x, max_y = MPA.total_bounds
    geometry = f"POLYGON(({max_x} {min_y}, {min_x} {min_y}, {min_x} {max_y}, {max_x} {max_y}, {max_x} {min_y}))"

    query = occurrences.search(geometry=geometry)
    query.execute()

    # returns the data
    df = query.data 
    
    # List of columns to keep
    columns_to_keep = ['scientificName','decimalLongitude', 'decimalLatitude', 'individualCount']

    # Select and keep only the desired columns
    df = df[columns_to_keep]
    
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.decimalLongitude, df.decimalLatitude))
    
    return df

def create_grid(MPA, grid_shape="hexagon", grid_size_deg=1.):
    """
    This function make a grid gdf with the shape choose by the user
    
    input(s):
    MPA <shapely polygon>: region of interest or the total project area
    grid_size_m <int>: size of the grid in degrees
    grid_shape <str>: either "square" or "hexagon"

    output(s):
    gdf <geopandas dataframe>: containts at least a geometry colum and a unique grid_id
    """

    # Slightly displace the minimum and maximum values of the feature extent by creating a buffer
    # This decreases likelihood that a feature will fall directly on a cell boundary (in between two cells)
    # Buffer is projection dependent (due to units)
    #feature = feature.buffer(20) #This is increase the boundary by 20 degrees!
    # print("no buffer")

    # Get extent of buffered input feature
    min_x, min_y, max_x, max_y = MPA.total_bounds


    # Create empty list to hold individual cells that will make up the grid
    cells_list = []

    # Create grid of squares if specified
    if grid_shape in ["square", "rectangle", "box"]:

        # Adapted from https://james-brennan.github.io/posts/fast_gridding_geopandas/
        # Create and iterate through list of x values that will define column positions with specified side length
        for x in np.arange(min_x - grid_size_deg, max_x + grid_size_deg, grid_size_deg):

            # Create and iterate through list of y values that will define row positions with specified side length
            for y in np.arange(min_y - grid_size_deg, max_y + grid_size_deg, grid_size_deg):

                # Create a box with specified side length and append to list
                cells_list.append(box(x, y, x + grid_size_deg, y + grid_size_deg))


    # Otherwise, create grid of hexagons
    elif grid_shape == "hexagon":

        # Set horizontal displacement that will define column positions with specified side length (based on normal hexagon)
        x_step = 1.5 * grid_size_deg

        # Set vertical displacement that will define row positions with specified side length (based on normal hexagon)
        # This is the distance between the centers of two hexagons stacked on top of each other (vertically)
        y_step = math.sqrt(3) * grid_size_deg

        # Get apothem (distance between center and midpoint of a side, based on normal hexagon)
        apothem = (math.sqrt(3) * grid_size_deg / 2)

        # Set column number
        column_number = 0

        # Create and iterate through list of x values that will define column positions with vertical displacement
        for x in np.arange(min_x, max_x + x_step, x_step):

            # Create and iterate through list of y values that will define column positions with horizontal displacement
            for y in np.arange(min_y, max_y + y_step, y_step):

                # Create hexagon with specified side length
                hexagon = [[x + math.cos(math.radians(angle)) * grid_size_deg, y + math.sin(math.radians(angle)) * grid_size_deg] for angle in range(0, 360, 60)]

                # Append hexagon to list
                cells_list.append(Polygon(hexagon))

            # Check if column number is even
            if column_number % 2 == 0:

                # If even, expand minimum and maximum y values by apothem value to vertically displace next row
                # Expand values so as to not miss any features near the feature extent
                min_y -= apothem
                max_y += apothem

            # Else, odd
            else:

                # Revert minimum and maximum y values back to original
                min_y += apothem
                max_y -= apothem

            # Increase column number by 1
            column_number += 1

    # Else, raise error
    else:
        raise Exception("Specify a rectangle or hexagon as the grid shape.")

    # Create grid from list of cells
    grid = gpd.GeoDataFrame(cells_list, columns = ['geometry'], crs = 4326)

    # Create a column that assigns each grid a number
    grid["Grid_ID"] = np.arange(len(grid))

    # Return grid
    return grid

#---------------------------------------------------------------------------------------------------------------------

def accounting_by_grid(gdf, grid_gdf):
    """
    This function calculates the number of polygons that intersect each grid

    input(s):
        gdf <geopandas dataframe>: contains at least a geometry colum with the polygons to count
        grid_gdf <geopandas dataframe>: 

    output(s):
    gdf <geopandas dataframe>: containts at least a geometry colum and the "count" column
    """
    
    # Create a new column in grid_gdf to store the species count
    grid_gdf['count'] = 0

    # Iterate over each row in grid_gdf
    for index, row in grid_gdf.iterrows():
        grid_geometry = row['geometry']
        for feature_index, feature_row in gdf.iterrows():
            feature_geometry = feature_row['geometry']

            # Check for intersection
            if grid_geometry.intersects(feature_geometry):
                # Increment the species count for the current grid row
                grid_gdf.at[index, 'count'] += 1
    
    return grid_gdf

#---------------------------------------------------------------------------------------------------------------------

#This function calculates a specific algebra operation of all values of an interest colum of overlapping geometries 
def algebraic_sum_by_grid(gdf, grid_gdf, col_name):
    """
    This function calculates the algebraic summatory with values from a specific column for polygons that 
    intersect each grid.

    input(s):
        gdf <geopandas dataframe>: contains at least a geometry column that represent the species distribution 
        globally
        grid_gdf <geopandas dataframe>: contains at least a geometry column

    output(s):
        grid_gdf <geopandas dataframe>: contains at least a geometry column and the "summatory" column
    """

    # Create a new column in grid_gdf to store the result of the algebraic operation
    grid_gdf['result'] = 0

    # Iterate over each row in grid_gdf
    for index, row in grid_gdf.iterrows():
        grid_geometry = row['geometry']
        for feature_index, feature_row in gdf.iterrows():
            feature_geometry = feature_row['geometry']

            # Check for intersection
            if grid_geometry.intersects(feature_geometry):
                # Perform your algebraic operation here (e.g., sum 'Y' values)
                grid_gdf.at[index, 'result'] += feature_row[col_name]

    return grid_gdf
#---------------------------------------------------------------------------------------------------------------------

def Clip_EFG(MPA, path_EFG):
    """
    Inputs: 
    - MPA <shapely polygon in CRS WGS84:EPSG 4326>: region of interest or the total project area
    - path_EFG <list>: consists of a list of EFG polygons' paths to be reference internally
    
    Outputs:
    - gdf <geopandas dataframe>: consists of a list of EFG polygons cut with respect to MPA
    """
    
    Geometry = []
    
    if not isinstance(path_EFG, (np.ndarray, list)):
        raise TypeError("Need an array with the paths where the EFG layers are located")
    
    if isinstance(path_EFG, (np.ndarray, list)):
        for x in range(len(path_EFG)):
            # Reading each EFG
            gdf = gpd.read_file(str(path_EFG[x]))

            # Choosing the crs
            gdf = gdf.set_crs(epsg=4326, allow_override=True)

            # Attempt to fix self-intersecting geometries
            gdf['geometry'] = gdf['geometry'].buffer(0)

            # We want only the polygons within our study area
            clip = gpd.clip(gdf, MPA)
            clip = clip.reset_index()

            # Selecting the 'geometry' column
            geo = clip.geometry

            Geometry.append(geo)

        joined = gpd.GeoDataFrame(pd.concat(Geometry, ignore_index=True))

    return joined

#---------------------------------------------------------------------------------------------------------------------
def generate_random_values(mean=0, std_deviation=0.1, num_samples=100):
    random_values = np.random.normal(0, 0.1, num_samples)
    return random_values
#---------------------------------------------------------------------------------------------------------------------
def random_changes(column):
    num_samples = len(column)

    random_values = generate_random_values(0, 0.1, num_samples)

    # Add random values to the existing array
    result = column + random_values

    # Clip the result to ensure it's within the [0, 1] range
    result = np.clip(result, 0, 1)

    return result
#---------------------------------------------------------------------------------------------------------------------
def assign_credit_y1(row):
    baseline = row["baseline"]
    year_value = row[year]
        
    if (baseline * 0.95 <= year_value <= baseline * 1.05):
        return 1
    elif year_value > baseline * 1.05:
        return 1
    else:
        return 0

#---------------------------------------------------------------------------------------------------------------------
def assign_credit_ys(row):
    baseline = row["baseline"]
    year_value = row[year]
        
    if (baseline * 0.95 <= year_value <= baseline * 1.05):
        return 1
    else:
        return 0

#---------------------------------------------------------------------------------------------------------------------

#Indices and metrics functions
#---------------------------------------------------------------------------------------------------------------------

def shannon(MPA, gdf, grid_gdf):
    """
    This function calculates the shannon index using the "indivudualCount" colum of the OBIS data as species abundance information

    input(s):
    MPA <shapely polygon in CRS WGS84:EPSG 4326>: Marine Proteted Area of interest
    gdf <geopandas dataframe>: contains at least the name of the species, their abundance and either
                            i) the distribution polygons of each of them or (presumbaly from IUCN or local surveys),
                            ii) points denoting the observations of each species - repeated observations for the same species
    grid_gdf <geopandas dataframe>: consists of polygons of grids typically generated by the gridding function
                                  : containts at least a geometry column and a unique grid_id
    gdf_col_name <string>: corresponds to the name of the abundance information column in the gdf
    source <str>: if the data is from OBIS or from IUCN
    
    output(s):
    gdf <geopandas dataframe>: with an additional column ('shannon') containing the calculation of that index per grid
                             : or geometry
    """ 
    #Join in a gdf all the geometries within MPA
    gdf = gpd.clip(gdf.set_crs(epsg=4326, allow_override=True), MPA.set_crs(epsg=4326, allow_override=True))
    
    #Spatial join of gdf and grid_gdf
    pointInPolys = sjoin(gdf, grid_gdf, how='inner', op='intersects')
    
    # 'individualCount' refers to the number of individual organisms observed or sampled 
    # for a particular species at a particular location and time.
    pointInPolys = pointInPolys.dropna(subset='individualCount')
    pointInPolys['individualCount'] = pointInPolys['individualCount'].astype(float).astype(int)
    
    # To calculate the total number of species by grid
    N = pd.DataFrame()
    N['N'] = pointInPolys.groupby('Grid_ID').apply(lambda x: x['individualCount'].sum())
    
    new = pd.merge(pointInPolys, N, on='Grid_ID')
    
    # delete duplicate rows
    new = new.drop_duplicates(subset = 'Grid_ID', keep = 'first', inplace = False)
    
    #Calculate the Shanoon index with the information available
    new['pi'] = new['individualCount']/new['N']
    new['shannon'] = (-1)*new['pi']*np.log(new['pi'])
    
    new = new.dissolve(by='Grid_ID')
        
    new = new.drop(['geometry'], axis = 1)
        
    merge = new.merge(grid_gdf, how='right', on='Grid_ID')
    
    columns_to_keep = ['Grid_ID', 'geometry', 'shannon']
    merge = merge[columns_to_keep]
    merge.fillna(0, inplace=True)
        
    grid_gdf = gpd.GeoDataFrame(merge)
    
    return grid_gdf

#---------------------------------------------------------------------------------------------------------------------

def simpson(MPA, gdf, grid_gdf):
    """
    This function calculates the shannon index using the "indivudualCount" colum of the OBIS data as species abundance information

    input(s):
    MPA <shapely polygon in CRS WGS84:EPSG 4326>: Marine Proteted Area of interest
    gdf <geopandas dataframe>: contains at least the name of the species, their abundance and either
                            i) the distribution polygons of each of them or (presumbaly from IUCN or local surveys),
                            ii) points denoting the observations of each species - repeated observations for the same species
    grid_gdf <geopandas dataframe>: consists of polygons of grids typically generated by the gridding function
                                  : containts at least a geometry column and a unique grid_id
    gdf_col_name <string>: corresponds to the name of the abundance information column in the gdf
    source <str>: if the data is from OBIS or from IUCN
    
    output(s):
    gdf <geopandas dataframe>: with an additional columns ('simpson') containing the calculation of that index per grid
                             : or geometry
    """ 
    #Join in a gdf all the geometries within MPA
    gdf = gpd.clip(gdf.set_crs(epsg=4326, allow_override=True), MPA.set_crs(epsg=4326, allow_override=True))
    
    #Spatial join of gdf and grid_gdf
    pointInPolys = sjoin(gdf, grid_gdf, how='inner')
    
    # 'individualCount' refers to the number of individual organisms observed or sampled 
    # for a particular species at a particular location and time.
    pointInPolys = pointInPolys.dropna(subset='individualCount')
    pointInPolys['individualCount'] = pointInPolys['individualCount'].astype(float).astype(int)
    
    #To calculate the total number of species
    N = pd.DataFrame()
    N['N'] = pointInPolys.groupby('Grid_ID').apply(lambda x: x['individualCount'].sum())
        
    #where num = n(n-1)
    pointInPolys['num'] = pointInPolys['individualCount']*(pointInPolys['individualCount']-1)
        
    #Merge the datasets based on the Grid_ID
    new = pd.merge(pointInPolys, N, on='Grid_ID')
    
    # delete duplicate rows
    new = new.drop_duplicates(subset = 'Grid_ID', keep = 'first', inplace = False)
        
    #Calculate the Simpson index
    new['simpson'] = 1-((new['num'])/(new['N']*(new['N']-1)))
    
    #Sum all the value in a grid
    new = new.dissolve(by='Grid_ID', aggfunc={'simpson': 'sum'})
        
    #Delete the geometries from OBIS data
    new = new.drop(['geometry'], axis = 1)
        
    #Merge with the grid_dgf
        
    merge = new.merge(grid_gdf, how='right', on='Grid_ID')
    merge.fillna(0, inplace=True)
        
    grid_gdf = gpd.GeoDataFrame(merge)
    
    return grid_gdf

#---------------------------------------------------------------------------------------------------------------------

def species_richness(MPA, gdf, grid_gdf):
    """
    This fucntion calculates the maximum number of species that we can find in a specific area/grid from two ways:
        1. using IUCN RedList data: count the overlapping geometries of each species
        2. using OBIS data: count the total number of species' ocurrences or species' observations 
        
    inputs:
    MPA <shapely polygon in CRS WGS84:EPSG 4326>: Marine Proteted Area of interest
    df <geopandas dataframe>: contains at least the name of the species and the distribution polygons of each of them
    gdf <geopandas dataframe>: contains at least the name of the species and either
                            i) the distribution polygons of each of them or (presumbaly from IUCN or local surveys),
                            ii) points denoting the observations of each species - repeated observations for the same species
    source <str>: if the data is from OBIS or from IUCN

    output(s):
    gdf <geopandas dataframe>: with an additional column ('species_richness') containing the calculation of this factor 
                             : per grid or geometry
    """      
    #Join in a gdf all the geometries within MPA
    gdf = gpd.clip(gdf.set_crs(epsg=4326, allow_override=True), MPA.set_crs(epsg=4326, allow_override=True))
    
    if isinstance(gdf.geometry[1], shapely.geometry.point.Point):
        
        #Spatial join of gdf and grid_gdf
        pointInPolys = sjoin(gdf, grid_gdf, how='right')
        new = pointInPolys.groupby(['Grid_ID']).size().reset_index(name='count')
        
        #Added count colum in the grid geodataframe
        grid_gdf['species_richness'] = new['count']
        
        grid_gdf = gpd.GeoDataFrame(grid_gdf)
        grid_gdf.fillna(0, inplace=True)

    elif not isinstance(gdf.geometry[1], shapely.geometry.point.Point):
        
        #Count the number of overlapping geometries 
        overlap = count_overlapping_geometries(gdf)
        
        #To count how many geometries are in each grid
        merged = gpd.sjoin(overlap, grid_gdf, how='left')
        merged['n_species']= overlap['count_intersections']
        
        # Compute stats per grid cell
        #aggfunc: assigne the max value of all the geometries that dissolve
        dissolve = merged.dissolve(by="index_right", aggfunc={'n_species': 'max'})

        # put this into cell
        grid_gdf.loc[dissolve.index,'species_richness'] = dissolve.n_species.values
        grid_gdf.fillna(0, inplace=True)
    
    return grid_gdf

#---------------------------------------------------------------------------------------------------------------------

def endemism(MPA, gdf, grid_gdf):
    """
    This function calculates a distribution ratio per species, using the global distribution polygon from the IUCN RedList
    and the species distribution within the MPA to deduce a endemic ratio per each species
    
    inputs:
    MPA <shapely polygon in CRS WGS84:EPSG 4326>: region of interest or the total project area
    gdf <geopandas dataframe>: contains at least the name of the species and the distribution polygons of each of them
    grid_gdf <geopandas dataframe>: consists of polygons of grids typically generated by the gridding function
                                  : containts atleast a geometry column and a unique grid_id
    output(s):
    gdf <geopandas dataframe>: with an additional column ('endemism') containing the calculation of this factor per grid
                             : or geometry
    """
    
    #Polygons of species distribution to be clipped to MPA
    df = gpd.clip(gdf.to_crs(epsg=4326), MPA.to_crs(epsg=4326))
    
    #Calculate the portion of the area covered by each species in MPA with respect to its global distribution
    dist_ratio2 = np.round(df.area/gdf.area, decimals=4, out=None)
    
    #Calculate the log of that ratio
    log_dist2 = 1/(-np.log2(dist_ratio2)+0.1)
    
    #Add these values into the new gdf
    df["DistRatio2"] = dist_ratio2
    df["log_dist2"] = log_dist2
    
    #Function that calculates the sum of the individual log_dist2 values of all species of overlapping geometries
    overlap_endemic_v = algebraic_sum_by_grid(df, grid_gdf, 'log_dist2')
    
    overlap_endemic_v.fillna(0, inplace=True)
    
    return overlap_endemic_v
 
#---------------------------------------------------------------------------------------------------------------------

def wege(MPA, gdf, grid_gdf):
    """
    This function calculates the Weighted Endemism including Global Endangerment (WEGE) index as it is described in 
    [Farooq et al. (2020)](https://onlinelibrary.wiley.com/doi/full/10.1111/ddi.13148).
    
    inputs:
    MPA <shapely polygon in CRS WGS84:EPSG 4326>: region of interest or the total project area
    gdf <geopandas dataframe>: contains at least the name of the species, the risk category of Red List IUCN 
                             :and the distribution polygons of each of them
    grid_gdf <geopandas dataframe>: consists of polygons of grids typically generated by the gridding function
                                  : containts atleast a geometry column and a unique grid_id
               
    output(s):
    gdf <geopandas dataframe>: with an additional column ('wege') containing the calculation of this factor per grid
                             : or geometry
    """

    def extinction_risk(cat: str = None) -> float:
        """
        Calculates extinction risk (ER) for species following Farooq et al. (2020)
        We assign probability of extinction for each IUCN category using extinction probabilities 
        from Table S2 in supplemental material of Davis et al (2018).
        Here we use use IUCN50 values, same as Farooq et al. (2020).
    
        Extinction risk for data deficiient (DD) category is assigned the vulnerable (VU) probability, 
        see Bland et al. (2015) for explanation.
    
        Args:
            cat (str): IUCN category
                - DD = Data Deficient
                - LC = Least Concern
                - NT = Near Threatened
                - VU = Vulnerable
                - EN = Endangered
                - CR = Critically Endangered
                - EW = Extinct in the wild 
                - EX = Extinct
            
        Returns:
            float: probability of extinction
            
        References:
            Bland et al. (2015) "Predicting the conservation status of data-deficient species" 
                https://doi.org/10.1111/cobi.12372
            Davis et al. (2018) "Mammal diversity will take millions of years to recover from the current biodiversity crisis"
                https://doi.org/10.1073/pnas.1804906115
            Farooq et al. (2020) "WEGE: A new metric for ranking locations for biodiversity conservation" 
                https://doi.org/10.1111/ddi.13148
        """
        cat_to_risk = dict(
            DD=0.0513, # using Bland et al. (2015) assumption
            LC=0.0009,
            NT=0.0071,
            VU=0.0513,
            EN=0.4276,
            CR=0.9688,
            EW=1.0,
            EX=1.0
        )

        if cat_to_risk.get(cat) is None:
            raise ValueError("Invalid value for 'cat', expected one of 'DD', 'LC', 'NT', 'VU', 'EN', 'CR', EW', 'EX'")

        return cat_to_risk.get(cat)
    
    #To extract the MPA area value
    MPA_area = MPA.to_crs(epsg=4326).area[0]
     
    #Polygons of species distribution to be clipped to MPA
    df = gpd.clip(gdf.set_crs(epsg=4326, allow_override=True), MPA.set_crs(epsg=4326, allow_override=True))

    #Calculate the portion of the area covered by each species in MPA with respect to the MPA area
    #This is called "Weighted Endemism" factor
    we = np.round(df.area/MPA_area, decimals=4, out=None)
    sq_we = we**(0.5)

    #Add this information into the new gdf
    df['we'] = we
    df['sq_we'] = sq_we

    # replaces long RedList name with two-letter code
    long_to_short = {
        'Data Deficient':'DD',
        'Least Concern':'LC',
        'Near Threatened':'NT',
        'Vulnerable':'VU',
        'Endangered':'EN',
        'Critically Endangered':'CR',
        'Extinct In The Wild':'EW',
        'Extinct':'EX'
    }

    df['redlistCat'] = df['redlistCat'].replace(long_to_short)

    #List of extinction probabilities for each species
    df['ER'] = [extinction_risk(cat) for cat in df['redlistCat']]

    #Calculate the "WEGE factor" individually
    df['wege_i'] = df['sq_we']*df['ER']

    #Function that calculates the sum of the individual wege_i values of all species of overlapping geometries
    overlap_wege_v =algebraic_sum_by_grid(df, grid_gdf, 'wege_i')

    overlap_wege_v.fillna(0, inplace=True)
    
    return overlap_wege_v
    
#---------------------------------------------------------------------------------------------------------------------    

def habitat_accounting(MPA, grid_gdf, path_EFG):
    """
    This fucntion calculates the maximum number of EFG from the IUCN Global Ecosystem Typology that we can find in a specific area/grid 
    
    inputs:
    MPA <shapely polygon in CRS WGS84:EPSG 4326>: region of interest or the total project area
    list_EFG <list>: consist in a list with path location of each EFG file 
    grid_gdf <geopandas dataframe>: consists of polygons of grids typically generated by the gridding function
                                  : containts atleast a geometry column and a unique grid_id
    output(s):
    gdf <geopandas dataframe>: with an additional column ('habitat_survey') containing the calculation of this factor
                             : per grid or geometry
    """
    # clip the EFG layers within ACMC polygon
    joined = Clip_EFG(MPA, path_EFG)
    
    #Count the number of overlappong geometries from joined gdf
    overlap_geo = accounting_by_grid(joined, grid_gdf)
    
    return overlap_geo

#---------------------------------------------------------------------------------------------------------------------
#Modulating Factor Functions
#---------------------------------------------------------------------------------------------------------------------

def mbu_shannon_index(MPA, gdf, grid_gdf, source):
    """
    This functions uses the Shannon Index to calculate marine species diversity and richness per grid or given area and c
    onverts these numbers into MBUs.
    It calls internally the Shannon Index function to do the calculations
    
    input(s):
    MPA <shapely polygon in CRS WGS84:EPSG 4326>: Marine Proteted Area of interest
    gdf <geopandas dataframe>: contains at least the name of the species and either
                            i) the distribution polygons of each of them or (presumbaly from IUCN or local surveys),
                            ii) points denoting the observations of each species - repeated observations for the same species
    grid_gdf <geopandas dataframe>: consists of polygons of grids typically generated by the gridding function
                                  : containts at least a geometry column and a unique grid_id
    source <str>: if the data is from OBIS or from IUCN
    crs_transformation_kms: coordinate reference system transformation applied to the MPA in meters 
    
    output(s):
    gdf <geopandas dataframe>: with an additional column ('mbu_shannon_index') containing the calculation of MBUs with this                                
                             :factor information per grid or geometry
    """
    
    if source == 'OBIS':
        
        #Shannon Index calculation
        df1 = shannon(MPA, gdf, grid_gdf)
        
        #Normalization factor
        Norm_factor1 = df1['shannon']/df1['shannon'].max()
        
        #Convert area from degrees to square kilometers
        #df1['area_sqkm'] = (df1.to_crs(crs=crs_transformation_kms).area)*10**(-6)

        #Calculate the MBUS from this MF
        df1['mbu_shannon_index'] = Norm_factor1
        
    elif source == 'IUCN':
        print('The Shannon Index - Modulating Factor is not available to IUCN data')
        
        #Calculate the MBUS from this MF
        df1['mbu_shannon_index'] = 'NA'
        
    else:
        raise ValueError("Unsupported source: {}".format(source))

    return df1

#---------------------------------------------------------------------------------------------------------------------

def mbu_simpson_index(MPA, gdf, grid_gdf, source):
    """
    This function uses the Simpson Index to quantify biodiversity in a specific habitat, in this the quantification is per grid or 
    given area, and converts these numbers into MBUs.
    It calls internally the Simpson function to do the calculations
    
    input(s):
    MPA <shapely polygon in CRS WGS84:EPSG 4326>: Marine Proteted Area of interest
    gdf <geopandas dataframe>: contains at least the name of the species and either
                            i) the distribution polygons of each of them or (presumbaly from IUCN or local surveys),
                            ii) points denoting the observations of each species - repeated observations for the same species
    grid_gdf <geopandas dataframe>: consists of polygons of grids typically generated by the gridding function
                                  : containts at least a geometry column and a unique grid_id
    source <str>: if the data is from OBIS or from IUCN
    crs_transformation_kms: coordinate reference system transformation applied to the MPA in meters 
    
    output(s):
    gdf <geopandas dataframe>: with an additional column ('mbu_biodiversity_score') containing the calculation of MBUs with this                                
                             :factor information per grid or geometry
    """
    
    if source == 'OBIS':
        
        #Simpson Index calculation
        df1 = simpson(MPA, gdf, grid_gdf)
        
        #Normalization factor
        Norm_factor1 = df1['simpson']/df1['simpson'].max()
        
        #Convert area from degrees to square kilometers
        #df1['area_sqkm'] = (df1.to_crs(crs=crs_transformation_kms).area)*10**(-6)

        #Calculate the MBUS from this MF
        df1['mbu_simpson_index'] = Norm_factor1
        
    elif source == 'IUCN':
        print('The Simpsion Index - Modulating Factor is not available to IUCN data')
        
        #Calculate the MBUS from this MF
        df1['mbu_simpsion_index'] = 'NA'
        
    else:
        raise ValueError("Unsupported source: {}".format(source))

    return df1

#---------------------------------------------------------------------------------------------------------------------

def mbu_species_richness(MPA, gdf, grid_gdf):
    """
    This function calculates the amount of MBUs from the species richness metric and converts these 
    numbers into MBUs in a given area in sqd kms.
    It calls internally the Species Richness function to do the calculations
    
    input(s):
    MPA <shapely polygon in CRS WGS84:EPSG 4326>: region of interest or the total project area
    gdf <geopandas dataframe>: contains at least the name of the species and either
                            i) the distribution polygons of each of them or (presumbaly from IUCN or local surveys),
                            ii) points denoting the observations of each species - repeated observations for the same species
    grid_gdf <geopandas dataframe>: consists of polygons of grids typically generated by the gridding function
                                  : containts at least a geometry column and a unique grid_id
    source <str>: if the data is from OBIS or from IUCN
    crs_transformation_kms: coordinate reference system transformation applied to the MPA in meters 
    
    output(s):
    gdf <geopandas dataframe>: with an additional column ('mbu_species_richness') containing the calculation of MBUs with this
                             : factor information per grid or geometry
    """
    
    #Species Richness calculation
    df1 = species_richness(MPA, gdf, grid_gdf)
        
    #Normalization factor
    Norm_factor1 = df1['species_richness']/df1['species_richness'].max()
        
    #Convert area from degrees to square kilometers
    #df1['area_sqkm'] = (df1.to_crs(crs=crs_transformation_kms).area)*10**(-6)

    #Calculate the MBUS from this MF
    df1['mbu_species_richness'] = Norm_factor1
        
    return df1
    
#---------------------------------------------------------------------------------------------------------------------
    
def mbu_endemism(MPA, gdf, grid_gdf, source):
    """
    This function calculates the amount of MBUs from the Endemic index and converts these numbers into 
    MBUs in a given area in sqd kms.
    It calls internally the Endemism function to do the calculations
    
    input(s):
    MPA <shapely polygon in CRS WGS84:EPSG 4326>: region of interest or the total project area
    gdf <geopandas dataframe>: contains at least the name of the species, their abundance and either
                         i) the distribution polygons of each of them or (presumbaly from IUCN or local surveys)
                        ii) points denoting the observations of each species - repeated observations for the same species
    grid_gdf <geopandas dataframe>: consists of polygons of grids typically generated by the gridding function
                                  : containts at least a geometry column and a unique grid_id
    source <str>: if the data is from OBIS or from IUCN											
    crs_transformation_kms: coordinate reference system transformation applied to the MPA in meters 
    
    output(s):
    gdf <geopandas dataframe>: with an additional column ('mbu_endemism') containing the calculation of MBUs with this
                             :factor information per grid or geometry
    """
    
    if source == 'OBIS':
        print('Endemic Modulating Factor is not available to OBIS data')
        
        #Calculate the MBUS from this MF
        df1['mbu_endemism'] = 'NA'
        
    elif source == 'IUCN':

        #Endemic factor calculation
        df1 = endemism(MPA, gdf, grid_gdf)

        #Normalization factor
        Norm_factor1 = df1['result']/df1['result'].max()

        #Calculate the MBUS from this MF
        df1['mbu_endemism'] = Norm_factor1
        
    else:
        raise ValueError("Unsupported source: {}".format(source))
    
    return df1

#--------------------------------------------------------------------------------------------------------------------- 
   
def mbu_wege(MPA, gdf, grid_gdf, source):
    """
    This function calculates the amount of MBUs from the WEGE index and converts these numbers into MBUs in a 
    given area in sqd kms.
    It calls internally the WEGE function to do the calculations
    
    input(s):
    MPA <shapely polygon in CRS WGS84:EPSG 4326>: region of interest or the total project area
    gdf <geopandas dataframe>: contains at least the name of the species, their abundance and either
                         i) the distribution polygons of each of them or (presumbaly from IUCN or local surveys)
                        ii) points denoting the observations of each species - repeated observations for the same species
    grid_gdf <geopandas dataframe>: consists of polygons of grids typically generated by the gridding function
                                  : containts at least a geometry column and a unique grid_id
    source <str>: if the data is from OBIS or from IUCN											
    crs_transformation_kms: coordinate reference system transformation applied to the MPA in meters 
    
    output(s):
    gdf <geopandas dataframe>: with an additional column ('mbu_wege') containing the calculation of MBUs with this                                              
                             :factor information per grid or geometry
    """
    if source == 'OBIS':
        print('WEGE Modulating Factor is not available to OBIS data')
        
        #Calculate the MBUS from this MF
        df1['mbu_endemism'] = 'NA'
        
    elif source == 'IUCN':
    
        #Wege factor calculation
        df1 = wege(MPA, gdf, grid_gdf)

        #Normalization factor
        Norm_factor1 = df1['result']/df1['result'].max()

        #Calculate the MBUS from this MF
        df1['mbu_wege'] = Norm_factor1
        
    else:
        raise ValueError("Unsupported source: {}".format(source))
    
    return df1

#---------------------------------------------------------------------------------------------------------------------

def mbu_habitats_survey(MPA, grid_gdf, path_EFG):
    """
    This function calculates the amount of MBUs from the Habitats Survey calculation and converts these numbers into MBUs in 
    a given area in sqd kms.
    It calls internally the habitats Survey function to do the calculations
    
    MPA <shapely polygon in CRS WGS84:EPSG 4326>: region of interest or the total project area
    path_EFG <list>: consist in a list with path location of each EFG file 
    grid_gdf <geopandas dataframe>: consists of polygons of grids typically generated by the gridding function
                                  : containts at least a geometry column and a unique grid_id
    crs_transformation_kms: coordinate reference system transformation applied to the MPA in meters 
    
    output(s):
    gdf <geopandas dataframe>: with an additional column ('mbu_habitats_survey') containing the calculation of MBUs with this                                  
                             :factor information per grid or geometry
    """
    
    #Wege factor calculation
    df1 = habitat_accounting(MPA, grid_gdf, path_EFG)
    
    #Normalization factor
    Norm_factor1 = df1['count']/df1['count'].max()

    #Calculate the MBUS from this MF
    df1['mbu_habitats_survey'] = Norm_factor1
    
    return df1
    