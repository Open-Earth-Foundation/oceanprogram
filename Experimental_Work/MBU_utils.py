import os
import glob
import math
import numpy as np
import pandas as pd

import shapely
import geopandas as gpd
from geopandas.tools import sjoin
from shapely.geometry import Polygon, Point, box
from shapely.ops import linemerge, unary_union, polygonize

#---------------------------------------------------------------------------------------------------------------------
#High Level Helper Functions
#---------------------------------------------------------------------------------------------------------------------

def create_grid(roi, grid_shape="hexagon", grid_size_deg=1.):
    """
input(s):
roi <shapely polygon>: region of interest or the total project area
grid_size_m <int>: size of the grid in degrees
grid_shape <str>: either "square" or "hexagon"

output(s):
geopandas frame
    """

    # Slightly displace the minimum and maximum values of the feature extent by creating a buffer
    # This decreases likelihood that a feature will fall directly on a cell boundary (in between two cells)
    # Buffer is projection dependent (due to units)
    #feature = feature.buffer(20) #This is increase the boundary by 20 degrees!
    # print("no buffer")

    # Get extent of buffered input feature
    min_x, min_y, max_x, max_y = roi.total_bounds


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

def count_overlapping_geometries(gdf):
    #main source: https://gis.stackexchange.com/questions/387773/count-overlapping-features-using-geopandas
    """
    input(s):
    gdf <geopandas dataframe>: consists of polygons

    output(s):
    gdf <geopandas dataframe>:
    """
    
    #Get the name of the column containing the geometries
    geom_col = gdf.geometry.name
    
    # Setting up a single piece that will be split later
    input_parts = [gdf.unary_union.buffer(0)]
    
    # Finding all the "cutting" boundaries. Note: if the input GDF has 
    # MultiPolygons, it will treat each of the geometry's parts as individual
    # pieces.
    cutting_boundaries = []
    for i, row in gdf.iterrows():
        this_row_geom = row[geom_col]
        this_row_boundary = this_row_geom.boundary
        if this_row_boundary.type[:len('multi')].lower() == 'multi':
            cutting_boundaries = cutting_boundaries + list(this_row_boundary.geoms)
        else:
            cutting_boundaries.append(this_row_boundary)
    
    # Split the big input geometry using each and every cutting boundary
    for boundary in cutting_boundaries:
        splitting_results = []
        for j,part in enumerate(input_parts):
            new_parts = list(shapely.ops.split(part, boundary).geoms)
            splitting_results = splitting_results + new_parts
        input_parts = splitting_results
    
    # After generating all of the split pieces, create a new GeoDataFrame
    new_gdf = gpd.GeoDataFrame({'id':range(len(splitting_results)),
                                geom_col:splitting_results,
                                },
                               crs=gdf.crs,
                               geometry=geom_col)
    
    # Find the new centroids.
    new_gdf['geom_centroid'] = new_gdf.centroid
    
    # Starting the count at zero
    new_gdf['count_intersections'] = 0
    
    # For each of the `new_gdf`'s rows, find how many overlapping features 
    # there are from the input GDF.
    for i,row in new_gdf.iterrows():
        new_gdf.loc[i,'count_intersections'] = gdf.intersects(row['geom_centroid']).astype(int).sum()
        pass
    
    # Dropping the column containing the centroids
    new_gdf = new_gdf.drop(columns=['geom_centroid'])[['id','count_intersections',geom_col]]
    
    return new_gdf

#---------------------------------------------------------------------------------------------------------------------

#This function calculates the sum of all values of an interest colum of overlapping geometries 
def map_algebra(gdf, gdf_col_name, operation):
    """
    This function calculates the sum of all values of an interest colum of overlapping geometries 
    Input:
    gdf <geopandas dataframe>: consists of polygons
    gdf_col_name <string>: column name that has the value that we want to apply the algebra operation
    operation <string>: algebra operation
                      : options ('sum','mean','max','min','prod')

    Output:
    gdf <geopandas dataframe>: consists of polygons with a new colum with a summation of interest values
    """
    #main source: https://stackoverflow.com/questions/65073549/combine-and-sum-values-of-overlapping-polygons-in-geopandas

    #The explode() method converts each element of the specified column(s) into a row
    #This is useful if there are multipolygons
    new_gdf = gdf.explode('geometry')
    new_gdf['new_colum'] = new_gdf[str(gdf_col_name)]

    #convert all polygons to lines and perform union
    lines = unary_union(linemerge([geom.exterior for geom in new_gdf.geometry]))

    #convert again to (smaller) intersecting polygons and to geodataframe
    polygons = list(polygonize(lines))
    intersects = gpd.GeoDataFrame({'geometry': polygons}, crs="EPSG:4326")

    #to fix invalid geometries
    intersects['geometry'] = intersects['geometry'].buffer(0)

    #Perform sjoin with original geoframe to get overlapping polygons.
    #Afterwards group per intersecting polygon to perform (arbitrary) aggregation
    
    if operation == 'sum':
        intersects['algebra_overlaps'] = (intersects
                                      .sjoin(new_gdf, predicate='within')
                                      .reset_index()
                                      .groupby(['level_0', 'index_right0'])
                                      .head(1)
                                      .groupby('level_0')
                                      .new_colum.sum())
    elif operation == 'mean':
        intersects['algebra_overlaps'] = (intersects
                                      .sjoin(new_gdf, predicate='within')
                                      .reset_index()
                                      .groupby(['level_0', 'index_right0'])
                                      .head(1)
                                      .groupby('level_0')
                                      .new_colum.mean())
    elif operation == 'max':
        intersects['algebra_overlaps'] = (intersects
                                      .sjoin(new_gdf, predicate='within')
                                      .reset_index()
                                      .groupby(['level_0', 'index_right0'])
                                      .head(1)
                                      .groupby('level_0')
                                      .new_colum.max())
    elif operation == 'min':
        intersects['algebra_overlaps'] = (intersects
                                      .sjoin(new_gdf, predicate='within')
                                      .reset_index()
                                      .groupby(['level_0', 'index_right0'])
                                      .head(1)
                                      .groupby('level_0')
                                      .new_colum.min())
    elif operation == 'prod':
        intersects['algebra_overlaps'] = (intersects
                                      .sjoin(new_gdf, predicate='within')
                                      .reset_index()
                                      .groupby(['level_0', 'index_right0'])
                                      .head(1)
                                      .groupby('level_0')
                                      .new_colum.prod())
    else:
        raise ValueError("Unsupported operation: {}".format(operation))
    
    return intersects

#---------------------------------------------------------------------------------------------------------------------

def Clip_EFG(roi, path_EFG):
    """
    Inputs: 
    - roi <shapely polygon in CRS WGS84:EPSG 4326>: region of interest or the total project area
    - path_EFG <list>: consists of a list of EFG polygons' paths to be reference internally
    
    Outputs:
    - gdf <geopandas dataframe>: consists of a list of EFG polygons cut with respect to roi
    """
    
    Geometry = []

    for x in range(len(path_EFG)):
        #Reading each EFG
        gdf = gpd.read_file(str(path_EFG[x]))
    
        #Choosing the crs
        gdf = gdf.set_crs(epsg=4326, allow_override=True)
    
        #We want only the polygons within our study place
        clip = gpd.clip(gdf, roi)
        clip = clip.reset_index()
    
        #Selecting the 'geometry' column
        geo = clip.geometry

        Geometry.append(geo)
        
    joined = gpd.GeoDataFrame(pd.concat(Geometry, ignore_index=True))
    
    return joined

#---------------------------------------------------------------------------------------------------------------------
#Indices and metrics functions
#---------------------------------------------------------------------------------------------------------------------

def shannon(roi, gdf, grid_gdf, gdf_col_name, source):
    """
    input(s):
    roi <shapely polygon in CRS WGS84:EPSG 4326>: region of interest or the total project area
    gdf <geopandas dataframe>: contains at least the name of the species, their abundance and either
                            i) the distribution polygons of each of them or (presumbaly from IUCN or local surveys),
                            ii) points denoting the observations of each species - repeated observations for the same species
    grid_gdf <geopandas dataframe>: consists of polygons of grids typically generated by the gridding function
                                  : containts at least a geometry column and a unique grid_id
    gdf_col_name <string>: corresponds to the name of the abundance information column in the gdf
    source <str>: if the data is from obis or from IUCN
    
    output(s):
    gdf <geopandas dataframe>: with an additional column ('shannon') containing the calculation of that index per grid
                             : or geometry
    """
    if source == 'obis':
        # convert OBIS dataframe to geodataframe
        obis = gpd.GeoDataFrame(gdf, geometry=gpd.points_from_xy(df.decimalLongitude, df.decimalLatitude))
        
        #Join in a gdf all the geometries within roi
        gdf = gpd.clip(obis.set_crs(epsg=4326, allow_override=True), roi.set_crs(epsg=4326, allow_override=True))
        
        #Spatial join of gdf and grid_gdf
        pointInPolys = sjoin(obis, grid_gdf, how='inner')
        
        #To count the total abundance number from all the species in a grid
        N = pd.DataFrame()
        N['N'] = pointInPolys.groupby('Grid_ID').apply(lambda x: x[str(gdf_col_name)].sum())
        
        #Merge the datasets based on the Grid_ID
        new = pd.merge(pointInPolys, N, on='Grid_ID')
        
        #Calculate the Shannon index
        new['pi'] = new['abundance']/new['N']
        new['shannon'] = (-1)*new['pi']*np.log(new['pi'])
    
        #Sum all the value in a grid
        new = new.dissolve(by='Grid_ID', aggfunc={'shannon': 'sum'})
        
        #Delete the geometries from obis data
        new = new.drop(['geometry'], axis = 1)
        
        #Merge with the grid_dgf
        grid_gdf = new.merge(grid, how='right', on='Grid_ID')

    elif source == 'IUCN':
        #Join in a gdf all the geometries within ROI
        gdf = gpd.clip(gdf.set_crs(epsg=4326, allow_override=True), roi.set_crs(epsg=4326, allow_override=True))

        #This function calculates the sum of all abundances of overlapping species
        overlap = map_algebra(gdf, gdf_col_name, 'sum')

        #Merged the overlap values of overlapping geometries with the grid gdf
        merged = gpd.sjoin(overlap, grid_gdf, how='left')
        merged['n_value'] = overlap['algebra_overlaps']
    
        #for Shannon calculations
        #calculates the Shannon Index per grid cell and its corresponding MBU value
        #pi = (n/N): where n is the abundance number per species and N is the total abundance number in the dataset 
    
        #Calculate the pi value per row
        pi = merged['n_value']/np.sum(merged['n_value'])
        pi = pi.fillna(0)
        merged['pilogpi'] = pi*np.log(pi)

        #Dissolve the DataFrame by 'index_right' and aggregate using the calculated Shannon entropy
        dissolve = merged.dissolve(by="index_right", aggfunc={'pilogpi': 'sum'})
    
        #Calculate the Shannon index per grid
        dissolve['pilogpi'] = (-1)*dissolve['pilogpi']

        #Put this into cell
        grid_gdf.loc[dissolve.index, 'Shannon'] = dissolve.pilogpi.values
        
    else:
        raise ValueError("Unsupported source: {}".format(operation))
    
    return grid_gdf

#---------------------------------------------------------------------------------------------------------------------

def simpson(roi, gdf, grid_gdf, gdf_col_name, source):
    """
    input(s):
    roi <shapely polygon in CRS WGS84:EPSG 4326>: region of interest or the total project area
    gdf <geopandas dataframe>: contains at least the name of the species, their abundance and either
                            i) the distribution polygons of each of them or (presumbaly from IUCN or local surveys),
                            ii) points denoting the observations of each species - repeated observations for the same species
    grid_gdf <geopandas dataframe>: consists of polygons of grids typically generated by the gridding function
                                  : containts at least a geometry column and a unique grid_id
    gdf_col_name <string>: corresponds to the name of the abundance information column in the gdf
    source <str>: if the data is from obis or from IUCN
    
    output(s):
    gdf <geopandas dataframe>: with an additional columns ('simpson') containing the calculation of that index per grid
                             : or geometry
    """

    if source == 'obis':
        # convert OBIS dataframe to geodataframe
        obis = gpd.GeoDataFrame(gdf, geometry=gpd.points_from_xy(df.decimalLongitude, df.decimalLatitude))
        
        #Join in a gdf all the geometries within roi
        gdf = gpd.clip(obis.set_crs(epsg=4326, allow_override=True), roi.set_crs(epsg=4326, allow_override=True))
        
        #Spatial join of gdf and grid_gdf
        pointInPolys = sjoin(df2, grid, how='inner')
        
        #To count the total abundance number from all the species in a grid
        N = pd.DataFrame()
        N['N'] = pointInPolys.groupby('Grid_ID').apply(lambda x: x[str(gdf_col_name)].sum())
        
        #where num = n(n-1)
        pointInPolys['num'] = pointInPolys['abundance']*(pointInPolys['abundance']-1)
        
        #Merge the datasets based on the Grid_ID
        new = pd.merge(pointInPolys, N, on='Grid_ID')
        
        #Calculate the Simpson index
        new['simpson'] = 1-((new['num'])/(new['N']*(new['N']-1)))
    
        #Sum all the value in a grid
        new = new.dissolve(by='Grid_ID', aggfunc={'simpson': 'sum'})
        
        #Delete the geometries from obis data
        new = new.drop(['geometry'], axis = 1)
        
        #Merge with the grid_dgf
        grid_gdf = new.merge(grid, how='right', on='Grid_ID')

    elif source == 'IUCN':
        #Join in a gdf all the geometries within ROI
        gdf = gpd.clip(gdf.set_crs(epsg=4326, allow_override=True), roi)

        #This function calculates the sum of all abundances of overlapping species
        overlap = map_algebra(gdf, gdf_col_name, 'sum')

        #Merged the overlap values of overlapping geometries with the grid gdf
        merged = gpd.sjoin(overlap, grid_gdf, how='left')
        merged['n_value'] = overlap['algebra_overlaps']
        
        #for Simpson calculations
        #Calculate the Simpson Index per grid cell and its corresponding MBU value
    
        #Calculate the numerator and denominator needed per row
        num = merged['n_value']*(merged['n_value']-1)
        den = np.sum(merged['n_value'])*(np.sum(merged['n_value'])-1)
        merged['num'] = num

        #Dissolve the DataFrame by 'index_right' and aggregate using the calculated Shannon entropy
        dissolve = merged.dissolve(by="index_right", aggfunc={'num': 'sum'})
    
        #Calculate the Shannon index per grid
        dissolve['simpson'] = 1-(dissolve['num']/den)

        #Put this into cell
        grid_gdf.loc[dissolve.index, 'Simpson'] = dissolve.simpson.values
        
    else:
        raise ValueError("Unsupported source: {}".format(operation))
    
    return grid_gdf

#---------------------------------------------------------------------------------------------------------------------

def species_richness(roi, df, grid_gdf, source):
    """
    inputs:
    roi <shapely polygon in CRS WGS84:EPSG 4326>: region of interest or the total project area
    df <geopandas dataframe>: contains at least the name of the species and the distribution polygons of each of them
    gdf <geopandas dataframe>: contains at least the name of the species and either
                            i) the distribution polygons of each of them or (presumbaly from IUCN or local surveys),
                            ii) points denoting the observations of each species - repeated observations for the same species
    source <str>: if the data is from obis or from IUCN

    output(s):
    gdf <geopandas dataframe>: with an additional column ('species_richness') containing the calculation of this factor 
                             : per grid or geometry
    """
    
    if source == 'obis':
        # convert OBIS dataframe to geodataframe
        obis = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.decimalLongitude, df.decimalLatitude))
        
        #Join in a gdf all the geometries within ACMC
        gdf = gpd.clip(obis.set_crs(epsg=4326, allow_override=True), roi.set_crs(epsg=4326, allow_override=True))
        
        #Spatial join of gdf and grid_gdf
        pointInPolys = sjoin(gdf, grid_gdf, how='right')
        new = pointInPolys.groupby(['Grid_ID']).size().reset_index(name='count')
        
        #Added count colum in the grid geodataframe
        grid_gdf['species_richness'] = new['count']

    elif source == 'IUCN':
        #Join in a gdf all the geometries within ACMC
        df2 = gpd.clip(df.set_crs(epsg=4326, allow_override=True), roi.set_crs(epsg=4326, allow_override=True))
        
        #Count the number of overlapping geometries 
        overlap = count_overlapping_geometries(df2)
        
        #To count how many geometries are in each grid
        merged = gpd.sjoin(overlap, grid_gdf, how='left')
        merged['n_species']= overlap['count_intersections']
        
        # Compute stats per grid cell
        #aggfunc: assigne the max value of all the geometries that dissolve
        dissolve = merged.dissolve(by="index_right", aggfunc={'n_species': 'max'})

        # put this into cell
        grid.loc[dissolve.index,'species_richness'] = dissolve.n_species.values
        
    else:
        raise ValueError("Unsupported source: {}".format(operation))
    
    return grid_gdf

#---------------------------------------------------------------------------------------------------------------------

def endemism(roi, gdf, grid_gdf):
    """
    inputs:
    roi <shapely polygon in CRS WGS84:EPSG 4326>: region of interest or the total project area
    gdf <geopandas dataframe>: contains at least the name of the species and the distribution polygons of each of them
    grid_gdf <geopandas dataframe>: consists of polygons of grids typically generated by the gridding function
                                  : containts atleast a geometry column and a unique grid_id
    transform <str>: either "none" or "square-root" or "log"
                   : if "square-root" then it is square-rooted
                   : else 1/(-Log2(score))
               
    output(s):
    gdf <geopandas dataframe>: with an additional column ('endemism') containing the calculation of this factor per grid
                             : or geometry
    """
    
    #Polygons of species distribution to be clipped to roi
    df2 = gpd.clip(gdf.set_crs(epsg=4326, allow_override=True), roi)
    
    #Calculate the portion of the area covered by each species in roi with respect to its global distribution
    dist_ratio2 = np.round(df2.area/gdf.area, decimals=4, out=None)
    
    #Calculate the log of that ratio
    log_dist2 = 1/(-np.log2(dist_ratio2)+0.1)
    
    #Add these values into the new gdf
    df2["DistRatio2"] = dist_ratio2
    df2["log_dist2"] = log_dist2
    
    #Function that calculates the sum of the individual log_dist2 values of all species of overlapping geometries
    overlap_endemic_v = map_algebra(df2, 'log_dist2', 'sum')
    
    #Merged the log_dist2 values of overlapping geometries with the grid gdf
    merged = gpd.sjoin(overlap_endemic_v, grid_gdf, how='left')
    merged['n_value']= overlap_endemic_v['algebra_overlaps']

    # Compute stats per grid cell
    #aggfunc: sum the values of all the geometries that dissolve
    dissolve = merged.dissolve(by="index_right", aggfunc={'n_value': 'sum'})

    #Put this into cell
    grid_gdf.loc[dissolve.index, 'endemism'] = dissolve.n_value.values
    
    return grid_gdf
 
#---------------------------------------------------------------------------------------------------------------------

def wege(roi, gdf, grid_gdf):
    """
    inputs:
    roi <shapely polygon in CRS WGS84:EPSG 4326>: region of interest or the total project area
    gdf <geopandas dataframe>: contains at least the name of the species, the risk category of Red List IUCN 
                             :and the distribution polygons of each of them
    grid_gdf <geopandas dataframe>: consists of polygons of grids typically generated by the gridding function
                                  : containts atleast a geometry column and a unique grid_id
    transform <str>: either "none" or "square-root" or "log"
                   : if "square-root" then it is square-rooted
                   : else 1/(-Log2(score))
               
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
    
    #To extract the roi area value
    roi_area = roi.area[0]
    
    #Polygons of species distribution to be clipped to roi
    df = gpd.clip(gdf.set_crs(epsg=4326, allow_override=True), roi)
    
    #Calculate the portion of the area covered by each species in roi with respect to the roi area
    #This is called "Weighted Endemism" factor
    we = np.round(df.area/roi_area, decimals=4, out=None)
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
    overlap_wege_v = map_algebra(df, 'wege_i', 'sum')
    
    #Merged the log_dist2 values of overlapping geometries with the grid gdf
    merged = gpd.sjoin(overlap_wege_v, grid_gdf, how='left')
    merged['n_value']= overlap_wege_v['algebra_overlaps']

    # Compute stats per grid cell
    #aggfunc: sum the values of all the geometries that dissolve
    dissolve = merged.dissolve(by="index_right", aggfunc={'n_value': 'sum'})

    #Put this into cell
    grid_gdf.loc[dissolve.index, 'wege'] = dissolve.n_value.values
    
    return grid_gdf
    
#---------------------------------------------------------------------------------------------------------------------    

def habitats_survey(roi, grid_gdf, path_EFG):
    """
    inputs:
    roi <shapely polygon in CRS WGS84:EPSG 4326>: region of interest or the total project area
    list_EFG <list>: consist in a list with path location of each EFG file 
    grid_gdf <geopandas dataframe>: consists of polygons of grids typically generated by the gridding function
                                  : containts atleast a geometry column and a unique grid_id
    output(s):
    gdf <geopandas dataframe>: with an additional column ('mbu_habitat_survey') containing the number
                             : of units for that grid or geometry
    """
    
    #Join in a gdf all the geometries within ROI
    joined = Clip_EFG(roi, path_EFG)
    
    #Count the number of overlappong geometries from joined gdf
    overlap_geo = count_overlapping_geometries(joined)
    
    #This is to count how many geometries are in each grid 
    merged = gpd.sjoin(overlap_geo, grid_gdf, how='left')
    merged['n_habitats']= overlap_geo['count_intersections']

    # Compute stats per grid cell
    #aggfunc: select the max value of all the geometries that dissolve
    dissolve = merged.dissolve(by="index_right", aggfunc={'n_habitats': 'max'})

    # put this into cell
    grid_gdf.loc[dissolve.index, 'habitats_survey'] = dissolve.n_habitats.values
    
    return grid_gdf

#---------------------------------------------------------------------------------------------------------------------
#Modulating Factor Functions
#---------------------------------------------------------------------------------------------------------------------

def mbu_biodiversity_score(roi, gdf, grid_gdf, gdf_col_name, crs_transformation_kms):
    """
    input(s):
    roi <shapely polygon in CRS WGS84:EPSG 4326>: region of interest or the total project area
    gdf <geopandas dataframe>: contains at least the name of the species, the distribution polygons of each of them 
                             :and their abundance
    grid_gdf <geopandas dataframe>: consists of polygons of grids typically generated by the gridding function
                                  : containts at least a geometry column and a unique grid_id
    gdf_col_name <string>: corresponds to the name of the abundance information column in the gdf
    crs_transformation_kms: coordinate reference system transformation applied to the roi in meters 
    
    output(s):
    gdf <geopandas dataframe>: with an additional column ('mbu_biodiversity_score') containing the calculation of that index per grid
                             : or geometry
    """
    
    #Shannon Index calculation
    df = shannon(roi, gdf, grid_gdf, gdf_col_name)
    
    #Simpson Index calculation
    df = simpson(roi, gdf, grid_gdf, gdf_col_name)
    
    #Normalization factor
    Norm_factor1 = df['Shannon']/df['Shannon'].max()
    Norm_factor2 = df['Simpson']/df['Simpson'].max()
    
    #Convert area from degrees to square kilometers
    df['area_sqkm'] = (df.to_crs(crs=crs_transformation_kms).area)*10**(-6)

    #Calculate the MBUS from this MF
    df['mbu_biodiversity_score'] = Norm_factor1*df['area_sqkm'] + Norm_factor2*df['area_sqkm']
    
    return df

#---------------------------------------------------------------------------------------------------------------------

def mbu_species_richness(roi, gdf, grid_gdf, source, crs_transformation_kms):
    """
    input(s):
    roi <shapely polygon in CRS WGS84:EPSG 4326>: region of interest or the total project area
    gdf <geopandas dataframe>: contains at least the name of the species, the distribution polygons of each of them 
                             :and their abundance
    grid_gdf <geopandas dataframe>: consists of polygons of grids typically generated by the gridding function
                                  : containts at least a geometry column and a unique grid_id
    crs_transformation_kms: coordinate reference system transformation applied to the roi in meters 
    
    output(s):
    gdf <geopandas dataframe>: with an additional column ('mbu_species_richness') containing the calculation of that 
                             : index per grid or geometry
    """
    
    #Species Richness calculation
    df = species_richness(roi, gdf, grid_gdf, source)
    
    #Normalization factor
    Norm_factor1 = df['species_richness']/df['species_richness'].max()
    
    #Convert area from degrees to square kilometers
    df['area_sqkm'] = (df.to_crs(crs=crs_transformation_kms).area)*10**(-6)

    #Calculate the MBUS from this MF
    df['mbu_species_richness'] = Norm_factor1*df['area_sqkm']
    
    return df
    
#---------------------------------------------------------------------------------------------------------------------
    
def mbu_endemism(roi, gdf, grid_gdf, crs_transformation_kms):
    """
    input(s):
    roi <shapely polygon in CRS WGS84:EPSG 4326>: region of interest or the total project area
    gdf <geopandas dataframe>: contains at least the name of the species, the distribution polygons of each of them 
                             :and their abundance
    grid_gdf <geopandas dataframe>: consists of polygons of grids typically generated by the gridding function
                                  : containts at least a geometry column and a unique grid_id
    crs_transformation_kms: coordinate reference system transformation applied to the roi in meters 
    
    output(s):
    gdf <geopandas dataframe>: with an additional column ('mbu_endemism') containing the calculation of that index per grid
                             : or geometry
    """
    
    #Endemic factor calculation
    df = endemism(roi, gdf, grid_gdf)
    
    #Normalization factor
    Norm_factor1 = df['endemism']/df['endemism'].max()
    
    #Convert area from degrees to square kilometers
    df['area_sqkm'] = (df.to_crs(crs=crs_transformation_kms).area)*10**(-6)

    #Calculate the MBUS from this MF
    df['mbu_endemism'] = Norm_factor1*df['area_sqkm']
    
    return df

#--------------------------------------------------------------------------------------------------------------------- 
   
def mbu_wege(roi, gdf, grid_gdf, crs_transformation_kms):
    """
    input(s):
    roi <shapely polygon in CRS WGS84:EPSG 4326>: region of interest or the total project area
    gdf <geopandas dataframe>: contains at least the name of the species, the distribution polygons of each of them 
                             :and their abundance
    grid_gdf <geopandas dataframe>: consists of polygons of grids typically generated by the gridding function
                                  : containts at least a geometry column and a unique grid_id
    crs_transformation_kms: coordinate reference system transformation applied to the roi in meters 
    
    output(s):
    gdf <geopandas dataframe>: with an additional column ('mbu_wege') containing the calculation of that index per grid
                             : or geometry
    """
    
    #Wege factor calculation
    df = wege(roi, gdf, grid_gdf)
    
    #Normalization factor
    Norm_factor1 = df['wege']/df['wege'].max()
    
    #Convert area from degrees to square kilometers
    df['area_sqkm'] = (df.to_crs(crs=crs_transformation_kms).area)*10**(-6)

    #Calculate the MBUS from this MF
    df['mbu_wege'] = Norm_factor1*df['area_sqkm']
    
    return df

#---------------------------------------------------------------------------------------------------------------------

def mbu_habitats_survey(roi, grid_gdf, path_EFG, crs_transformation_kms):
    """
    inputs:
    roi <shapely polygon in CRS WGS84:EPSG 4326>: region of interest or the total project area
    list_EFG <list>: consist in a list with path location of each EFG file 
    grid_gdf <geopandas dataframe>: consists of polygons of grids typically generated by the gridding function
                                  : containts atleast a geometry column and a unique grid_id
    crs_transformation_kms: coordinate reference system transformation applied to the roi in meters 
    
    output(s):
    gdf <geopandas dataframe>: with an additional column ('mbu_habitats_survey') containing the number
                             : of units for that grid or geometry
    """
    
    #Wege factor calculation
    df = habitats_survey(roi, grid_gdf, path_EFG)
    
    #Normalization factor
    Norm_factor1 = df['habitats_survey']/df['habitats_survey'].max()
    
    #Convert area from degrees to square kilometers
    df['area_sqkm'] = (df.to_crs(crs=crs_transformation_kms).area)*10**(-6)

    #Calculate the MBUS from this MF
    df['mbu_habitats_survey'] = Norm_factor1*df['area_sqkm']
    
    return df

#---------------------------------------------------------------------------------------------------------------------
#General MBU function
#---------------------------------------------------------------------------------------------------------------------

def give_mbu_score(modulating_factor_names, roi, grid_size_deg, grid_shape, gdf_col_name, path_EFG, crs_transformation_kms):
    """
    inputs:
    modulating_factor_names<list>: list of names of the modulating factors, e.g. ["species_richness", "habitat_survey"]
    modulating_factor_names:
            - biodiversity_score
            - species_richness
            - endemism
            - wege
            - habitats_survey
    roi <shapely polygon>: region of interest or the total project area
    grid_size_deg <int>: size of the grid in degrees. Minimum values are enforced
                     : if grid_size_deg = 0: it means there are no grids
    grid_shape <str>: either "square" or "hexagonal"
    gdf_col_name <string>: corresponds to the name of the abundance information column in the gdf
    path_EFG <list>: consists of a list of EFG polygons' paths to be reference internally
    crs_transformation_kms: coordinate reference system transformation applied to the roi in meters 

    output(s):
    geopandas frame
    """