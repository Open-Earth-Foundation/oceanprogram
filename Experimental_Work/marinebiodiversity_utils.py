# load basic libraries
import os
import glob
import numpy as np
import math
import pandas as pd

# to manage shapefiles
import shapely
import geopandas as gpd
from shapely.geometry import Polygon, Point, box

#------------------------------------------------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------------------------------------------------

#Obtainning all the geometries within our study place

def Clip_EFG(fnames, shape, crs):
    """
    You have to be sure that you're using the same crs in all geometries
    """
    
    """
    Inputs: 
    - fnames: list of ecosystems shapefiles
    - shape: shapefile of the study place
    - crs: number of the Coordinate Reference System
    """

    EcoTypo = []
    Geometry = []

    for x in range(len(fnames)):
        #Reading each EFG
        gdf = gpd.read_file(str(fnames[x]))
    
        #Choosing the crs
        gdf = gdf.set_crs(epsg=crs, allow_override=True)
    
        #We want only the polygons within our study place
        clip = gpd.clip(gdf, shape)
        clip = clip.reset_index()
    
        #Selecting the 'geometry' column
        geo = clip.geometry

        Geometry.append(geo)
        
    joined = gpd.GeoDataFrame(pd.concat(Geometry, ignore_index=True))
    
    return joined

#Overlapping Geometry Accounting

def count_overlapping_geometries(in_gdf):
    """
    Input: geoDataframe with geometries to count
    Output: new geoDataframe with accounting
    
    """
    #Get the name of the column containing the geometries
    geom_col = in_gdf.geometry.name
    
    # Setting up a single piece that will be split later
    input_parts = [in_gdf.unary_union.buffer(0)]
    
    # Finding all the "cutting" boundaries. Note: if the input GDF has 
    # MultiPolygons, it will treat each of the geometry's parts as individual
    # pieces.
    cutting_boundaries = []
    for i, row in in_gdf.iterrows():
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
                               crs=in_gdf.crs,
                               geometry=geom_col)
    
    # Find the new centroids.
    new_gdf['geom_centroid'] = new_gdf.centroid
    
    # Starting the count at zero
    new_gdf['count_intersections'] = 0
    
    # For each of the `new_gdf`'s rows, find how many overlapping features 
    # there are from the input GDF.
    for i,row in new_gdf.iterrows():
        new_gdf.loc[i,'count_intersections'] = in_gdf.intersects(row['geom_centroid']).astype(int).sum()
        pass
    
    # Dropping the column containing the centroids
    new_gdf = new_gdf.drop(columns=['geom_centroid'])[['id','count_intersections',geom_col]]
    
    return new_gdf

#Gridding

def create_grid(feature, shape, side_length, crs):
    """
    Create a grid consisting of either rectangles or hexagons with a specified side length that covers the extent of input    feature.
    """

    # Slightly displace the minimum and maximum values of the feature extent by creating a buffer
    # This decreases likelihood that a feature will fall directly on a cell boundary (in between two cells)
    # Buffer is projection dependent (due to units)
    #feature = feature.buffer(20) #This is increase the boundary by 20 degrees!
    # print("no buffer")

    # Get extent of buffered input feature
    min_x, min_y, max_x, max_y = feature.total_bounds


    # Create empty list to hold individual cells that will make up the grid
    cells_list = []

    # Create grid of squares if specified
    if shape in ["square", "rectangle", "box"]:

        # Adapted from https://james-brennan.github.io/posts/fast_gridding_geopandas/
        # Create and iterate through list of x values that will define column positions with specified side length
        for x in np.arange(min_x - side_length, max_x + side_length, side_length):

            # Create and iterate through list of y values that will define row positions with specified side length
            for y in np.arange(min_y - side_length, max_y + side_length, side_length):

                # Create a box with specified side length and append to list
                cells_list.append(box(x, y, x + side_length, y + side_length))


    # Otherwise, create grid of hexagons
    elif shape == "hexagon":

        # Set horizontal displacement that will define column positions with specified side length (based on normal hexagon)
        x_step = 1.5 * side_length

        # Set vertical displacement that will define row positions with specified side length (based on normal hexagon)
        # This is the distance between the centers of two hexagons stacked on top of each other (vertically)
        y_step = math.sqrt(3) * side_length

        # Get apothem (distance between center and midpoint of a side, based on normal hexagon)
        apothem = (math.sqrt(3) * side_length / 2)

        # Set column number
        column_number = 0

        # Create and iterate through list of x values that will define column positions with vertical displacement
        for x in np.arange(min_x, max_x + x_step, x_step):

            # Create and iterate through list of y values that will define column positions with horizontal displacement
            for y in np.arange(min_y, max_y + y_step, y_step):

                # Create hexagon with specified side length
                hexagon = [[x + math.cos(math.radians(angle)) * side_length, y + math.sin(math.radians(angle)) * side_length] for angle in range(0, 360, 60)]

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
    grid = gpd.GeoDataFrame(cells_list, columns = ['geometry'], crs = crs)

    # Create a column that assigns each grid a number
    grid["Grid_ID"] = np.arange(len(grid))

    # Return grid
    return grid