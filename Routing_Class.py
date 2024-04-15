# Class for routing betweeen responders and patients through AED locations

import openrouteservice
from openrouteservice import client
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import geometry


class route:
    def __init__(self):
        self.Client = openrouteservice.Client(key='5b3ce3597851110001cf624802e069d6633748a5ae4e9842334f1dc2')

    # function to find the resopnders in a 10 minute walking distance from the patient
    def closest_Responders(self, Patient, Responders):
        # patient must be a tuple
        # responders must be a dataframe with latitude at first column and longitude at second column

        profile = 'foot-walking' # only taking walking distance

        # set the parameters for conducting an isochrone search
        isochrones_parameters = {
        'locations': [Patient],
        'profile': profile,
        'range_type': 'time',
        'range': [600] # 10 minutes away (600 seconds)
        }

        # personal API key for openrouteservices
        ors_client_custom = client.Client(key='5b3ce3597851110001cf624802e069d6633748a5ae4e9842334f1dc2', base_url='https://api.openrouteservice.org')

        # Get the area that can be reached in 10 minutes walking.
        # This creates an isochrone wich is the whole area surounding the patient that is reachable in 10 minutes by foot.
        isochrone = ors_client_custom.isochrones(**isochrones_parameters)

        # Transform the isochrone into a polygon and add it to a Geo Data Frame
        poly = geometry.Polygon(isochrone['features'][0]['geometry']['coordinates'][0])
        d = {'geometry': [poly]}
        gdf = gpd.GeoDataFrame(d)

        # Get the coordinates (lat, lon) for all Responders.
        df = pd.DataFrame({'lat':Responders.iloc[:,0], 'lon':Responders.iloc[:,1]}) # lat as 1st column longitude as 2nd
        df['coords'] = list(zip(df['lat'],df['lon']))

        # Transform the coordinates to geodataframe points
        df['coords'] = df['coords'].apply(geometry.Point)
        points = gpd.GeoDataFrame(df, geometry='coords', crs=gdf.crs)

        # Define the points of all Responders
        points = list(points['coords'])

        # Check which points are within the polygon.
        # Empty list where all points are stored later.
        points_inside_polygon = []
        # For loop to iterate through all possible responders.
        for point in points:
            if point.within(gdf['geometry'][0]):
                points_inside_polygon.append(point)

        # Create tuple that includes all points in 10 minute walking distance from the patient
        coordinate_tuples = [(point.x, point.y) for point in points_inside_polygon]

        # Returns a list of tuples of the coordinates of the responders.
        return coordinate_tuples

    # function to find the AEDs in a 10 minute walking distance from the patient
    # Nearly same as closest Responders
    def closest_AED(self, Patient, Responders):
        # patient must be a tuple
        # AEDS must be a dataframe with columns (This is gathered in another file) named latitude and longitude
        profile = 'foot-walking' # only taking walking distance

        # set the parameters for conducting an isochrone search
        isochrones_parameters = {
        'locations': [Patient],
        'profile': profile,
        'range_type': 'time',
        'range': [600] # 10 minutes away (600 seconds)
        }

        # personal API key for openrouteservices
        ors_client_custom = client.Client(key='5b3ce3597851110001cf624802e069d6633748a5ae4e9842334f1dc2', base_url='https://api.openrouteservice.org')

        # Get the area that can be reached in 10 minutes walking.
        # This creates an isochrone wich is the whole area surounding the patient that is reachable in 10 minutes by foot.
        # 10 minutes are taken as most research assumes that 8 minutes is the longest time
        # first responders should take. As we assume the responders to walk faster than normal,
        # a 10 minute radius should be fine.
        isochrone = ors_client_custom.isochrones(**isochrones_parameters)

        # Transform the isochrone into a polygon and add it to a Geo Data Frame
        poly = geometry.Polygon(isochrone['features'][0]['geometry']['coordinates'][0])
        d = {'geometry': [poly]}
        gdf = gpd.GeoDataFrame(d)

        # Get the coordinates (lat, lon) for all Responders.
        df = pd.DataFrame({'lat':Responders.loc[:,"latitude"], 'lon':Responders.loc[:,"longitude"]}) # lat as 1st column longitude as 2nd
        df['coords'] = list(zip(df['lat'],df['lon']))

        # Transform the coordinates to a geodataframe points
        df['coords'] = df['coords'].apply(geometry.Point)
        points = gpd.GeoDataFrame(df, geometry='coords', crs=gdf.crs)

        # Define the points of all Responders
        points = list(points['coords'])

        # Check which points are within the polygon.
        # Empty list where all points are stored later.
        points_inside_polygon = []
        # For loop to iterate through all possible responders.
        for point in points:
            if point.within(gdf['geometry'][0]):
                points_inside_polygon.append(point)

        # Create tuple that includes all points in 10 minute walking distance from the patient
        coordinate_tuples = [(point.x, point.y) for point in points_inside_polygon]

        # Returns a list of tuples of the coordinates of the responders.
        return coordinate_tuples
    
