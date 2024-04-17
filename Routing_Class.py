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

    # function to find the AEDs and Responders in a 10 minute walking distance from the patient
    # Nearly same as closest Responders
    def closest_location(self, Patient, Location):
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
        df = pd.DataFrame({'lat':Location.loc[:,"latitude"], 'lon':Location.loc[:,"longitude"]}) # lat as 1st column longitude as 2nd
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
    
    # function to calculate the durations if Responders go directly to the Patient
    def duration_to_Patient(self, Patient, Responders):
        # find the closest Responders
        x = self.closest_location(Patient, Responders)
        # append the location of the patient to use it later
        x.append(Patient)
        # calculate a matrix that gives the duration of every closes responder 
        # to the patient directly
        client = openrouteservice.Client(key='5b3ce3597851110001cf624802e069d6633748a5ae4e9842334f1dc2')
        matrix = client.distance_matrix(
                    locations=x,
                    profile="foot-walking",
                    sources = list(range(len(x)-1)),
                    destinations = [len(x)-1],
                    metrics=['duration']
                )
        source = []
        for i in range(len(matr["sources"])):
            source.append([matr['sources'][i]['location'][0], matr['sources'][i]['location'][1],float(matr['durations'][i][0])])

        # build a data frame with the duration and the coordinates of the Responder
        df_duration = pd.DataFrame(source, columns = ['longitude', 'latitude', 'duration_direct']) 
        return df_duration

    # Function that gets the duration, route and coordinates of a route
    # The route can be direct or go through other points first
    def foot_walking(self, coordinates):
        client = self.Client
        route = client.directions(coordinates=coordinates,
                                   profile='foot-walking',
                                   format='geojson',
                                   validate=False)
        route_dict = {}
        route_dict['duration'] = route.get('features')[0]['properties']['summary']['duration']
        route_dict['route'] = route
        route_dict['coordinates'] = route.get('features')[0]['geometry']['coordinates']
        return route_dict
    

    