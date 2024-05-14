# Class for routing betweeen responders and patients through AED locations

import openrouteservice
from openrouteservice import client
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import geometry
import time
import geopy.distance
import plotly.express as px
import plotly.graph_objects as go


class route:
    def __init__(self):
        self.Client_ors = openrouteservice.Client(key='5b3ce3597851110001cf624802e069d6633748a5ae4e9842334f1dc2')

    # function to find the AEDs and Responders in a 10 minute walking distance from the patient
    # Nearly same as closest Responders
    def closest_location(self, Patient, Location, profile = "foot-walking", threshold = 600):
        # patient must be a tuple
        # AEDS must be a dataframe with columns (This is gathered in another file) named latitude and longitude
        # profile by default is walking by foot

        # set the parameters for conducting an isochrone search
        isochrones_parameters = {
        'locations': [Patient],
        'profile': profile,
        'range_type': 'time',
        'range': [threshold] # 10 minutes away (600 seconds)
        }

        # personal API key for openrouteservices
        ors_client_custom = self.Client_ors

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
        df['coords'] = list(zip(df['lon'],df['lat']))

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
    def duration_to_Patient(self, Patient, Responders, profile = 'foot-walking'):
        # find the closest Responders
        x = self.closest_location(Patient, Responders, profile)
        # append the location of the patient to use it later
        x.append(Patient)
        # calculate a matrix that gives the duration of every closes responder 
        # to the patient directly
        client = self.Client_ors
        matrix = client.distance_matrix(
                    locations=x,
                    profile=profile,
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
    def directions(self, coordinates, profile = 'foot-walking'):
        client = self.Client_ors
        time.sleep(1.0)
        route = client.directions(coordinates=coordinates,
                                   profile=profile,
                                   format='geojson',
                                   validate=False)
        route_dict = {}
        route_dict['duration'] = route.get('features')[0]['properties']['summary']['duration']
        route_dict['route'] = route
        route_dict['coordinates'] = route.get('features')[0]['geometry']['coordinates']
        return route_dict
    
    # Function to get all possible routes through the AEDs that are close to the patient
    # Returns a data frame with the coordinates of the Responder, duration through the specific AED,
    # duration for the direct route, and the coordinates of the used AED
    def possible_routing(self, Patient, Responders, AEDs, threshold = 700):
        if len(Responders) < 3:    # If there are less than 3 responders in total (unrealistic case)
            Responders_loc = self.closest_location(Patient, Responders, threshold=10000)    # Set a high threshold
        else:
            # Check if responders exist in the direct circumference (first in 8 minute difference of the patient)
            t_loc = 480
            Responders_loc = self.closest_location(Patient, Responders, threshold=t_loc)
            # if there are less than 3 responders nearby the distance of the isochrone is increased by 2 minutes
            while len(Responders_loc) < 3:
                t_loc += 120
                Responders_loc = self.closest_location(Patient, Responders, threshold=t_loc)
        
        # create a data frame based on the coordinates of the close responders
        Responder_df = pd.DataFrame(Responders_loc, columns =['longitude', 'latitude'])
        # also get the coordinates as a tuple
        Responder_df['Responder_loc'] = list(zip(Responder_df['longitude'],Responder_df['latitude']))
        # get the coordinates of the patient
        Responder_df['Patient_lon'] = Patient[0]
        Responder_df['Patient_lat'] = Patient[1]
        Responder_df['Patient_loc'] = list(zip(Responder_df['Patient_lon'],Responder_df['Patient_lat']))
        # get the distance between responders and patients
        Responder_df['dist_patient'] = Responder_df.apply(lambda row: geopy.distance.distance(row['Responder_loc'], row['Patient_loc']).meters, axis=1)
        # if the distance is lower than the threshold (default is 700 meters), the foot walking distance is calculated and otherwise the value
        # is set to a high value.
        # This is done to minimize the amount the API is used as this is restricted in the free version
        Responder_df['duration_direct']=[self.directions([i, Patient])['duration'] if d<threshold else 5000 for i, d in zip(Responder_df['Responder_loc'], Responder_df['dist_patient'])]

        # Duration through AED
        if len(AEDs) < 3:    # If there are less than 3 AEDs in total (unrealistic case)
            AED_loc = self.closest_location(Patient, AEDs, threshold=10000)    # Set a high threshold
        else:
            # Check if AEDs are close by (first in 8 minute difference of the patient)
            t_AED = 480
            AED_loc = self.closest_location(Patient, AEDs, threshold=t_AED)
            while len(AED_loc) < 3:
                # if there are less than 3 AEDs nearby the distance of the isochrone is increased by 2 minutes
                t_AED += 120
                AED_loc = self.closest_location(Patient, AEDs, threshold=t_AED)

        # Create data frames with the coordinates
        AED_df = pd.DataFrame(zip(AED_loc), columns =['AED_coordinates'])
        # combine the both data frames
        df_merged = pd.merge(Responder_df.assign(key=1), AED_df.assign(key=1),
                        on='key').drop('key', axis=1)
        # Similar as before calculate the distance between AED and the responders.
        df_merged['dist_AED'] = df_merged.apply(lambda row: geopy.distance.distance(row['Responder_loc'], row['AED_coordinates']).meters, axis=1)
        # If the Responders are closer to the AED than the threshold (by default 700 meters as the bird flies, as this takes around 10 minutes to walk),
        # the duration by foot form the responder through the AED to the patient is calculated and stored in the Data Frame.
        df_merged['duration_through_AED']=[self.directions([df_merged['Responder_loc'][i], df_merged['AED_coordinates'][i],df_merged['Patient_loc'][i]])['duration'] if df_merged['dist_AED'][i] < threshold else 5000 for i in range(len(df_merged['dist_AED']))]
        return df_merged
        
    # Transform a list of lists of coordinates to a data frame with two columns
    def get_coordinates(self, coordinate_list):
        # first sublist element = longitude; second = latitude
        lon = list(list(zip(*coordinate_list))[0])
        lat = list(list(zip(*coordinate_list))[1])
        dict = {'lon':lon,'lat':lat}
        df_latlong = pd.DataFrame(dict)
        return df_latlong
    
    # Function to find the responder that is send directly and through the AED.
    # The results are plotted using plotly
    def send_responders(self, Patient, Responders, AEDs):
        df_duration = self.possible_routing(Patient, Responders, AEDs)
        # latitude of the closest responder
        lat_direct = df_duration.iloc[df_duration.idxmin()['duration_direct']]['latitude']
        # latitude of the Responder with the fastest time through an AED
        lat_AED = df_duration.iloc[df_duration.idxmin()['duration_through_AED']]['latitude']
        # longitude of the closest responder
        lon_direct = df_duration.iloc[df_duration.idxmin()['duration_direct']]['longitude']
        # longitude of the Responder with the fastest time through an AED
        lon_AED = df_duration.iloc[df_duration.idxmin()['duration_through_AED']]['longitude']
        # coordinates of the AED with the fastest route
        subset = df_duration[(df_duration['duration_direct']>df_duration.min()['duration_direct']) & (df_duration['duration_direct']>df_duration.min()['duration_direct'])]
        # Check if the fastest response time with AED is only slightly slower/faster than the direct routing and how different it is
        # for the second fastest
        dif_AED_direct = df_duration[df_duration['duration_direct']==df_duration.min()['duration_direct']].min()['duration_through_AED'] - df_duration.min()['duration_direct']
        # difference between fastest and second fastest direct way
        dif_2nd_1st_direct = df_duration.iloc[df_duration.drop_duplicates(subset=['Responder_loc']).nsmallest(2,'duration_direct').index[1]]['duration_direct'] - df_duration.min()['duration_direct']

        # First check if any responder exist that is not furhter away than 600 seconds
        # DISCUSS
        if df_duration[df_duration['duration_direct']<1200].any()['duration_direct'] and df_duration[df_duration['duration_through_AED']<1200].any()['duration_through_AED']:
            # Now check if the fastest through AED is the same as the fastest direct 
            if lat_direct==lat_AED and lon_direct==lon_AED:
                # Check if the difference between direct route and route through AED is miner (less than 30 seconds)
                # and if the difference between second fastest direct and the fastest direct is not to big (60 seconds)
                # This is done because time is of essence and otherwise the fast responder could be left out
                if ((dif_AED_direct < 30) and(dif_2nd_1st_direct < 60)):
                    # If both is true:
                    # - Second fastest direct time will be send directly
                    # - Fastest direct and AED responder will be send through the AED
                    coord_direct = (df_duration.iloc[df_duration.drop_duplicates(subset=['Responder_loc']).nsmallest(2,'duration_direct').index[1]]['longitude'], df_duration.iloc[df_duration.drop_duplicates(subset=['Responder_loc']).nsmallest(2,'duration_direct').index[1]]['latitude'])
                    coord_AED =  (lon_direct, lat_direct)
                    AED_coordinates = df_duration.iloc[df_duration.idxmin()['duration_direct']]['AED_coordinates']
                # If this is not true:
                # - Fastes direct responder will be send directly
                # - Second fastest through AED responder will be send through the AED
                else:
                    coord_direct = (lon_direct, lat_direct)
                    lat_AED_2nd = subset.iloc[subset.idxmin()['duration_through_AED']]['latitude']
                    lon_AED_2nd = subset.iloc[subset.idxmin()['duration_through_AED']]['longitude']
                    coord_AED = (lon_AED_2nd, lat_AED_2nd)
                    AED_coordinates = subset.iloc[subset.idxmin()['duration_through_AED']]['AED_coordinates']
            else:
                # If the fastest direct responder and thorugh AED responder are different:
                # - Take the fastest responders for both
                coord_direct = (lon_direct, lat_direct)
                coord_AED = (lon_AED, lat_AED)
                AED_coordinates = df_duration.iloc[df_duration.idxmin()['duration_through_AED']]['AED_coordinates']
        
        return {'coord_direct': coord_direct, 'coord_AED': coord_AED, 'AED_coordinates':AED_coordinates}
    
        '''
        # Get both routes
        direct_route = self.directions([coord_direct, Patient])
        AED_route = self.directions([coord_AED, AED_coordinates, Patient])

        # Get a dataframe of the description of the route for plotting
        # To transform the route into usable data frame for plotting with the get_coordinates function
        df_latlong_direct = self.get_coordinates(direct_route['coordinates'])
        df_latlong_AED = self.get_coordinates(AED_route['coordinates'])

        # plot the AEDs
        fig = px.scatter_mapbox(AEDs, lat="latitude", lon="longitude", zoom=3, height=300, color_discrete_sequence=["green"])
        fig.update_traces(marker=dict(size=7)) 

        # plot the direct way
        fig.add_trace(px.line_mapbox(df_latlong_direct, lat="lat", lon="lon").data[0])
        # Add the route through the AED
        fig.add_trace(px.line_mapbox(df_latlong_AED, lat='lat', lon='lon').data[0]) 
            
        # Add marker for the first responders initial location
        beginning_direct = go.Scattermapbox(
            lat=[df_latlong_direct['lat'].iloc[0]],
            lon=[df_latlong_direct['lon'].iloc[0]],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=10,
                color='darkblue'
            ),
            text='First responder direct',  # Text to display when hovering over the marker
            hoverinfo='text'
        )
        
        # Add markers for the first responder that takes the route through the AED
        beginning_AED = go.Scattermapbox(
            lat=[df_latlong_AED['lat'].iloc[0]],
            lon=[df_latlong_AED['lon'].iloc[0]],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=10,
                color='orange'
            ),
            text='Start responder through AED',
            hoverinfo='text'
        )

        # Add marker for the Patient
        Patient = go.Scattermapbox(
            lat=[df_latlong_AED['lat'].iloc[-1]],
            lon=[df_latlong_AED['lon'].iloc[-1]],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=15,
                color='red'
            ),
            text='Patient',
            hoverinfo='text' 
        )

        # Add a marker for the AED
        AED_marker = go.Scattermapbox(
            lat=[AED_coordinates[1]],
            lon=[AED_coordinates[0]],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=15,
                color='green'
            ),
            text='AED device',
            hoverinfo='text'
        )
        
        # Add the markers to the figure
        fig.add_trace(beginning_direct)
        fig.add_trace(beginning_AED)
        fig.add_trace(Patient)
        fig.add_trace(AED_marker)
        
        # Color the direct responder in darkblue and the one through the AED in orange
        fig.update_traces(line=dict(color='orange', width = 4), selector=2)
        fig.update_traces(line=dict(color='darkblue', width = 4), selector=1)
        fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=14, mapbox_center_lat=df_latlong_direct['lat'].iloc[0],
                          margin={"r": 0, "t": 0, "l": 0, "b": 0})
        return fig
        '''