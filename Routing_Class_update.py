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
        ## Adam's Key
        self.Client_ors = openrouteservice.Client(key='5b3ce3597851110001cf6248a8348d6a74544602bd7cbc7936c635d1')
        ## Dillon's Key
        # self.Client_ors = openrouteservice.Client(key='5b3ce3597851110001cf6248748ec775959f4a5ea085161e2eef2cd1')
        self.AED_Iso = gpd.read_file('C:/Users/leonw/OneDrive - KU Leuven/Documents/GitHub/AED-Route-Optimization-MDA-Project/Data/temp.gpkg', layer='AED_data')

    # function to find the AEDs and Responders in a 10 minute walking distance from the patient
    # Nearly same as closest Responders

    def find_isochrone(self, Midpoint, profile = "foot-walking", threshold = 600):
        # Midpoint must be a tuple
        # AEDS must be a dataframe with columns (This is gathered in another file) named latitude and longitude
        # profile by default is walking by foot

        # set the parameters for conducting an isochrone search
        isochrones_parameters = {
        'locations': [Midpoint],
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

        return isochrone

    def closest_location(self, Patient, Location, threshold = 600):
        # patient must be a tuple
        # AEDS must be a dataframe with columns (This is gathered in another file) named latitude and longitude
        # profile by default is walking by foot

        # Transform the isochrone into a polygon and add it to a Geo Data Frame
        isochrone = self.find_isochrone(Patient, profile = "foot-walking", threshold = threshold)

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
    
    def closest_location_AED(self, Location, polygon, threshold = 600):
        # patient must be a tuple
        # AEDS must be a dataframe with columns (This is gathered in another file) named latitude and longitude
        # profile by default is walking by foot

        d = {'geometry': [polygon]}
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
        #time.sleep(1.0)
        route = client.directions(coordinates=coordinates,
                                   profile=profile,
                                   format='geojson',
                                   validate=False)
        route_dict = {}
        # check if all coordinates are the same
        # If so duration would be empty thus it is set to 0
        if coordinates.count(coordinates[0]) == len(coordinates):
            route_dict['duration'] = 0
            route_dict['route'] = route
            route_dict['coordinates'] = route.get('features')[0]['geometry']['coordinates']
        else:
            route_dict['duration'] = route.get('features')[0]['properties']['summary']['duration']
            route_dict['route'] = route
            route_dict['coordinates'] = route.get('features')[0]['geometry']['coordinates']
        return route_dict
    
    # Function to get all possible routes through the AEDs that are close to the patient
    # Returns a data frame with the coordinates of the Responder, duration through the specific AED,
    # duration for the direct route, and the coordinates of the used AED
    def possible_routing_direct(self, Patient, Responders, threshold = 600):
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
        # only keep the 5 closest responders. keep='all' so that more that all responders with the 5 lowest values are kept.
        Responder_df = Responder_df.nsmallest(5, 'dist_patient')#, keep='all'
        Responder_df = Responder_df.reset_index(drop=True)
        Responder_df['duration_direct']=[self.directions([i, Patient_cood], profile = 'foot-walking')['duration'] for i, 
                                          Patient_cood in zip(Responder_df['Responder_loc'], Responder_df['Patient_loc'])]

        return Responder_df
    
    def possible_routing_indirect(self, Patient, Responders, AEDs, threshold=600):
        # create a data frame based on the coordinates of the close responders
        Responder_df = pd.DataFrame(Responders, columns =['longitude', 'latitude'])
        # also get the coordinates as a tuple
        Responder_df['Responder_loc'] = list(zip(Responder_df['longitude'],Responder_df['latitude']))
        # Duration through AED
        if len(AEDs) < 3:  # If there are less than 3 AEDs in total (unrealistic case)
            AED_loc = self.closest_location(Patient, AEDs, threshold=10000)  # Set a high threshold
        else:
            # Check if AEDs are close by (first in 8 minute difference of the patient)
            t_AED = 480
            AED_loc = self.closest_location(Patient, AEDs, threshold=t_AED)
            while len(AED_loc) < 3:
                # if there are less than 3 AEDs nearby the distance of the isochrone is increased by 2 minutes
                t_AED += 120
                AED_loc = self.closest_location(Patient, AEDs, threshold=t_AED)

        AED_df = pd.DataFrame(AED_loc, columns =['longitude', 'latitude'])
        # also get the coordinates as a tuple
        AED_df['AED'] = list(zip(AED_df['longitude'],AED_df['latitude']))
        # get the coordinates of the patient
        AED_df['Patient_lon'] = Patient[0]
        AED_df['Patient_lat'] = Patient[1]
        AED_df['Patient_loc'] = list(zip(AED_df['Patient_lon'],AED_df['Patient_lat']))


        # Create data frames with the coordinates
        AED_ISO = self.AED_Iso
        AED_ISO['AED'] = list(zip(AED_ISO["lon"], AED_ISO["lat"]))
        AED_df = AED_ISO.merge(AED_df, on='AED')
        # Convert AED_df to a GeoDataFrame with the correct CRS
        print(AED_df.head())
        AED_df = gpd.GeoDataFrame(AED_df, geometry='geometry', crs="EPSG:4326")

        all_points = []  # Store points and midpoints as tuples

        # Iterate through each row in AED_df to use the 'Iso' polygons
        for _, row in AED_df.iterrows():
            polygon = row['geometry']
            midpoint = row['AED']
            
            points_in_polygon = self.closest_location_AED(Responder_df, polygon, threshold=threshold)

            # Add the points along with the midpoint of the polygon
            all_points.extend([(point, midpoint) for point in points_in_polygon])

        # Create a DataFrame with the points and their associated polygon midpoints
        result_df = pd.DataFrame(all_points, columns=['Responder_loc', 'AED_coordinates'])
        result_df = result_df.drop_duplicates()

        # get the coordinates of the patient
        result_df['Patient_lon'] = Patient[0]
        result_df['Patient_lat'] = Patient[1]
        result_df['Patient_loc'] = list(zip(result_df['Patient_lon'],result_df['Patient_lat']))
        

        result_df['duration_through_AED']=[self.directions([result_df['Responder_loc'][i], result_df['AED_coordinates'][i],result_df['Patient_loc'][i]])['duration'] for i in range(len(result_df['Responder_loc']))]
        return result_df
    
    # Change so the that the dataframe only includes the fastest instance for every Responder
            
    # Transform a list of lists of coordinates to a data frame with two columns
    def get_coordinates(self, coordinate_list):
        # first sublist element = longitude; second = latitude
        lon = list(list(zip(*coordinate_list))[0])
        lat = list(list(zip(*coordinate_list))[1])
        dict = {'lon':lon,'lat':lat}
        df_latlong = pd.DataFrame(dict)
        return df_latlong
    
    # Function to simulate survival probability.
    # Used in send_responders to decide which responder should be send directly.
    def survival_probability(self, x, z):
        return 0.9 **(x/60)* 0.97**(z/60)
    
    # Function to find the responder that is send directly and through the AED.
    # The results are plotted using plotly
    def send_responders(self, Patient, Responders, AEDs):
        df_duration_direct = self.possible_routing_direct(Patient, Responders)
        df_duration_indirect = self.possible_routing_indirect(Patient, Responders, AEDs)

        df_duration_direct = df_duration_direct.sort_values(by=['duration_direct'])
        df_duration_indirect = df_duration_indirect.sort_values(by=['duration_through_AED'])

        fastest_aed = df_duration_indirect.loc[0]
        fastest_gpr = df_duration_direct.loc[0]
        second_fastest_aed = df_duration_indirect.loc[1]
        second_fastest_gpr = df_duration_direct.loc[1]

        # Fastest responder going for the AED
        # Check if the 2nd fastest direct responder is faster than the fastes direct through AED
        if second_fastest_gpr['duration_direct'] < fastest_aed['duration_through_AED']:
            # if so, z equal to CPR by 2nd fastest responder
            CPR_time = fastest_aed['duration_through_AED'] - second_fastest_gpr['duration_direct']
            surv_A = self.survival_probability(second_fastest_gpr['duration_direct'], CPR_time)
        else:
            # - z equals zero as no one does any CPR
            surv_A = self.survival_probability(fastest_aed['duration_through_AED'], 0)
        # 2nd fastest responder arriving with AED
        # - time until AED arrives minus time CPR arrives is the time without CPR
        surv_B = self.survival_probability(fastest_gpr['duration_direct'], second_fastest_aed['duration_through_AED']-fastest_gpr['duration_direct'])    
            
        # Check if the fastest through AED is the same as the fastest direct 
        if fastest_aed['Responder_loc']==fastest_gpr['Responder_loc']:
            # Find best strategy which is the maximal survival chances
            best_strategy = max(surv_A, surv_B)
            # Send responders
            if best_strategy == surv_A:
                # - Second fastest direct time will be send directly  
                # - Fastest direct and AED responder will be send through the AED
                coord_direct = second_fastest_gpr['Responder_loc']
                coord_AED =  fastest_aed['Responder_loc']
                AED_coordinates = fastest_aed['AED_coordinates']
            # If this is not true:
            # - Fastes direct responder will be send directly
            # - Second fastest through AED responder will be send through the AED
            else:
                coord_direct = fastest_gpr['Responder_loc']
                coord_AED = second_fastest_aed['Responder_loc']
                AED_coordinates = second_fastest_aed['AED_coordinates']
        else:
            # If the fastest direct responder and thorugh AED responder are different:
            # - Take the fastest responders for both
            coord_direct = fastest_gpr['Responder_loc']
            coord_AED = fastest_aed['Responder_loc']
            AED_coordinates = fastest_aed['AED_coordinates']
        
        return {'coord_direct': coord_direct, 'coord_AED': coord_AED, 'AED_coordinates': AED_coordinates}
    

         
    # Function to find the responder that is send directly and through the AED.
    # The results are plotted using plotly
    def send_responders(self, Patient, Responders, AEDs, N_responders, decline_rate, AED_rate):
        df_duration_direct = self.possible_routing_direct(Patient, Responders)
        df_duration_indirect = self.possible_routing_indirect(Patient, Responders, AEDs)

        df_duration_direct = df_duration_direct.sort_values(by=['duration_direct'], ascending=True)
        df_duration_direct = df_duration_direct.nsmallest(round((N_responders)*(1-AED_rate)), 'duration_direct')
        # Only keep the fastest route through AED for every responder
        df_duration_indirect.sort_values(by=['duration_through_AED'], ascending=True).drop_duplicates('Responder_loc').sort_index()
        df_duration_indirect = df_duration_indirect.nsmallest(round((N_responders)*(AED_rate)), 'duration_through_AED')



        fastest_aed = df_duration_indirect.loc[0]
        fastest_gpr = df_duration_direct.loc[0]
        second_fastest_aed = df_duration_indirect.loc[1]
        second_fastest_gpr = df_duration_direct.loc[1]

        # Fastest responder going for the AED
        # Check if the 2nd fastest direct responder is faster than the fastes direct through AED
        if second_fastest_gpr['duration_direct'] < fastest_aed['duration_through_AED']:
            # if so, z equal to CPR by 2nd fastest responder
            CPR_time = fastest_aed['duration_through_AED'] - second_fastest_gpr['duration_direct']
            surv_A = self.survival_probability(second_fastest_gpr['duration_direct'], CPR_time)
        else:
            # - z equals zero as no one does any CPR
            surv_A = self.survival_probability(fastest_aed['duration_through_AED'], 0)
        # 2nd fastest responder arriving with AED
        # - time until AED arrives minus time CPR arrives is the time without CPR
        surv_B = self.survival_probability(fastest_gpr['duration_direct'], second_fastest_aed['duration_through_AED']-fastest_gpr['duration_direct'])    
            
        # Check if the fastest through AED is the same as the fastest direct 
        if fastest_aed['Responder_loc']==fastest_gpr['Responder_loc']:
            # Find best strategy which is the maximal survival chances
            best_strategy = max(surv_A, surv_B)
            # Send responders
            if best_strategy == surv_A:
                # - Second fastest direct time will be send directly  
                # - Fastest direct and AED responder will be send through the AED
                coord_direct = second_fastest_gpr['Responder_loc']
                coord_AED =  fastest_aed['Responder_loc']
                AED_coordinates = fastest_aed['AED_coordinates']
            # If this is not true:
            # - Fastes direct responder will be send directly
            # - Second fastest through AED responder will be send through the AED
            else:
                coord_direct = fastest_gpr['Responder_loc']
                coord_AED = second_fastest_aed['Responder_loc']
                AED_coordinates = second_fastest_aed['AED_coordinates']
        else:
            # If the fastest direct responder and thorugh AED responder are different:
            # - Take the fastest responders for both
            coord_direct = fastest_gpr['Responder_loc']
            coord_AED = fastest_aed['Responder_loc']
            AED_coordinates = fastest_aed['AED_coordinates']
        
        return {'coord_direct': coord_direct, 'coord_AED': coord_AED, 'AED_coordinates': AED_coordinates}
    
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