import openrouteservice
from openrouteservice import client
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import geometry
import time
import geopy.distance
import random


# Exception class
#------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------

class NoResponderAcceptedError(Exception):
    """Raised when no responder accepts the request."""
    pass

class NoAEDResponderAcceptedError(Exception):
    """Raised when no responder through AED accepts the request."""
    pass

#--------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------



class RoutingSimulationMatrixBatch:
    def __init__(self, ip, all_open=False):
        self.IP = ip
        self.CLIENT_ORS = openrouteservice.Client(base_url=f'http://{self.IP}:8080/ors')
        self.AED_FILE_PATH = 'Data/temp.gpkg'
        self.AED_LAYER = 'AED_data'
        
        aeds = gpd.read_file(self.AED_FILE_PATH, layer=self.AED_LAYER)
        new_open = aeds["Opens"].str.split(":", n=1, expand=True)
        new_open = new_open.apply(pd.to_numeric, errors='coerce') 
        new_open['TIME'] = new_open[0] + (new_open[1] / 60)
        aeds['Opens'] = new_open['TIME']

        new_close = aeds["Closes"].str.split(":", n=1, expand=True)
        new_close = new_close.apply(pd.to_numeric, errors='coerce') 
        new_close['TIME'] = new_close[0] + (new_close[1] / 60)
        aeds['Closes'] = new_close['TIME']
        if all_open:
            aeds["Opens"] = 0
            aeds["Closes"] = 24
        self.AED_ISO = aeds


    # function to find the aeds and responders in a 10 minute walking distance from the patient
    # Nearly same as closest responders
    def __closest_location(self, patient, location, profile = "foot-walking", threshold =600):
        '''
        Function that gets the isochrone around the patient and finds all locations in the isochrone.
        Used in funciton: def fastest_time
        Uses function:

        Parameters:
        patient (pd.Series): patients coordinates in a series with name "longitude", "latitude". !!!CASE SENSITIVE!!!
        location (pd.Dataframe): Dataframe with column "longitude", "latitude" for the locations. !!!CASE SENSITIVE!!!
        profile (string): Can be either foot-waling, car-driving, cycling-* [-regular, -electric, -road, -mountain], driving-hgv, or wheelchair.
        threshold (integer): Time in seconds it takes to travel to the furthest away point still included in the area.
        
        Returns:
        pd.Dataframe: Returns dataframe with the coordinates of the locations in the threshold distance.
                      Has one column only with coordinates as tuple (longitude, latitude).
        '''

        # set the parameters for conducting an isochrone search
        isochrones_parameters = {
            'locations': [(patient['longitude'], patient['latitude'])],
            'profile': profile,
            'range_type': 'time',
            'range': [threshold]
        }
        
        # personal API key for openrouteservices
        ors_client_custom = self.CLIENT_ORS

        # Get the area that can be reached in 10 minutes walking.
        # This creates an isochrone wich is the whole area surounding the patient that is reachable in 10 minutes by foot.
        # 10 minutes are taken as most research assumes that 8 minutes is the longest time
        # first responders should take. As we assume the responders to walk faster than normal,
        # a 10 minute radius should be fine.
        isochrone = ors_client_custom.isochrones(**isochrones_parameters)

        # Transform the isochrone into a polygon and add it to a Geo Data Frame
        gdf = gpd.GeoDataFrame({'geometry': [geometry.Polygon(isochrone['features'][0]['geometry']['coordinates'][0])]})

        # Transform the coordinates to a geodataframe points
        # Define the points of all locations
        points = list([geometry.Point(location.loc[i,"longitude"], location.loc[i,"latitude"]) for i in range(len(location))])

        # Check which points are within the polygon.
        # Create tuple that includes all points in 10 minute walking distance from the patient
        coordinate_tuples = [[(point.x, point.y)] for point in points if point.within(gdf.loc[0,'geometry'])]
        df = pd.DataFrame(coordinate_tuples, columns = ['coordinates'])
        # Returns a list of tuples of the coordinates of the responders. (long, lat)
        
        return df
     
    
    def __closest_location_aed(self, location, polygon):
        '''
        Function that finds all locations inside an area.
        Used in function: def __possible_routing_indirect
        Uses function:

        Parameters:
        location (pd.Dataframe): Dataframe with column "longitude", "latitude" for the locations. !!!CASE SENSITIVE!!!
        polygon (shapely.geometry.polygon.Polygon): Polygon of the area in question.
        
        Returns:
        pd.Dataframe: Returns a list of tuples of the coordinates of the loations inside the area (longitude, latitude).
        '''

        area = {'geometry': [polygon]}
        gdf = gpd.GeoDataFrame(area) 
        # Get the coordinates (lat, lon) for all responders.
        df = pd.DataFrame({'coords':location.loc[:,"responder_loc"]}) # lat as 1st column longitude as 2nd

        # Transform the coordinates to a geodataframe points
        df['coords'] = df['coords'].apply(geometry.Point)
        # points = gpd.GeoDataFrame(df, geometry='coords', crs=gdf.crs)

        # # Define the points of all responders
        # points = list(points['coords'])

        # # Check which points are within the polygon.
        # # Empty list where all points are stored later.
        points = gpd.GeoDataFrame(df, geometry='coords', crs=gdf.crs)
        points_inside_polygon = [point for point in points['coords'] if point.within(gdf['geometry'][0])]

        # # Create tuple that includes all points in 10 minute walking distance from the patient
        return [(point.x, point.y) for point in points_inside_polygon]
    
        # points = gpd.GeoDataFrame(df, geometry='coords', crs=gdf.crs)

        # # Define the points of all responders
        # points = list(points['coords'])

        # # Check which points are within the polygon.
        # # Empty list where all points are stored later.
        # points_inside_polygon = []
        # # For loop to iterate through all possible responders.
        # for point in points:
        #     if point.within(gdf['geometry'][0]):
        #         points_inside_polygon.append(point)

        # # Create tuple that includes all points in 10 minute walking distance from the patient
        # coordinate_tuples = [(point.x, point.y) for point in points_inside_polygon]

        # Returns a list of tuples of the coordinates of the responders.

    def __directions(self, coordinates, profile = 'foot-walking'):
        """
        Function that gets the duration, route and coordinates of a route. The route can be direct or go through other points first
        Used in function: def __fastest_vector, def __direct_vs_vector, def __possible_routing_direct, def __possible_routing_indirect, def __fastest_comparisson
        Uses function:
        
        Parameters:
        coordinates (list): List containing [[lon,lat],[lon,lat],...] in the order of routing.
        prifile (string): can be either foot-waling, car-driving, cycling-* [-regular, -electric, -road, -mountain], driving-hgv, or wheelchair .
        sleep (float): seconds to wait.
        
        Returns:
        ditionary: Returns dictionary with duration, route and coordinates of the route.
        """
        custom_client = self.CLIENT_ORS
        #time.sleep(sleep)
        route = custom_client.directions(
            coordinates=coordinates,
            profile=profile,
            format='geojson',
            validate=False
        )
        # empty dictionary to be filled later
        route_dict = {}
        # check if all coordinates are the same
        # If so duration would be empty thus it is set to 0
        if coordinates.count(coordinates[0]) == len(coordinates):
            route_dict['duration'] = 0
        else:
            route_dict['duration'] = route.get('features')[0]['properties']['summary']['duration']
        route_dict['route'] = route
        route_dict['coordinates'] = route.get('features')[0]['geometry']['coordinates']

        return route_dict

    def __matrix_duration(self, coordinates, profile = 'foot-walking'):
        """
        Function that gets the duration, route and coordinates of a route. The route can be direct or go through other points first
        Used in function: def __fastest_vector, def __direct_vs_vector, def __possible_routing_direct, def __possible_routing_indirect, def __fastest_comparisson
        Uses function:
        
        Parameters:
        coordinates (list): List containing [[lon,lat],[lon,lat],...] in the order of routing.
        prifile (string): can be either foot-waling, car-driving, cycling-* [-regular, -electric, -road, -mountain], driving-hgv, or wheelchair .
        sleep (float): seconds to wait.
        
        Returns:
        ditionary: Returns dictionary with duration, route and coordinates of the route.
        """
        custom_client = self.CLIENT_ORS
        #time.sleep(sleep)
        matrix = custom_client.distance_matrix(
            locations=coordinates,
            destinations = [0],
            profile = profile,
            metrics=['distance', 'duration'],
            validate=False,
        )
        
        return matrix

    # Function to get all possible routes through the aeds that are close to the patient
    # Returns a data frame with the coordinates of the Responder, duration through the specific AED,
    # duration for the direct route, and the coordinates of the used AED

    def fastest_time(self, patient, responders, vectors, 
                     decline_rate, max_number_responders, opening_hours, filter_values,  
                     dist_responder = 600, dist_AED = 600, dist_Vector = 600):
        """
        Main function calculating the time until the first AED arrives.
        Used in function:
        Uses function: def __closest_location, def __fastest_vector, def __direct_vs_vector, def __fastest_comparisson
        
        Parameters:
        patient (pd.Series): patients coordinates in a series with name "longitude", "latitude". !!!CASE SENSITIVE!!!
        responders (pd.Dataframe): Dataframe of responders location with column "longitude", "latitude" for the locations. !!!CASE SENSITIVE!!!
        vectors (pd.Dataframe): Dataframe of vector (ambulance,...) location with column "longitude", "latitude" for the locations. !!!CASE SENSITIVE!!!
        decline_rate (float): !!!Must be between 0 and 1!!! Represents the percentage of responders declining the call to action.
        max_number_responders(int): Number of responders contacted
        opening_hours(float): time of the incident
        filter_values (list): List with all strings that the AED dataframe column "available" will be subseted by. If None every AED will be considered !!!CASE SENSITIVE!!!
        dist_responder = 600 (float): Time in seconds it takes to travel by foot to the furthest away point still included in the area (Used to find responders in that area).
        dist_AED = 600 (float): Time in seconds it takes to travel by foot to the furthest away point still included in the area (Used to find aeds in that area).
        dist_Vector = 600 (float): Time in seconds it takes to travel by car to the furthest away point still included in the area (Used to find vectors in that area).
        
        Returns:
        pd.DataFrame: Dataframe with 
                {'patient_loc':patient location,
            	'responder_loc':responders location, 
                'duration_Responder':Duration it takes the direct responder to get to the patient, 
                'aed_loc':aeds location,
                'duration_AED':Duration it takes the indirect responder to get to the patient,
                'vector_loc':Vector location,
                'duration_Vector':Duration it takes the vector to get to the patient}
        """

        # Time that the isochrones covers in seconds
        responder_loc = self.__closest_location(patient, responders,threshold=dist_responder)
        # Time that the isochrones covers in seconds
        aed_loc = self.__closest_location(patient, self.AED_ISO, threshold=dist_AED)
        # Time that the isochrones covers in seconds
        vector_loc = self.__closest_location(patient, vectors, profile = 'driving-car', threshold = dist_Vector)

        # Set empty Dataframe to be fillled later
        # Needed?
        df_duration = pd.DataFrame(columns = ['patient_loc', 'responder_loc', 'duration_Responder', 
                                              'aed_loc','duration_AED','vector_loc', 'duration_Vector'])
        
        # Check if any vector is reached in 10 minutes. 
        # If this is not the case the isochrone radius is increased for 2 minutes until at least one vector is found. 
        dist_Vector = dist_Vector
        while (len(vector_loc) == 0):
            #print('No vector is close by. Increase of thresholds')
            dist_Vector += 120
            vector_loc = self.__closest_location(patient, vectors, profile = 'driving-car', threshold = dist_Vector)
        
        # If there are no responders close by the isochrone radius for them and the AED is increased to the same time as the one of the vector before.
        # This is done as the responders could still be faster than the vectors and thus incerease the survival rate.
        if (len(responder_loc)==0):
            responder_loc = self.__closest_location(patient, responders, threshold=dist_Vector)
            aed_loc = self.__closest_location(patient, self.AED_ISO, threshold=dist_Vector)
        else:
            pass
        
        decline_rate = decline_rate
        opening_hours = opening_hours
        max_number_responders = max_number_responders
        # If the Responder is still 0 only the vectors time will be calculated
        if ((len(responder_loc) == 0) and (len(vector_loc) > 0)):
            #print('No responder is in a 10 minute radius')
            df_duration = self.__fastest_vector(patient, vector_loc)
        # Otherwise the two have to be compared.
        elif ((len(responder_loc) > 0) and (len(vector_loc) > 0)):
            if ((len(aed_loc) == 0) or (len(responder_loc)<2)):
                #print('Comparing direct responder vs. vectors')
                try:
                    df_duration = self.__direct_vs_vector(patient, vector_loc, responder_loc, decline_rate, max_number_responders)
                except NoResponderAcceptedError as e:
                    #print(f"Warning: {e}. Falling back to fastest vector.")
                    df_duration = self.__fastest_vector(patient, vector_loc)
            else:
                #print('Comparing responders vs. vectors')
                try:
                    df_duration = self.__fastest_comparisson(patient, vector_loc, responder_loc, aed_loc, max_number_responders, decline_rate, opening_hours, filter_values)
                except NoResponderAcceptedError as e:
                    #print(f"Warning: {e}. Falling back to fastest vector.")
                    df_duration = self.__fastest_vector(patient, vector_loc)
                except NoAEDResponderAcceptedError as e2:
                    #print(f"Warning: {e2}. Falling back to direct vs. vector.")
                    try:
                        df_duration = self.__direct_vs_vector(patient, vector_loc, responder_loc, decline_rate, max_number_responders)
                    except NoResponderAcceptedError as e:
                        #print(f"Warning: {e}. Falling back to fastest vector.")
                        df_duration = self.__fastest_vector(patient, vector_loc)

        # Returns a data frame.          
        return df_duration

    # Function when only the duration for the vector is needed.
    def __fastest_vector(self, patient, vector_loc):
        """
        Function calculating the time the Vector is the only thing near.
        Used in function: def fastest_time
        Uses function: def __directions
        
        Parameters:
        patient (pd.Series): patients coordinates in a series with name "longitude", "latitude". !!!CASE SENSITIVE!!!
        vectors (pd.Dataframe): Dataframe with the coordinates named "coordinates" !!!CASE SENSITIVE!!!. Normally, that is the output of def __closest_location.
        decline_rate (float): !!!Must be between 0 and 1!!! Represents the percentage of responders declining the call to action.
                
        Returns:
        pd.DataFrame: Dataframe with 
                {'patient_loc':patient location,
            	'responder_loc':'No responder', 
                'duration_Responder':'No responder', 
                'aed_loc':'No AED',
                'duration_AED':'No AED
                'vector_loc':Vector location,
                'duration_Vector':Duration it takes the vector to get to the patient}
        """
        if isinstance(patient, pd.Series):
            patient = patient.to_frame().T
        elif isinstance(patient, pd.DataFrame) and patient.shape[0] != 1:
            # If patient DataFrame has multiple rows, take the first one (or adjust as needed)
            patient = patient.iloc[[0]]
            
        # Reset index so that we can safely index using .loc[0, ...]
        patient = patient.reset_index(drop=True)
        # patient = pd.DataFrame(patient)
        # # transpose into a row with columns of coordinates
        # patient = patient.transpose()
        # # reset index to be able to get longitude with 0 for the longitude and latitude.
        # patient = patient.reset_index(drop = True)
        Vector_df = pd.DataFrame(vector_loc)
        Vector_df.rename(columns = {'coordinates':'vector_loc'}, inplace = True)
        Vector_df[['vector_lon', 'vector_lat']] = pd.DataFrame(Vector_df['vector_loc'].tolist(), index=Vector_df.index)
        Vector_df['patient_lon'] = patient.loc[0, ('longitude')]
        Vector_df['patient_lat'] = patient.loc[0, ('latitude')]
        Vector_df['patient_loc'] = list(zip(Vector_df['patient_lon'],Vector_df['patient_lat']))
        Vector_df['dist_patient'] = Vector_df.apply(lambda row: geopy.distance.distance(row['vector_loc'], row['patient_loc']).meters, axis=1)
        # Only keep the 15 closest vectors. keep='all' so that all responders with the 5 lowest values are kept.
        # This is done to minimize the requests to the API (max = 2000 a day)
        subset_vector = Vector_df.nsmallest(15, 'dist_patient', keep='all')
        coordination_list = [[patient.loc[0, ('longitude')], patient.loc[0, ('latitude')]]]+subset_vector.apply(lambda row: [row["vector_lon"], 
                                                                                                              row["vector_lat"]], axis=1).tolist()
        
        #print(f'There are {len(coordination_list)-1} vector options.')

        # Use the batch function (which is assumed to split the origins into batches, process each batch, 
        # and return a single list of durations corresponding to the responder options).
        duration_results_vec = self.batch_matrix_duration(coordination_list, batch_size=10, profile="foot-walking")

        # Since the batch function returns only the durations for each responder (without the self-distance),
        # its output is a list with length equal to the number of responders.
        subset_vector["duration"] = duration_results_vec
        # duration_results = self.__matrix_duration(coordination_list, profile="driving-car")
        # subset_vector["duration"] = [item for row in duration_results["durations"][1:len(subset_vector)+1] for item in row]
            
        # reset the index of the subset to make indexing possible again
        subset_vector = subset_vector.reset_index(drop = True)
        # select the fastest overall time
        min_idx_vec = subset_vector['duration'].idxmin()
        fastest_Vector = subset_vector.loc[min_idx_vec, 'duration']
        loc_Vector = subset_vector.loc[min_idx_vec, 'vector_loc']
        #route_Vector = subset_vector.loc[min_idx_vec, 'coordinates']
        loc_patient = subset_vector.loc[min_idx_vec, 'patient_loc']
        #print('Duration for vectors found')
        
        dict = {'patient_loc':[loc_patient], 
                'responder_loc':'No responder', 
                'duration_Responder':'No responder',
                'Indirect_Responder_loc':'No AED', 'aed_loc':'No AED',
                'duration_AED':'No AED',
                'vector_loc':[loc_Vector],
                'duration_Vector':[fastest_Vector]}
        df_vector = pd.DataFrame(dict)
        return df_vector

    # Function if direct responders and vectors duration are calculated
    def __direct_vs_vector(self, patient, vector_loc, responder_loc, decline_rate, max_number_responders):
        """
        Function calculating the time there are no aeds close by.
        Used in function: def fastest_time
        Uses function: def __directions
        
        Parameters:
        patient (pd.Series): patients coordinates in a series with name "longitude", "latitude". !!!CASE SENSITIVE!!!
        vector_loc (pd.Dataframe): Dataframe with the coordinates named "coordinates" !!!CASE SENSITIVE!!!. Normally, that is the output of def __closest_location.
        responder_loc (pd.Dataframe): Dataframe with the coordinates named "coordinates" !!!CASE SENSITIVE!!!. Normally, that is the output of def __closest_location.
                
        Returns:
        pd.DataFrame: Dataframe with 
                {'patient_loc':patient location,
            	'responder_loc':Responder location, 
                'duration_Responder':Duration it takes the direct responder to get to the patient, 
                'aed_loc':'No AED',
                'duration_AED':'No AED
                'vector_loc':Vector location,
                'duration_Vector':Duration it takes the vector to get to the patient}
        """
        if isinstance(patient, pd.Series):
            patient = patient.to_frame().T
        elif isinstance(patient, pd.DataFrame) and patient.shape[0] != 1:
            # If patient DataFrame has multiple rows, take the first one (or adjust as needed)
            patient = patient.iloc[[0]]
            
        # Reset index so that we can safely index using .loc[0, ...]
        patient = patient.reset_index(drop=True)
        
        Responder_df = pd.DataFrame(responder_loc)
        Responder_df.rename(columns = {'coordinates':'responder_loc'}, inplace = True)
        Responder_df[['responder_lon', 'responder_lat']] = pd.DataFrame(Responder_df['responder_loc'].tolist(), index=Responder_df.index)
        Responder_df['patient_lon'] = patient.loc[0, ('longitude')]
        Responder_df['patient_lat']  = patient.loc[0, ('latitude')]
        Responder_df['patient_loc'] = list(zip(Responder_df['patient_lon'],Responder_df['patient_lat']))
        
        acceptance_list = []
        for i in range(len(Responder_df)):
                acceptance = int(np.random.choice([0,1], p=[decline_rate, (1-decline_rate)], size = 1))
                acceptance_list.append(acceptance)

        #acceptance = list(np.random.choice([0,1], p=[decline_rate, (1-decline_rate)], size = len(Responder_df)))
        df_acceptance = pd.DataFrame({'responder_loc' : list(Responder_df['responder_loc']),
                                'Probability' : acceptance_list})
        
        # merge the acceptance and the data
        # Filter out the ones that did not exept
        Responder_df = Responder_df.merge(df_acceptance, on='responder_loc')
        Responder_df = Responder_df[Responder_df['Probability'] > 0]

        # If no responders are available, raise an exception
        if Responder_df.empty:
            raise NoResponderAcceptedError("No responder accepted the request.")

        
        Responder_df['dist_patient'] = Responder_df.apply(lambda row: geopy.distance.distance(row['responder_loc'], row['patient_loc']).meters, axis=1)
        # only keep the 15 closest responders. keep='all' so  that all responders with the 5 lowest values are kept.
        # This is done to minimize the requests to the API (max = 2000 a day)  
        subset_responder = Responder_df.nsmallest(max_number_responders, 'dist_patient', keep='all')

        coordination_list_resp = [[patient.loc[0, ('longitude')], patient.loc[0, ('latitude')]]]+subset_responder.apply(lambda row: [row["responder_lon"], 
                                                                                                              row["responder_lat"]], axis=1).tolist()
        
        #print(f'There are {len(coordination_list_resp)-1} direct responder options.')

        # Use the batch function (which is assumed to split the origins into batches, process each batch, 
        # and return a single list of durations corresponding to the responder options).
        duration_results_resp = self.batch_matrix_duration(coordination_list_resp, batch_size=10, profile="foot-walking")

        # Since the batch function returns only the durations for each responder (without the self-distance),
        # its output is a list with length equal to the number of responders.
        subset_responder["duration_direct"] = duration_results_resp
        # duration_results_resp = self.__matrix_duration(coordination_list_resp, profile="foot-walking")
        # subset_responder["duration_direct"] = [item for row in duration_results_resp["durations"][1:len(subset_responder)+1] for item in row]
            
        # reset the index of the subset to make indexing possible again
        subset_responder = subset_responder.reset_index(drop = True)

        # select the fastest overall time
        min_idx = subset_responder['duration_direct'].idxmin()
        fastest_Responder = subset_responder.loc[min_idx, 'duration_direct']
        loc_Responder = subset_responder.loc[min_idx, 'responder_loc']
        # route_Responder = subset_responder.loc[min_idx, 'coordinates_direct']
        #print('Duration for responders found')
        
        Vector_df = pd.DataFrame(vector_loc)
        Vector_df.rename(columns = {'coordinates':'vector_loc'}, inplace = True)
        Vector_df[['vector_lon', 'vector_lat']] = pd.DataFrame(Vector_df['vector_loc'].tolist(), index=Vector_df.index)
        Vector_df['patient_lon'] = patient.loc[0, ('longitude')]
        Vector_df['patient_lat'] = patient.loc[0, ('latitude')]
        Vector_df['patient_loc'] = list(zip(Vector_df['patient_lon'],Vector_df['patient_lat']))
        Vector_df['dist_patient'] = Vector_df.apply(lambda row: geopy.distance.distance(row['vector_loc'], row['patient_loc']).meters, axis=1)
        # Only keep the 15 closest vectors. keep='all' so that all responders with the 5 lowest values are kept.
        # This is done to minimize the requests to the API (max = 2000 a day)
        subset_vector = Vector_df.nsmallest(max_number_responders, 'dist_patient', keep='all')
        coordination_list = [[patient.loc[0, ('longitude')], patient.loc[0, ('latitude')]]]+subset_vector.apply(lambda row: [row["vector_lon"], 
                                                                                                              row["vector_lat"]], axis=1).tolist()
        #print(f'There are {len(coordination_list)-1} vector options.')
        
        # Use the batch function (which is assumed to split the origins into batches, process each batch, 
        # and return a single list of durations corresponding to the responder options).
        duration_results_vec = self.batch_matrix_duration(coordination_list, batch_size=10, profile="foot-walking")

        # Since the batch function returns only the durations for each responder (without the self-distance),
        # its output is a list with length equal to the number of responders.
        subset_vector["duration"] = duration_results_vec
        
        # duration_results = self.__matrix_duration(coordination_list, profile="driving-car")
        # subset_vector["duration"] = [item for row in duration_results["durations"][1:len(subset_vector)+1] for item in row]

        # reset the index of the subset to make indexing possible again
        subset_vector = subset_vector.reset_index(drop = True)
        # select the fastest overall time
        min_idx_vec = subset_vector['duration'].idxmin()
        fastest_Vector = subset_vector.loc[min_idx_vec, 'duration']
        loc_Vector = subset_vector.loc[min_idx_vec, 'vector_loc']
        #route_Vector = subset_vector.loc[min_idx_vec, 'coordinates']
        loc_patient = subset_vector.loc[min_idx_vec, 'patient_loc']
        #print('Duration for vectors found')
        
        dict = {'patient_loc':[loc_patient], 
                'responder_loc':[loc_Responder], 
                'duration_Responder':[fastest_Responder],
                'Indirect_Responder_loc':'No AED', 'aed_loc':'No AED',
                'duration_AED':'No AED', 
                'vector_loc':[loc_Vector],
                'duration_Vector':[fastest_Vector]}
        
        df = pd.DataFrame(dict)
        
        return df

    # Function to simulate survival probability.
    # Used in __fastest_comparisson to decide which responder should be send directly.
    def __survival_probability(self, x, z):
        return 0.9 **(x/60)* 0.97**(z/60)
    
    # Function to get all possible routes through the aeds that are close to the patient
    # Returns a data frame with the coordinates of the Responder, duration through the specific AED,
    # duration for the direct route, and the coordinates of the used AED
    def __possible_routing_direct(self, patient, responder_loc, max_number_responders):
        """
        Function to calculate all possible direct routes.
        Used in function: df __send_responder
        Uses function: def __directions
        
        Parameters:
        patient (pd.Series): patients coordinates in a series with name "longitude", "latitude". !!!CASE SENSITIVE!!!
        responder_loc (pd.Dataframe): Dataframe of responders location with column "longitude", "latitude" for the locations. !!!CASE SENSITIVE!!!
                
        Returns:
        pd.DataFrame: Dataframe with 
                {'responder_loc': responders location,
                'patient_lon': patient longitude,
                'patient_lat': patient latitude,
                'patient_loc': patient location,
                'duration_direct': Duration it takes the direct responder to get to the patient
                }
        """

        if isinstance(patient, pd.Series):
            patient = patient.to_frame().T
        elif isinstance(patient, pd.DataFrame) and patient.shape[0] != 1:
            # If patient DataFrame has multiple rows, take the first one (or adjust as needed)
            patient = patient.iloc[[0]]
            
        # Reset index so that we can safely index using .loc[0, ...]
        patient = patient.reset_index(drop=True)
        
        Responder_df = pd.DataFrame(responder_loc)
        Responder_df.rename(columns={'coordinates': 'responder_loc'}, inplace=True)
        Responder_df[['responder_lon', 'responder_lat']] = pd.DataFrame(
            Responder_df['responder_loc'].tolist(), index=Responder_df.index
        )
        
        # Use simple indexing assuming patient is now a DataFrame with one row.
        Responder_df['patient_lon'] = patient.loc[0, 'longitude']
        Responder_df['patient_lat'] = patient.loc[0, 'latitude']
        Responder_df['patient_loc'] = list(zip(Responder_df['patient_lon'], Responder_df['patient_lat']))
        
        # Uncomment to filter out the ones that have a far away distance as the crow flies
        Responder_df['dist_patient'] = Responder_df.apply(
            lambda row: geopy.distance.distance(row['responder_loc'], row['patient_loc']).meters, axis=1
        )
        
        # Keep only the closest responders, up to max_number_responders.
        Responder_df = Responder_df.nsmallest(max_number_responders, 'dist_patient', keep='all')
        Responder_df = Responder_df.reset_index(drop=True)
        
        coordination_list_resp = (
            [[patient.loc[0, 'longitude'], patient.loc[0, 'latitude']]]
            + Responder_df.apply(lambda row: [row["responder_lon"], row["responder_lat"]], axis=1).tolist()
        )
        #print(f'There are {len(coordination_list_resp)-1} direct responder options.')
        
        # Use the batch function (which is assumed to split the origins into batches, process each batch, 
        # and return a single list of durations corresponding to the responder options).
        duration_results_resp = self.batch_matrix_duration(coordination_list_resp, batch_size=10, profile="foot-walking")

        # Since the batch function returns only the durations for each responder (without the self-distance),
        # its output is a list with length equal to the number of responders.
        Responder_df["duration_direct"] = duration_results_resp
        # duration_results_resp = self.__matrix_duration(coordination_list_resp, profile="foot-walking")
        # Responder_df["duration_direct"] = [item for row in duration_results_resp["durations"][1:len(Responder_df)+1] for item in row]
            
        #Responder_df[['duration_direct', 'route_direct', 'coordinates_direct']] = Responder_df.apply(
        #    lambda row: pd.Series(self.__directions([row['responder_loc'], row['patient_loc']], 
        #                                            profile='foot-walking')).reindex(
        #        ['duration_direct', 'route_direct', 'coordinates_direct'], fill_value=None), axis=1)
        
        #Responder_df[['duration_direct', 'route_direct', 'coordinates_direct']] = Responder_df.apply(
        #    lambda row: pd.Series(self.__directions([row['responder_loc'], row['patient_loc']], 
        #                                            profile='foot-walking')),axis=1)

        return Responder_df
    
    def __possible_routing_indirect(self, patient, responder_loc, aed_loc,  opening_hours, filter_values):
        """
        Function to find all possible indirect routes through AEDs.
        Used in function: def __send_responders
        Uses function: def __closest_location_aed, def __directions
        
        Parameters:
        patient (pd.Series): patients coordinates in a series with name "longitude", "latitude". !!!CASE SENSITIVE!!!
        responder_loc (pd.Dataframe): Dataframe with the responder coordinates named "coordinates" !!!CASE SENSITIVE!!!. Normally, that is the output of def __closest_location.
        aed_loc (pd.Dataframe): Dataframe with the aed coordinates named "coordinates" !!!CASE SENSITIVE!!!. Normally, that is the output of def __closest_location.
        opening_hours (float): Hour in question
        filter_values (list): List with all strings that the AED dataframe column "available" will be subseted by. If None every AED will be considered !!!CASE SENSITIVE!!!
                
        Returns:
        pd.DataFrame: Dataframe with 
                {'responder_loc': responders location,
                'patient_lon': patient longitude,
                'patient_lat': patient latitude,
                'patient_loc': patient location,
                'AED': AED location
                'duration_through_AED': Duration it takes the responder to get to aed and then the patient
                }
        """
        if isinstance(patient, pd.Series):
            patient = patient.to_frame().T
        elif isinstance(patient, pd.DataFrame) and patient.shape[0] != 1:
            # If patient DataFrame has multiple rows, take the first one (or adjust as needed)
            patient = patient.iloc[[0]]
            
        # Reset index so that we can safely index using .loc[0, ...]
        patient = patient.reset_index(drop=True)

        Responder_df = pd.DataFrame(responder_loc)
        Responder_df.rename(columns = {'coordinates':'responder_loc'}, inplace = True)
        Responder_df['patient_lon'] = patient.loc[0, ('longitude')]
        Responder_df['patient_lat']  = patient.loc[0, ('latitude')]
        Responder_df['patient_loc'] = list(zip(Responder_df['patient_lon'],Responder_df['patient_lat']))

        AED_df = pd.DataFrame(aed_loc)
        AED_df.rename(columns = {'coordinates':'AED'}, inplace = True)
        # get the coordinates of the patient
        AED_df['patient_lon'] = patient.loc[0, ('longitude')]
        AED_df['patient_lat'] = patient.loc[0, ('latitude')]
        AED_df['patient_loc'] = list(zip(AED_df['patient_lon'],AED_df['patient_lat']))


        # Create data frames with the coordinates
        # If opening hours hard coded use this!!!!
        # AED_ISO = self.AED_ISO
        AED_ISO = self.AED_ISO
        AED_ISO['AED'] = list(zip(AED_ISO["longitude"], AED_ISO["latitude"]))


        # USE assert
        # Use literals
        # And give a list of opertunities
        # select certain subsets
        if filter_values is not None:
            possible_inputs = set(pd.unique(AED_ISO["available"]))
            assert all(x in possible_inputs for x in filter_values)
            AED_ISO = AED_ISO[AED_ISO["available"].isin(filter_values)]


        AED_df = AED_ISO.merge(AED_df, on='AED')

        # Delete all aeds that are not open yet
        AED_df.loc[:, 'open'] = AED_df.apply(lambda row: 1 if (row['Opens'] < opening_hours and row['Closes'] > opening_hours) else 0, axis=1)
        AED_df = AED_df[AED_df['open'] > 0]

        if AED_df.empty:
            # If the fastest direct responder and thorugh AED responder are different:
            # - Take the fastest responders for both
            raise NoAEDResponderAcceptedError("No AED is available during the opening hours.")

        # Convert AED_df to a GeoDataFrame with the correct CRS
        AED_df = gpd.GeoDataFrame(AED_df, geometry='geometry', crs="EPSG:4326")
        all_points = []  # Store points and midpoints as tuples

        # Iterate through each row in AED_df to use the 'Iso' polygons
        for _, row in AED_df.iterrows():
            polygon = row['geometry']
            midpoint = row['AED']
            
            points_in_polygon = self.__closest_location_aed(Responder_df, polygon)

            # Add the points along with the midpoint of the polygon
            all_points.extend([(point, midpoint) for point in points_in_polygon])

        # Create a DataFrame with the points and their associated polygon midpoints
        result_df = pd.DataFrame(all_points, columns=['responder_loc', 'AED_coordinates'])
        result_df = result_df.drop_duplicates()
        # result_df[['aed_lon', 'aed_lat']] = pd.DataFrame(
        #     result_df['AED_coordinates'].tolist(), index=Responder_df.index
        # )
        # get the coordinates of the patient
        result_df['patient_lon'] = patient.loc[0, ('longitude')]
        result_df['patient_lat'] = patient.loc[0, ('latitude')]
        result_df['patient_loc'] = list(zip(result_df['patient_lon'],result_df['patient_lat']))

        # Build a unique list of AED coordinates from the DataFrame
        unique_AEDs = result_df['AED_coordinates'].unique()

        #print(f'{len(unique_AEDs)} AEDs are possible')

        # Get patient coordinate as a list
        patient_coord = [patient.loc[0, 'longitude'], patient.loc[0, 'latitude']]
        # Create the input for the distance matrix: the first entry is the patient destination.
        # The rest of the entries are all unique AED coordinates.
        coords_AED_to_patient = [patient_coord] + [list(aed) for aed in unique_AEDs]
    
        # Call the distance matrix function once.
        # The function will compute durations from each AED (i.e. each subsequent point)
        # to the patient (the first point) in one batch.
        # Assume that coords_AED_to_patient is built as:
        # [patient_coord] + [list(aed) for aed in unique_AEDs]
        matrix_aed_patient = self.batch_matrix_duration(coordinates=coords_AED_to_patient, profile='foot-walking')

        # Since matrix_aed_patient is a list, update the dictionary comprehension:
        aed_to_patient_durations = {
            aed: matrix_aed_patient[i]  # using numeric index because the output is a list
            for i, aed in enumerate(unique_AEDs)
        }

        # Group the DataFrame by AED location and compute durations for each group in one call.
        df_grouped = result_df.groupby('AED_coordinates', group_keys=False).apply(
            lambda grp: self.__compute_responder_to_aed_durations(grp, self.batch_matrix_duration)
        )


        # Now add the AED-to-patient durations from the batch computed dictionary.
        df_grouped['duration_AED_to_patient'] = df_grouped['AED_coordinates'].apply(
            lambda aed: aed_to_patient_durations[aed]
        )

        # The total travel time is the sum of the two durations.
        df_grouped['duration_through_AED'] = df_grouped['duration_responder_to_AED'] + df_grouped['duration_AED_to_patient']
                
        result_df = result_df.merge(
            df_grouped[['responder_loc', 'AED_coordinates', 'duration_through_AED',
                        'duration_responder_to_AED', 'duration_AED_to_patient']],
            on=['responder_loc', 'AED_coordinates'],
            how='left'
        )
        # result_df[['duration_through_AED', 'route_indirect', 'coordinates_through_AED']] = result_df.apply(
        #     lambda row: pd.Series(self.__directions([row['responder_loc'], row['AED_coordinates'], row['patient_loc']], 
        #                                             profile='foot-walking')).reindex(
        #         ['duration_through_AED', 'route_indirect', 'coordinates_through_AED'], fill_value=None), axis=1)
        #result_df[['duration_through_AED', 'route_indirect', 'coordinates_through_AED']] = result_df.apply(
        #    lambda row: pd.Series(self.__directions([row['responder_loc'], row['AED_coordinates'], row['patient_loc']], 
        #                                            profile='foot-walking')),axis=1)
        return result_df
    



    def __compute_responder_to_aed_durations(self, group, matrix_duration_func):
        """
        For all rows sharing the same AED coordinate, compute:
        - Duration from each responder (origin) to the AED (destination).
        The coordinates list is built with the AED as the first element and then all responder locations.
        """
        # The AED coordinate is the same for all responders in this group.
        aed = group.iloc[0]['AED_coordinates']
        # Ensure the coordinates are lists, not tuples.
        coords = [list(aed)] + [list(loc) for loc in group['responder_loc']]
        #print(f'{len(coords)} indirect responders')
        
        # Call the matrix function (e.g., your batch_matrix_duration function)
        matrix = matrix_duration_func(coordinates=coords, profile="foot-walking")
        
        # Adjust the indexing: since matrix now only has responder durations (len(coords)-1 entries),
        # we subtract 1 from our index when extracting durations.
        durations = [matrix[i-1] for i in range(1, len(coords))]
        
        group = group.copy()  # Avoid modifying the original DataFrame directly.
        group['duration_responder_to_AED'] = durations
        return group
    

    def batch_matrix_duration(self, coordinates, batch_size=10, profile='foot-walking'):
        if len(coordinates) < 2:
            raise ValueError("At least one origin and one destination required.")

        destination = coordinates[0]
        origins = coordinates[1:]
        combined_results = []

        i = 0
        while i < len(origins):
            batch_processed = False
            for size in [batch_size, batch_size // 2, max(1, batch_size // 4)]:
                batch_origins = origins[i:i+size]
                batch_coords = [destination] + batch_origins
                try:
                    matrix = self.__matrix_duration(coordinates=batch_coords, profile=profile)
                    for j in range(1, len(matrix['durations'])):
                        combined_results.append(matrix['durations'][j][0])
                    i += size  # move index forward
                    batch_processed = True
                    break  # exit fallback loop
                except Exception as e:
                    continue  # try smaller batch
            
            if not batch_processed:
                # Skip this batch and log the failure
                print(f"[Warning] Failed to process origins at index {i}. Skipping batch.")
                i += 1  # skip just one origin to avoid getting stuck in an infinite loop

        return combined_results

    # Function to find the responder that is send directly and through the AED.
    # The results are plotted using plotly
    # patient = location of patient
    # Responder = Dataframe of responders and their location
    # AED = list of aeds
    # max_number_responders = total amount of responders that will be contacted
    # AED_rate = proportion of max_number_responders that is requethrough AED
    # decline_rate = proportion of people excpected to decline the call to action
    def __send_responders(self, patient, responder_loc, aed_loc, max_number_responders, decline_rate, opening_hours, filter_values):
        """
        Function to find the optimal responders to send directly or indirectly.
        Used in function: def __fastest_comparisson
        Uses function: def __possible_routing_direct, def __possible_routing_indirect, def __survival_probability
        
        Parameters:
        patient (pd.Series): patients coordinates in a series with name "longitude", "latitude". !!!CASE SENSITIVE!!!
        responder_loc (pd.Dataframe): Dataframe with the responder coordinates named "coordinates" !!!CASE SENSITIVE!!!. Normally, that is the output of def __closest_location.
        aed_loc (pd.Dataframe): Dataframe with the aed coordinates named "coordinates" !!!CASE SENSITIVE!!!. Normally, that is the output of def __closest_location.
        max_number_responders (int): Maximal number of responders contacted to reply to the call
        decline_rate (float): likelihood of an responder of declining
        opening_hours (float): Hour in question
        filter_values (list): List with all strings that the AED dataframe column "available" will be subseted by. If None every AED will be considered !!!CASE SENSITIVE!!!
                
        Returns:
        dictionary: with 
                {'coord_direct': Coordinates of the direct responder, 
                'duration_direct': Duration it takes the responder to get to the patient,
                'coord_AED': Coordinates of the indirect responder, 
                'AED_coordinates': Coordinates of the AED, 
                'duration_through_AED':Duration it takes the responder to get to aed and then the patient}
        """
        df_duration_direct = self.__possible_routing_direct(patient, responder_loc, max_number_responders)
        df_duration_indirect = self.__possible_routing_indirect(patient, responder_loc, aed_loc,  opening_hours, filter_values)

        df_duration_direct = df_duration_direct.sort_values(by=['duration_direct'], ascending=True)
        df_duration_direct = df_duration_direct.nsmallest(round((max_number_responders)), 'duration_direct')
        # Only keep the fastest route through AED for every responder
        df_duration_indirect.sort_values(by=['duration_through_AED'], ascending=True).drop_duplicates('responder_loc').sort_index()
        df_duration_indirect = df_duration_indirect.nsmallest(round((max_number_responders)), 'duration_through_AED')

        # create list of all possible responders
        possible_responder = list(df_duration_direct['responder_loc']) + list(df_duration_indirect['responder_loc'])
        # Only take individual responders
        seen = set()
        possible_responder = [val for val in possible_responder if val not in seen and (seen.add(val) or True)]
        # Randomly generate numbers with a decline_rate. 

        acceptance_list = []
        for i in range(len(possible_responder)):
                acceptance = int(np.random.choice([0,1], p=[decline_rate, (1-decline_rate)], size = 1))
                acceptance_list.append(acceptance)
                
        # Data frame with all the possible responders and their probability to accept the call. 1 for acceptance
        #acceptance = list(np.random.choice([0,1], p=[decline_rate, (1-decline_rate)], size = len(Responder_df))        
        # merge the acceptance and the data
        # Filter out the ones that did not exept
        df_acceptance = pd.DataFrame({'responder_loc' : possible_responder,
                                'Probability' : acceptance})
        
        # merge the acceptance and the data
        # Filter out the ones that did not exept
        df_duration_direct = df_duration_direct.merge(df_acceptance, on='responder_loc')
        df_duration_direct = df_duration_direct[df_duration_direct['Probability'] > 0]
        df_duration_direct = df_duration_direct.reset_index(drop=True)
        df_duration_indirect = df_duration_indirect.merge(df_acceptance, on='responder_loc')
        df_duration_indirect = df_duration_indirect[df_duration_indirect['Probability'] > 0]
        df_duration_indirect = df_duration_indirect.reset_index(drop=True)

        if df_duration_direct.empty:
            raise NoResponderAcceptedError("Neither AED responder nor direct responder accepted the request.")

        if df_duration_indirect.empty:
            # If the fastest direct responder and thorugh AED responder are different:
            # - Take the fastest responders for both
            raise NoAEDResponderAcceptedError("No AED responder accepted the request.")

        if len(df_duration_direct) <2 or len(df_duration_indirect)<2:
            print('We have a problem here')

        fastest_aed = df_duration_indirect.loc[0]
        fastest_cpr = df_duration_direct.loc[0]
        second_fastest_aed = df_duration_indirect.loc[1]
        second_fastest_cpr = df_duration_direct.loc[1]

        # Fastest responder going for the AED
        # Check if the 2nd fastest direct responder is faster than the fastes direct through AED
        if second_fastest_cpr['duration_direct'] < fastest_aed['duration_through_AED']:
            # if so, z equal to CPR by 2nd fastest responder
            CPR_time = fastest_aed['duration_through_AED'] - second_fastest_cpr['duration_direct']
            surv_A = self.__survival_probability(second_fastest_cpr['duration_direct'], CPR_time)
        else:
            # - z equals zero as no one does any CPR
            surv_A = self.__survival_probability(fastest_aed['duration_through_AED'], 0)
        # 2nd fastest responder arriving with AED
        # - time until AED arrives minus time CPR arrives is the time without CPR
        surv_B = self.__survival_probability(fastest_cpr['duration_direct'], second_fastest_aed['duration_through_AED']-fastest_cpr['duration_direct'])    
                
        # Check if the fastest through AED is the same as the fastest direct 
        if fastest_aed['responder_loc']==fastest_cpr['responder_loc']:
            # Find best strategy which is the maximal survival chances
            best_strategy = max(surv_A, surv_B)
            # Send responders
            if best_strategy == surv_A:
                # - Second fastest direct time will be send directly  
                # - Fastest direct and AED responder will be send through the AED
                coord_direct = second_fastest_cpr['responder_loc']
                duration_direct = second_fastest_cpr['duration_direct']
                #route_direct = second_fastest_cpr['route_direct']
                coord_AED =  fastest_aed['responder_loc']
                AED_coordinates = fastest_aed['AED_coordinates']
                duration_indirect = fastest_aed['duration_through_AED']
                #route_indirect = fastest_aed['route_indirect']
            # If this is not true:
            # - Fastes direct responder will be send directly
            # - Second fastest through AED responder will be send through the AED
            else:
                coord_direct = fastest_cpr['responder_loc']
                duration_direct = fastest_cpr['duration_direct']
                #route_direct = fastest_cpr['route_direct']
                coord_AED = second_fastest_aed['responder_loc']
                AED_coordinates = second_fastest_aed['AED_coordinates']
                duration_indirect = second_fastest_aed['duration_through_AED']
                #route_indirect = second_fastest_aed['route_indirect']
        else:
            # If the fastest direct responder and thorugh AED responder are different:
            # - Take the fastest responders for both
            coord_direct = fastest_cpr['responder_loc']
            duration_direct = fastest_cpr['duration_direct']
            #route_direct = fastest_cpr['route_direct']
            coord_AED = fastest_aed['responder_loc']
            AED_coordinates = fastest_aed['AED_coordinates']
            duration_indirect = fastest_aed['duration_through_AED']
            #route_indirect = fastest_aed['route_indirect']
            
            
        return {'coord_direct': coord_direct, 'duration_direct':duration_direct,
        'coord_AED': coord_AED, 'AED_coordinates': AED_coordinates, 'duration_through_AED':duration_indirect}

    # Function to build a data frame with the fastest direct, indirect and vector duration.
    def __fastest_comparisson(self, patient, vector_loc, responder_loc, aed_loc, max_number_responders, decline_rate, opening_hours, filter_values):
        """
        Function calculating the time of the direct, indirect responder and vector to arrive at the patient.
        Used in function: def fastest_time
        Uses function: def send_responder, def __directions
        
        Parameters:
        patient (pd.Series): patients coordinates in a series with name "longitude", "latitude". !!!CASE SENSITIVE!!!
        vector_loc (pd.Dataframe): Dataframe with the coordinates named "coordinates" !!!CASE SENSITIVE!!!. Normally, that is the output of def __closest_location.
        responder_loc (pd.Dataframe): Dataframe with the coordinates named "coordinates" !!!CASE SENSITIVE!!!. Normally, that is the output of def __closest_location.
        aed_loc (pd.Dataframe): Dataframe with the coordinates named "coordinates" !!!CASE SENSITIVE!!!. Normally, that is the output of def __closest_location.
        max_number_responders(int): Numbers of responders contacted
        decline_rate (float): !!!Must be between 0 and 1!!! Represents the percentage of responders declining the call to action.
        opening_hours (float): Time of the day. Checks of the AED is open during that time. 
        filter_values (list): List with all strings that the AED dataframe column "available" will be subseted by. If None every AED will be considered !!!CASE SENSITIVE!!!
                
        Returns:
        pd.DataFrame: Dataframe with 
                {'patient_loc':patient location,
            	'responder_loc':Responder location, 
                'duration_Responder':Duration it takes the direct responder to get to the patient, 
                'aed_loc':AED location,
                'duration_AED':Duration it takes the indirect responder to get to the patient, 
                'vector_loc':Vector location,
                'duration_Vector':Duration it takes the vector to get to the patient}
        """
        responders_send = self.__send_responders(patient,  responder_loc, aed_loc, max_number_responders, decline_rate, opening_hours, filter_values)
        loc_Responder = responders_send['coord_direct']
        fastest_Responder = responders_send['duration_direct']
        #route_Responder = responders_send['route_direct']
        loc_AED = responders_send['AED_coordinates']
        loc_indirect_Responder = responders_send['coord_AED']
        fastest_AED = responders_send['duration_through_AED']
        #route_AED = responders_send['route_indirect']
        
        if isinstance(patient, pd.Series):
            patient = patient.to_frame().T
        elif isinstance(patient, pd.DataFrame) and patient.shape[0] != 1:
            # If patient DataFrame has multiple rows, take the first one (or adjust as needed)
            patient = patient.iloc[[0]]
            
        # Reset index so that we can safely index using .loc[0, ...]
        patient = patient.reset_index(drop=True)
        
        Vector_df = pd.DataFrame(vector_loc)
        Vector_df.rename(columns = {'coordinates':'vector_loc'}, inplace = True)
        Vector_df[['vector_lon', 'vector_lat']] = pd.DataFrame(Vector_df['vector_loc'].tolist(), index=Vector_df.index)
        Vector_df['patient_lon'] = patient.loc[0, ('longitude')]
        Vector_df['patient_lat'] = patient.loc[0, ('latitude')]
        Vector_df['patient_loc'] = list(zip(Vector_df['patient_lon'],Vector_df['patient_lat']))
        Vector_df['dist_patient'] = Vector_df.apply(lambda row: geopy.distance.distance(row['vector_loc'], row['patient_loc']).meters, axis=1)
        # Only keep the 15 closest vectors. keep='all' so that all responders with the 5 lowest values are kept.
        # This is done to minimize the requests to the API (max = 2000 a day)
        subset_vector = Vector_df.nsmallest(15, 'dist_patient', keep='all')
        coordination_list = [[patient.loc[0, ('longitude')], patient.loc[0, ('latitude')]]]+subset_vector.apply(lambda row: [row["vector_lon"], 
                                                                                                              row["vector_lat"]], axis=1).tolist()
        
        #print(f'There are {len(coordination_list)-1} vector options.')

        # Use the batch function (which is assumed to split the origins into batches, process each batch, 
        # and return a single list of durations corresponding to the responder options).
        duration_results_vec = self.batch_matrix_duration(coordination_list, batch_size=10, profile="foot-walking")

        # Since the batch function returns only the durations for each responder (without the self-distance),
        # its output is a list with length equal to the number of responders.
        subset_vector["duration"] = duration_results_vec

        # duration_results = self.__matrix_duration(coordination_list, profile="driving-car")
        # subset_vector["duration"] = [item for row in duration_results["durations"][1:len(subset_vector)+1] for item in row]
        # reset the index of the subset to make indexing possible again
        subset_vector = subset_vector.reset_index(drop = True)
        # select the fastest overall time
        min_idx_vec = subset_vector['duration'].idxmin()
        fastest_Vector = subset_vector.loc[min_idx_vec, 'duration']
        loc_Vector = subset_vector.loc[min_idx_vec, 'vector_loc']
        #route_Vector = subset_vector.loc[min_idx_vec, 'coordinates']
        loc_patient = subset_vector.loc[min_idx_vec, 'patient_loc']
        #print('Duration for vectors found')
        
        dict = {'patient_loc':[loc_patient], 
                'responder_loc':[loc_Responder], 
                'duration_Responder':[fastest_Responder],
                'Indirect_Responder_loc':[loc_indirect_Responder], 'aed_loc':[loc_AED],
                'duration_AED':[fastest_AED],
                'vector_loc':[loc_Vector],
                'duration_Vector':[fastest_Vector]}
        df = pd.DataFrame(dict)
        
        return df