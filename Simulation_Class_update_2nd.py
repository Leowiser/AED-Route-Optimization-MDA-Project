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

class DistributionError(Exception):
    '''
    Raised when distribution in opening hours does not add to 1
    '''
    def __init__(self, distribution, message="Distribution must sum to 1"):
        self.distribution = distribution
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.distribution} -> {self.message}'

#--------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------



class simulation:
    def __init__(self):
        self.Client_ors = openrouteservice.Client(base_url='http://ec2-3-76-12-187.eu-central-1.compute.amazonaws.com:8080/ors')
        self.AED_Iso = gpd.read_file('C:/Users/leonw/OneDrive - KU Leuven/Documents/GitHub/AED-Route-Optimization-MDA-Project/Data/temp.gpkg', layer='AED_data')


    # function to find the AEDs and Responders in a 10 minute walking distance from the patient
    # Nearly same as closest Responders
    def closest_location(self, Patient, Location, profile = "foot-walking", threshold =600):
        '''
        Function that gets the isochrone around the patient and finds all locations in the isochrone.
        Used in funciton: def fastest_time
        Uses function:

        Parameters:
        Patient (pd.Series): Patients coordinates in a series with name "longitude", "latitude". !!!CASE SENSITIVE!!!
        Location (pd.Dataframe): Dataframe with column "longitude", "latitude" for the locations. !!!CASE SENSITIVE!!!
        profile (string): Can be either foot-waling, car-driving, cycling-* [-regular, -electric, -road, -mountain], driving-hgv, or wheelchair.
        threshold (integer): Time in seconds it takes to travel to the furthest away point still included in the area.
        
        Returns:
        pd.Dataframe: Returns dataframe with the coordinates of the locations in the threshold distance.
                      Has one column only with coordinates as tuple (longitude, latitude).
        '''
     

        # set the parameters for conducting an isochrone search
        isochrones_parameters = {
        'locations':  [(Patient['longitude'], Patient['latitude'])],
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
        gdf = gpd.GeoDataFrame({'geometry': [geometry.Polygon(isochrone['features'][0]['geometry']['coordinates'][0])]})

        # Transform the coordinates to a geodataframe points
        # Define the points of all locations
        points = list([geometry.Point(Location.loc[i,"longitude"], Location.loc[i,"latitude"]) for i in range(len(Location))])

        # Check which points are within the polygon.
        # Create tuple that includes all points in 10 minute walking distance from the patient
        coordinate_tuples = [[(point.x, point.y)] for point in points if point.within(gdf.loc[0,'geometry'])]
        df = pd.DataFrame(coordinate_tuples, columns = ['coordinates'])
        # Returns a list of tuples of the coordinates of the responders. (long, lat)
        
        return df
    
    def closest_location_AED(self, Location, polygon):
        '''
        Function that finds all locations inside an area.
        Used in function: def possible_routing_indirect
        Uses function:

        Parameters:
        Location (pd.Dataframe): Dataframe with column "longitude", "latitude" for the locations. !!!CASE SENSITIVE!!!
        polygon (shapely.geometry.polygon.Polygon): Polygon of the area in question.
        
        Returns:
        pd.Dataframe: Returns a list of tuples of the coordinates of the loations inside the area (longitude, latitude).
        '''


        d = {'geometry': [polygon]}
        gdf = gpd.GeoDataFrame(d) 
        
        # Get the coordinates (lat, lon) for all Responders.
        df = pd.DataFrame({'coords':Location.loc[:,"Responder_loc"]}) # lat as 1st column longitude as 2nd

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
    

    def directions(self, coordinates, profile = 'foot-walking', sleep = 1):
        """
        Function that gets the duration, route and coordinates of a route. The route can be direct or go through other points first
        Used in function: def fastest_vector, def direct_vs_vector, def possible_routing_direct, def possible_routing_indirect, def fastest_comparisson
        Uses function:
        
        Parameters:
        coordinates (list): List containing [[lon,lat],[lon,lat],...] in the order of routing.
        prifile (string): can be either foot-waling, car-driving, cycling-* [-regular, -electric, -road, -mountain], driving-hgv, or wheelchair .
        sleep (float): seconds to wait.
        
        Returns:
        ditionary: Returns dictionary with duration, route and coordinates of the route.
        """
        client = self.Client_ors
        time.sleep(sleep)
        route = client.directions(coordinates=coordinates,
                                   profile=profile,
                                   format='geojson',
                                   validate=False)
        # empty dictionary to be filled later
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
    
    def assign_opening_hours(self, distribution = {(6.0, 22.0): 0.5,(0.0, 24.0): 0.05,(8.0, 20.0): 0.3,(9.0, 18.0):0.15}):
        """
        Assigns opening hours to AEDs based on a specified distribution.
        Used in function: def possible_routing_indirect
        Uses function:
        
        Parameters:
        distribution (dict): Dictionary where keys are tuples of (opening_hour, closing_hour) and values are proportions. Distribution proportions must sum to 1.
        + Uses self.AED_Iso 
        
        Returns:
        pd.DataFrame: Updated DataFrame of self.AED_Iso with 'opening_hour' and 'closing_hour' columns.
        """
        
        AEDs = self.AED_Iso
        AEDs['AED'] = list(zip(AEDs["lon"], AEDs["lat"]))
        # Ensure proportions sum to 1 otherwise raise Exception DistributionError
        if sum(distribution.values()) != 1:
            raise DistributionError(distribution)

        
        # Sample opening hours based on the distribution
        choices = list(distribution.keys())
        probabilities = list(distribution.values())
        assigned_hours = np.random.choice(len(choices), size=len(AEDs), p=probabilities)
        
        # Assign opening and closing hours
        AEDs["opening_hour"] = [choices[i][0] for i in assigned_hours]
        AEDs["closing_hour"] = [choices[i][1] for i in assigned_hours]
        
        return AEDs

    # Function to get all possible routes through the AEDs that are close to the patient
    # Returns a data frame with the coordinates of the Responder, duration through the specific AED,
    # duration for the direct route, and the coordinates of the used AED
    def fastest_time(self, Patient, Responders, AEDs, Vectors, decline_rate, N_responder, opening_hour, Dist_responder = 600, Dist_AED = 600, Dist_Vector = 600, 
                     distribution = {(6.0, 22.0): 0.5,(0.0, 24.0): 0.05,(8.0, 20.0): 0.3,(9.0, 18.0):0.15}):
        """
        Main function calculating the time until the first AED arrives.
        Used in function:
        Uses function: def closest_location, def fastest_vector, def direct_vs_vector, def fastest_comparisson
        
        Parameters:
        Patient (pd.Series): Patients coordinates in a series with name "longitude", "latitude". !!!CASE SENSITIVE!!!
        Responders (pd.Dataframe): Dataframe of responders location with column "longitude", "latitude" for the locations. !!!CASE SENSITIVE!!!
        AEDs (pd.Dataframe): Dataframe of AED location with column "longitude", "latitude" for the locations. !!!CASE SENSITIVE!!!
        Vectors (pd.Dataframe): Dataframe of vector (ambulance,...) location with column "longitude", "latitude" for the locations. !!!CASE SENSITIVE!!!
        decline_rate (float): !!!Must be between 0 and 1!!! Represents the percentage of responders declining the call to action.
        N_responder: 
        opening_hour(float): time of the incident
        Dist_responder = 600 (float): Time in seconds it takes to travel by foot to the furthest away point still included in the area (Used to find responders in that area).
        Dist_AED = 600 (float): Time in seconds it takes to travel by foot to the furthest away point still included in the area (Used to find AEDs in that area).
        Dist_Vector = 600 (float): Time in seconds it takes to travel by car to the furthest away point still included in the area (Used to find Vectors in that area).
        distribution (dictionary): Dictionary where keys are tuples of (opening_hour, closing_hour) and values are proportions. !!!Distribution proportions must sum to 1!!!
        
        Returns:
        pd.DataFrame: Dataframe with 
                {'Patient_loc':Patient Location,
            	'Responder_loc':Responders Location, 
                'duration_Responder':Duration it takes the direct responder to get to the patient, 
                'AED_loc':AEDs Location,
                'duration_AED':Duration it takes the indirect responder to get to the patient,
                'Vector_loc':Vector Location,
                'duration_Vector':Duration it takes the vector to get to the patient}
        """

        # Time that the isochrones covers in seconds
        Responders_loc = self.closest_location(Patient, Responders,threshold=Dist_responder)
        # Time that the isochrones covers in seconds
        AED_loc = self.closest_location(Patient, AEDs, threshold=Dist_AED)
        # Time that the isochrones covers in seconds
        Vector_loc = self.closest_location(Patient, Vectors, profile = 'driving-car', threshold = Dist_Vector)

        # Set empty Dataframe to be fillled later
        # Needed?
        df_duration = pd.DataFrame(columns = ['Patient_loc', 'Responder_loc', 'duration_Responder', 
                                              'AED_loc','duration_AED','Vector_loc', 'duration_Vector'])
        
        # Check if any vector is reached in 10 minutes. 
        # If this is not the case the isochrone radius is increased for 2 minutes until at least one vector is found. 
        Dist_Vector = Dist_Vector
        while (len(Vector_loc) == 0):
            print('No vector is close by. Increase of thresholds')
            Dist_Vector += 120
            Vector_loc = self.closest_location(Patient, Vectors, profile = 'driving-car', threshold = Dist_Vector)
        
        # If there are no responders close by the isochrone radius for them and the AED is increased to the same time as the one of the vector before.
        # This is done as the Responders could still be faster than the Vectors and thus incerease the survival rate.
        if (len(Responders_loc)==0):
            Responders_loc = self.closest_location(Patient, Responders, threshold=Dist_Vector)
            AED_loc = self.closest_location(Patient, AEDs, threshold=Dist_Vector)
        else:
            pass
        
        decline_rate = decline_rate
        opening_hour = opening_hour
        # If the Responder is still 0 only the Vectors time will be calculated
        if ((len(Responders_loc) == 0) and (len(Vector_loc) > 0)):
            print('No responder is in a 10 minute radius')
            df_duration = self.fastest_vector(Patient, Vector_loc)
        # Otherwise the two have to be compared.
        elif ((len(Responders_loc) > 0) and (len(Vector_loc) > 0)):
            if ((len(AED_loc) == 0) or (len(Responders_loc)<2)):
                print('Comparing direct responder vs. vectors')
                try:
                    df_duration = self.direct_vs_vector(Patient, Vector_loc, Responders_loc, decline_rate)
                except NoResponderAcceptedError as e:
                    print(f"Warning: {e}. Falling back to fastest vector.")
                    df_duration = self.fastest_vector(Patient, Vector_loc)
            else:
                print('Comparing responders vs. vectors')
                try:
                    df_duration = self.fastest_comparisson(Patient, Vector_loc, Responders_loc, AED_loc, N_responder, decline_rate, opening_hour, distribution = distribution)
                except NoResponderAcceptedError as e:
                    print(f"Warning: {e}. Falling back to fastest vector.")
                    df_duration = self.fastest_vector(Patient, Vector_loc)
                except NoAEDResponderAcceptedError as e2:
                    print(f"Warning: {e2}. Falling back to direct vs. vector.")
                    df_duration = self.direct_vs_vector(Patient, Vector_loc, Responders_loc, decline_rate)
        # Returns a data frame.          
        return df_duration

    # Function when only the duration for the vector is needed.
    def fastest_vector(self, Patient, Vector_loc):
        """
        Function calculating the time the Vector is the only thing near.
        Used in function: def fastest_time
        Uses function: def directions
        
        Parameters:
        Patient (pd.Series): Patients coordinates in a series with name "longitude", "latitude". !!!CASE SENSITIVE!!!
        Vectors (pd.Dataframe): Dataframe with the coordinates named "coordinates" !!!CASE SENSITIVE!!!. Normally, that is the output of def closest_location.
        decline_rate (float): !!!Must be between 0 and 1!!! Represents the percentage of responders declining the call to action.
                
        Returns:
        pd.DataFrame: Dataframe with 
                {'Patient_loc':Patient Location,
            	'Responder_loc':'No responder', 
                'duration_Responder':'No responder', 
                'AED_loc':'No AED',
                'duration_AED':'No AED
                'Vector_loc':Vector Location,
                'duration_Vector':Duration it takes the vector to get to the patient}
        """

        Patient = pd.DataFrame(Patient)
        # transpose into a row with columns of coordinates
        Patient = Patient.transpose()
        # reset index to be able to get longitude with 0 for the longitude and latitude.
        Patient = Patient.reset_index(drop = True)
        Vector_df = pd.DataFrame(Vector_loc)
        Vector_df.rename(columns = {'coordinates':'Vector_loc'}, inplace = True)
        Vector_df['Patient_lon'] = Patient.loc[0, ('longitude')]
        Vector_df['Patient_lat'] = Patient.loc[0, ('latitude')]
        Vector_df['Patient_loc'] = list(zip(Vector_df['Patient_lon'],Vector_df['Patient_lat']))
        Vector_df['dist_patient'] = Vector_df.apply(lambda row: geopy.distance.distance(row['Vector_loc'], row['Patient_loc']).meters, axis=1)
        # Only keep the 15 closest vectors. keep='all' so that all responders with the 5 lowest values are kept.
        # This is done to minimize the requests to the API (max = 2000 a day)
        subset_vector = Vector_df.nsmallest(15, 'dist_patient', keep='all')
        subset_vector['duration']=[self.directions([i, Patient_cood], profile = 'driving-car')['duration'] for i, 
                                          Patient_cood in zip(subset_vector['Vector_loc'], subset_vector['Patient_loc'])]
        # reset the index of the subset to make indexing possible again
        subset_vector = subset_vector.reset_index(drop = True)
        # select the fastest overall time
        fastest_Vector = subset_vector.iloc[subset_vector.idxmin()['duration']]['duration']
        loc_Vector = subset_vector.iloc[subset_vector.idxmin()['duration']]['Vector_loc']
        loc_Patient = subset_vector.iloc[subset_vector.idxmin()['duration']]['Patient_loc']
        print('Duration for Vectors found')
        
        dict = {'Patient_loc':[loc_Patient], 
                'Responder_loc':'No responder', 
                'duration_Responder':'No responder', 'AED_loc':'No AED','duration_AED':'No AED',
                'Vector_loc':[loc_Vector],
                'duration_Vector':[fastest_Vector]}
        df_vector = pd.DataFrame(dict)
        return df_vector

    # Function if direct responders and vectors duration are calculated
    def direct_vs_vector(self, Patient, Vector_loc, Responder_loc, decline_rate):
        """
        Function calculating the time there are no AEDs close by.
        Used in function: def fastest_time
        Uses function: def directions
        
        Parameters:
        Patient (pd.Series): Patients coordinates in a series with name "longitude", "latitude". !!!CASE SENSITIVE!!!
        Vector_loc (pd.Dataframe): Dataframe with the coordinates named "coordinates" !!!CASE SENSITIVE!!!. Normally, that is the output of def closest_location.
        Responder_loc (pd.Dataframe): Dataframe with the coordinates named "coordinates" !!!CASE SENSITIVE!!!. Normally, that is the output of def closest_location.
                
        Returns:
        pd.DataFrame: Dataframe with 
                {'Patient_loc':Patient Location,
            	'Responder_loc':Responder Location, 
                'duration_Responder':Duration it takes the direct responder to get to the patient, 
                'AED_loc':'No AED',
                'duration_AED':'No AED
                'Vector_loc':Vector Location,
                'duration_Vector':Duration it takes the vector to get to the patient}
        """

        Patient = pd.DataFrame(Patient)
        # transpose into a row with columns of coordinates
        Patient = Patient.transpose()
        # reset index to be able to get longitude with 0 for the longitude and latitude.
        Patient = Patient.reset_index(drop = True)
        Responder_df = pd.DataFrame(Responder_loc)
        Responder_df.rename(columns = {'coordinates':'Responder_loc'}, inplace = True)
        Responder_df['Patient_lon'] = Patient.loc[0, ('longitude')]
        Responder_df['Patient_lat']  = Patient.loc[0, ('latitude')]
        Responder_df['Patient_loc'] = list(zip(Responder_df['Patient_lon'],Responder_df['Patient_lat']))

        acceptance = list(np.random.choice([0,1], p=[decline_rate, (1-decline_rate)], size = len(Responder_df)))
        df_acceptance = pd.DataFrame({'Responder_loc' : list(Responder_df['Responder_loc']),
                                'Probability' : acceptance})
        
        # merge the acceptance and the data
        # Filter out the ones that did not exept
        Responder_df = Responder_df.merge(df_acceptance, on='Responder_loc')
        Responder_df = Responder_df[Responder_df['Probability'] > 0]

        # If no responders are available, raise an exception
        if Responder_df.empty:
            raise NoResponderAcceptedError("No responder accepted the request.")

        
        Responder_df['dist_patient'] = Responder_df.apply(lambda row: geopy.distance.distance(row['Responder_loc'], row['Patient_loc']).meters, axis=1)
        # only keep the 15 closest responders. keep='all' so  that all responders with the 5 lowest values are kept.
        # This is done to minimize the requests to the API (max = 2000 a day)  
        subset_responder = Responder_df.nsmallest(15, 'dist_patient', keep='all')
        subset_responder['duration_direct']=[self.directions([i, Patient_cood], profile = 'foot-walking')['duration'] for i,
                                             Patient_cood in zip(subset_responder['Responder_loc'], subset_responder['Patient_loc'])]
        # reset the index of the subset to make indexing possible again
        subset_responder = subset_responder.reset_index(drop = True)
        # select the fastest overall time
        fastest_Responder = subset_responder.iloc[subset_responder.idxmin()['duration_direct']]['duration_direct']
        loc_Responder = subset_responder.iloc[subset_responder.idxmin()['duration_direct']]['Responder_loc']
        print('Duration for Responders found')
        

        Vector_df = pd.DataFrame(Vector_loc)
        Vector_df.rename(columns = {'coordinates':'Vector_loc'}, inplace = True)
        Vector_df['Patient_lon'] = Patient.loc[0, ('longitude')]
        Vector_df['Patient_lat'] = Patient.loc[0, ('latitude')]
        Vector_df['Patient_loc'] = list(zip(Vector_df['Patient_lon'],Vector_df['Patient_lat']))
        Vector_df['dist_patient'] = Vector_df.apply(lambda row: geopy.distance.distance(row['Vector_loc'], row['Patient_loc']).meters, axis=1)
        # only keep the 15 closest vectors. keep='all' so that more that all responders with the 5 lowest values are kept.
        # This is done to minimize the requests to the API (max = 2000 a day)
        subset_vector = Vector_df.nsmallest(15, 'dist_patient', keep='all')
        subset_vector['duration']=[self.directions([i, Patient_cood], profile = 'driving-car')['duration'] for i, 
                                          Patient_cood in zip(subset_vector['Vector_loc'], subset_vector['Patient_loc'])]
        # reset the index of the subset to make indexing possible again
        subset_vector = subset_vector.reset_index(drop = True)
        # select the fastest overall time
        fastest_Vector = subset_vector.iloc[subset_vector.idxmin()['duration']]['duration']
        loc_Vector = subset_vector.iloc[subset_vector.idxmin()['duration']]['Vector_loc']
        loc_Patient = subset_vector.iloc[subset_vector.idxmin()['duration']]['Patient_loc']
        print('Duration for Vectors found')
        
        dict = {'Patient_loc':[loc_Patient], 
                'Responder_loc':[loc_Responder], 
                'duration_Responder':[fastest_Responder], 'AED_loc':'No_AED','duration_AED':'No_AED',
                'Vector_loc':[loc_Vector],
                'duration_Vector':[fastest_Vector]}
        df = pd.DataFrame(dict)
        
        return df

    # Function to simulate survival probability.
    # Used in fastest_comparisson to decide which responder should be send directly.
    def survival_probability(self, x, z):
        return 0.9 **(x/60)* 0.97**(z/60)
    
    # Function to get all possible routes through the AEDs that are close to the patient
    # Returns a data frame with the coordinates of the Responder, duration through the specific AED,
    # duration for the direct route, and the coordinates of the used AED
    def possible_routing_direct(self, Patient, Responders):
        Patient = pd.DataFrame(Patient)
        # transpose into a row with columns of coordinates
        Patient = Patient.transpose()
        # reset index to be able to get longitude with 0 for the longitude and latitude.
        Patient = Patient.reset_index(drop = True)
        Responder_df = pd.DataFrame(Responders)
        Responder_df.rename(columns = {'coordinates':'Responder_loc'}, inplace = True)
        Responder_df['Patient_lon'] = Patient.loc[0, ('longitude')]
        Responder_df['Patient_lat']  = Patient.loc[0, ('latitude')]
        Responder_df['Patient_loc'] = list(zip(Responder_df['Patient_lon'],Responder_df['Patient_lat']))
        # Uncomment to filter out the ones that have a far away distance as the crow flies
        # Responder_df['dist_patient'] = Responder_df.apply(lambda row: geopy.distance.distance(row['Responder_loc'], row['Patient_loc']).meters, axis=1)
        # only keep the 15 closest responders. keep='all' so that more that all responders with the 15 lowest values are kept.
        # Responder_df = Responder_df.nsmallest(15, 'dist_patient')#, keep='all'
        # Responder_df = Responder_df.reset_index(drop=True)
        Responder_df['duration_direct']=[self.directions([i, Patient_cood], profile = 'foot-walking')['duration'] for i, 
                                          Patient_cood in zip(Responder_df['Responder_loc'], Responder_df['Patient_loc'])]

        return Responder_df
    
    def possible_routing_indirect(self, Patient, Responder_loc, AED_loc,  opening_hours, distribution = {(6.0, 22.0): 0.5,(0.0, 24.0): 0.05,(8.0, 20.0): 0.3,(9.0, 18.0):0.15}):
        Patient = pd.DataFrame(Patient)
        Patient = Patient.transpose()
        Patient = Patient.reset_index(drop = True)
        Responder_df = pd.DataFrame(Responder_loc)
        Responder_df.rename(columns = {'coordinates':'Responder_loc'}, inplace = True)
        Responder_df['Patient_lon'] = Patient.loc[0, ('longitude')]
        Responder_df['Patient_lat']  = Patient.loc[0, ('latitude')]
        Responder_df['Patient_loc'] = list(zip(Responder_df['Patient_lon'],Responder_df['Patient_lat']))

        AED_df = pd.DataFrame(AED_loc)
        AED_df.rename(columns = {'coordinates':'AED'}, inplace = True)
        # get the coordinates of the patient
        AED_df['Patient_lon'] = Patient.loc[0, ('longitude')]
        AED_df['Patient_lat'] = Patient.loc[0, ('latitude')]
        AED_df['Patient_loc'] = list(zip(AED_df['Patient_lon'],AED_df['Patient_lat']))


        # Create data frames with the coordinates
        # If opening hours hard coded use this!!!!
        # AED_ISO = self.AED_Iso
        AED_ISO = self.assign_opening_hours(distribution)
        AED_ISO['AED'] = list(zip(AED_ISO["lon"], AED_ISO["lat"]))
        AED_df = AED_ISO.merge(AED_df, on='AED')

        # Delete all AEDs that are not open yet
        #AED_df['open'] =  AED_df.apply(lambda row: row['opening_hour'] < opening_hours and row['closing_hour'] > opening_hours))
        AED_df.loc[:, 'open'] = AED_df.apply(lambda row: 1 if (row['opening_hour'] < opening_hours and row['closing_hour'] > opening_hours) else 0, axis=1)
        AED_df = AED_df[AED_df['open'] > 0]


        # Convert AED_df to a GeoDataFrame with the correct CRS
        print(AED_df.head())
        AED_df = gpd.GeoDataFrame(AED_df, geometry='geometry', crs="EPSG:4326")

        all_points = []  # Store points and midpoints as tuples

        # Iterate through each row in AED_df to use the 'Iso' polygons
        for _, row in AED_df.iterrows():
            polygon = row['geometry']
            midpoint = row['AED']
            
            points_in_polygon = self.closest_location_AED(Responder_df, polygon)

            # Add the points along with the midpoint of the polygon
            all_points.extend([(point, midpoint) for point in points_in_polygon])

        # Create a DataFrame with the points and their associated polygon midpoints
        result_df = pd.DataFrame(all_points, columns=['Responder_loc', 'AED_coordinates'])
        result_df = result_df.drop_duplicates()

        # get the coordinates of the patient
        result_df['Patient_lon'] = Patient.loc[0, ('longitude')]
        result_df['Patient_lat'] = Patient.loc[0, ('latitude')]
        result_df['Patient_loc'] = list(zip(result_df['Patient_lon'],result_df['Patient_lat']))
        

        result_df['duration_through_AED']=[self.directions([result_df['Responder_loc'][i], result_df['AED_coordinates'][i],result_df['Patient_loc'][i]])['duration'] for i in range(len(result_df['Responder_loc']))]
        return result_df
    
    # Function to find the responder that is send directly and through the AED.
    # The results are plotted using plotly
    # Patient = location of patient
    # Responder = Dataframe of responders and their location
    # AED = list of AEDs
    # N_responders = total amount of responders that will be contacted
    # AED_rate = proportion of N_responders that is requethrough AED
    # decline_rate = proportion of people excpected to decline the call to action
    def send_responders(self, Patient, Responders, AEDs, N_responders, decline_rate, opening_hours, distribution):
        df_duration_direct = self.possible_routing_direct(Patient, Responders)
        df_duration_indirect = self.possible_routing_indirect(Patient, Responders, AEDs,  opening_hours, distribution)

        df_duration_direct = df_duration_direct.sort_values(by=['duration_direct'], ascending=True)
        df_duration_direct = df_duration_direct.nsmallest(round((N_responders)), 'duration_direct')
        # Only keep the fastest route through AED for every responder
        df_duration_indirect.sort_values(by=['duration_through_AED'], ascending=True).drop_duplicates('Responder_loc').sort_index()
        df_duration_indirect = df_duration_indirect.nsmallest(round((N_responders)), 'duration_through_AED')

        # create list of all possible responders
        possible_responder = list(df_duration_direct['Responder_loc']) + list(df_duration_indirect['Responder_loc'])
        # Only take individual Responders
        seen = set()
        possible_responder = [val for val in possible_responder if val not in seen and (seen.add(val) or True)]
        # Randomly generate numbers with a decline_rate. 
        # Data frame with all the possible responders and their probability to accept the call. 1 for acceptance
        acceptance = list(np.random.choice([0,1], p=[decline_rate, (1-decline_rate)], size = len(possible_responder)))
        df_acceptance = pd.DataFrame({'Responder_loc' : possible_responder,
                                'Probability' : acceptance})
        
        # merge the acceptance and the data
        # Filter out the ones that did not exept
        df_duration_direct = df_duration_direct.merge(df_acceptance, on='Responder_loc')
        df_duration_direct = df_duration_direct[df_duration_direct['Probability'] > 0]
        df_duration_direct = df_duration_direct.reset_index(drop=True)
        df_duration_indirect = df_duration_indirect.merge(df_acceptance, on='Responder_loc')
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
                duration_direct = second_fastest_gpr['duration_direct']
                coord_AED =  fastest_aed['Responder_loc']
                AED_coordinates = fastest_aed['AED_coordinates']
                duration_indirect = fastest_aed['duration_through_AED']
            # If this is not true:
            # - Fastes direct responder will be send directly
            # - Second fastest through AED responder will be send through the AED
            else:
                coord_direct = fastest_gpr['Responder_loc']
                duration_direct = fastest_gpr['duration_direct']
                coord_AED = second_fastest_aed['Responder_loc']
                AED_coordinates = second_fastest_aed['AED_coordinates']
                duration_indirect = second_fastest_aed['duration_through_AED']
        else:
            # If the fastest direct responder and thorugh AED responder are different:
            # - Take the fastest responders for both
            coord_direct = fastest_gpr['Responder_loc']
            duration_direct = fastest_gpr['duration_direct']
            coord_AED = fastest_aed['Responder_loc']
            AED_coordinates = fastest_aed['AED_coordinates']
            duration_indirect = fastest_aed['duration_through_AED']
            
        return {'coord_direct': coord_direct, 'duration_direct':duration_direct,'coord_AED': coord_AED, 'AED_coordinates': AED_coordinates, 'duration_through_AED':duration_indirect}

    # Function to build a data frame with the fastest direct, indirect and vector duration.
    def fastest_comparisson(self, Patient, Vector_loc, Responder_loc, AED_loc, N_responders, decline_rate, opening_hour, distribution):
        """
        Function calculating the time there are no AEDs close by.
        Used in function: def fastest_time
        Uses function: def send_responder, def directions
        
        Parameters:
        Patient (pd.Series): Patients coordinates in a series with name "longitude", "latitude". !!!CASE SENSITIVE!!!
        Vector_loc (pd.Dataframe): Dataframe with the coordinates named "coordinates" !!!CASE SENSITIVE!!!. Normally, that is the output of def closest_location.
        Responder_loc (pd.Dataframe): Dataframe with the coordinates named "coordinates" !!!CASE SENSITIVE!!!. Normally, that is the output of def closest_location.
        AED_loc (pd.Dataframe): Dataframe with the coordinates named "coordinates" !!!CASE SENSITIVE!!!. Normally, that is the output of def closest_location.
        N_responder: 
        decline_rate (float): !!!Must be between 0 and 1!!! Represents the percentage of responders declining the call to action.
        opening_hour (float): Time of the day. Checks of the AED is open during that time. 
        distribution (dictionary): Dictionary where keys are tuples of (opening_hour, closing_hour) and values are proportions. !!!Distribution proportions must sum to 1!!!
                
        Returns:
        pd.DataFrame: Dataframe with 
                {'Patient_loc':Patient Location,
            	'Responder_loc':Responder Location, 
                'duration_Responder':Duration it takes the direct responder to get to the patient, 
                'AED_loc':AED Location,
                'duration_AED':Duration it takes the indirect responder to get to the patient, 
                'Vector_loc':Vector Location,
                'duration_Vector':Duration it takes the vector to get to the patient}
        """
        responders_send = self.send_responders(Patient,  Responder_loc, AED_loc, N_responders, decline_rate, opening_hour, distribution)
        loc_Responder = responders_send['coord_direct']
        fastest_Responder = responders_send['duration_direct']
        loc_AED = responders_send['coord_AED']
        fastest_AED = responders_send['duration_through_AED']
        
        Vector_df = pd.DataFrame(Vector_loc)
        Vector_df.rename(columns = {'coordinates':'Vector_loc'}, inplace = True)
        Vector_df['Patient_lon'] = Patient['longitude']
        Vector_df['Patient_lat'] = Patient['latitude']
        Vector_df['Patient_loc'] = list(zip(Vector_df['Patient_lon'],Vector_df['Patient_lat']))
        Vector_df['dist_patient'] = Vector_df.apply(lambda row: geopy.distance.distance(row['Vector_loc'], row['Patient_loc']).meters, axis=1)
        # only keep the 5 closest vectors. keep='all' so that more that all responders with the 5 lowest values are kept.
        # This is done to minimize the requests to the API (max = 2000 a day) 
        subset_vector = Vector_df.nsmallest(5, 'dist_patient', keep='all')
        subset_vector['duration']=[self.directions([i, Patient_cood], profile = 'driving-car')['duration'] for i, 
                                          Patient_cood in zip(subset_vector['Vector_loc'], subset_vector['Patient_loc'])]
        # reset the index of the subset to make indexing possible again
        subset_vector = subset_vector.reset_index(drop = True)
        # select the fastest overall time
        fastest_Vector = subset_vector.iloc[subset_vector.idxmin()['duration']]['duration']
        loc_Vector = subset_vector.iloc[subset_vector.idxmin()['duration']]['Vector_loc']
        loc_Patient = subset_vector.iloc[subset_vector.idxmin()['duration']]['Patient_loc']
        print('Duration for Vectors found')
        
        dict = {'Patient_loc':[loc_Patient], 
                'Responder_loc':[loc_Responder], 
                'duration_Responder':[fastest_Responder], 'AED_loc':[loc_AED],'duration_AED':[fastest_AED],
                'Vector_loc':[loc_Vector],
                'duration_Vector':[fastest_Vector]}
        df = pd.DataFrame(dict)
        
        return df