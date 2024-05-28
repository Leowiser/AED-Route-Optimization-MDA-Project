import openrouteservice
from openrouteservice import client
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import geometry
import time
import geopy.distance



class simulation:
    def __init__(self):
        self.Client_ors = openrouteservice.Client(key='5b3ce3597851110001cf624802e069d6633748a5ae4e9842334f1dc2')

    # function to find the AEDs and Responders in a 10 minute walking distance from the patient
    # Nearly same as closest Responders
    def closest_location(self, Patient, Location, profile = "foot-walking", threshold =600):
        # patient must be a tuple
        # AEDS must be a dataframe with columns (This is gathered in another file) named latitude and longitude
        # profile by default is walking by foot.
        # Up to 5 AEDs can be used as locations.

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
    

    # Function that gets the duration, route and coordinates of a route
    # The route can be direct or go through other points first
    def directions(self, coordinates, profile = 'foot-walking', sleep = 1.52):
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
    
    # Function to get all possible routes through the AEDs that are close to the patient
    # Returns a data frame with the coordinates of the Responder, duration through the specific AED,
    # duration for the direct route, and the coordinates of the used AED
    def fastest_time(self, Patient, Responders, AEDs, Vectors, Dist_responder = 600, Dist_AED = 600, Dist_Vector = 600, threshold = 700):
        # Time that the isochrones covers in seconds
        Responders_loc = self.closest_location(Patient, Responders,threshold=Dist_responder)
        # Time that the isochrones covers in seconds
        AED_loc = self.closest_location(Patient, AEDs, threshold=Dist_AED)
        # Time that the isochrones covers in seconds
        Vector_loc = self.closest_location(Patient, Vectors, profile = 'driving-car', threshold = Dist_Vector)

        # Set empty dictionary to be fillled later
        df_duration = pd.DataFrame(columns = ['Patient_loc', 'Responder_lon', 'Responder_lat', 
                'duration_Responder', 'AED_lon', 'AED_lat','duration_AED','Vector_lon', 'Vector_lat',
                'duration_Vector'])
        
        # Check if any vector is reached in 10 minutes. 
        # If this is not the case the isochrone radius is increased for 2 minutes until at least one vector is found. 
        Dist_Vector = 600
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
        
        # If the Responder is still 0 only the Vectors time will be calculated
        if ((len(Responders_loc) == 0) and (len(Vector_loc) > 0)):
            print('No responder is in a 10 minute radius')
            df_duration = self.fastest_vector(Patient, Vector_loc)
        # Otherwise the two have to be compared.
        elif ((len(Responders_loc) > 0) and (len(Vector_loc) > 0)):
            # If no AED is close by, the time of the direct responder is returned 
            if((len(AED_loc) == 0) or (len(Responders_loc)<2)):
                print('Comparing direct responder vs. vectors')
                df_duration = self.direct_vs_vector(Patient, Vector_loc, Responders_loc)
            # Else it is compared whether the fastest or second fastest responder should be send through the AED based on the surviveability value.
            else:
                print('Comparing resopnders vs. vectors')
                df_duration = self.fastest_comparisson(Patient, Vector_loc, Responders_loc, AED_loc)
        # Returns a data frame.          
        return df_duration

    # Function when only the duration for the vector is needed.
    def fastest_vector(self, Patient, Vector_loc):
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
        # Only keep the 5 closest vectors. keep='all' so that all responders with the 5 lowest values are kept.
        # This is done to minimize the requests to the API (max = 2000 a day)
        subset_vector = Vector_df.nsmallest(5, 'dist_patient', keep='all')
        subset_vector['duration']=[self.directions([i, Patient_cood], profile = 'driving-car')['duration'] for i, 
                                          Patient_cood in zip(subset_vector['Vector_loc'], subset_vector['Patient_loc'])]
        # reset the index of the subset to make indexing possible again
        subset_vector = subset_vector.reset_index(drop = True)
        # select the fastest overall time
        fastest_Vector = subset_vector.iloc[subset_vector.idxmin()['duration']]['duration']
        lat_Vector = subset_vector.iloc[subset_vector.idxmin()['duration']]['Vector_loc'][1]
        lon_Vector =subset_vector.iloc[subset_vector.idxmin()['duration']]['Vector_loc'][0]
        loc_Patient = subset_vector.iloc[subset_vector.idxmin()['duration']]['Patient_loc']
        print('Duration for Vectors found')
        
        dict = {'Patient_loc':[loc_Patient], 
                'Responder_lon':'No responder', 'Responder_lat':'No responder', 
                'duration_Responder':'No responder', 'AED_lon':'No AED', 
                'AED_lat':'No AED','duration_AED':'No AED',
                'Vector_lon':lon_Vector, 'Vector_lat':lat_Vector,
                'duration_Vector':fastest_Vector}
        df_vector = pd.DataFrame(dict)
        return df_vector

    # Function if direct responders and vectors duration are calculated
    def direct_vs_vector(self, Patient, Vector_loc, Responder_loc):
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
        Responder_df['dist_patient'] = Responder_df.apply(lambda row: geopy.distance.distance(row['Responder_loc'], row['Patient_loc']).meters, axis=1)
        # only keep the 5 closest responders. keep='all' so  that all responders with the 5 lowest values are kept.
        # This is done to minimize the requests to the API (max = 2000 a day)  
        subset_responder = Responder_df.nsmallest(5, 'dist_patient', keep='all')
        subset_responder['duration_direct']=[self.directions([i, Patient_cood], profile = 'foot-walking')['duration'] for i,
                                             Patient_cood in zip(subset_responder['Responder_loc'], subset_responder['Patient_loc'])]
        # reset the index of the subset to make indexing possible again
        subset_responder = subset_responder.reset_index(drop = True)
        # select the fastest overall time
        fastest_Responder = subset_responder.iloc[subset_responder.idxmin()['duration_direct']]['duration_direct']
        lat_Responder = subset_responder.iloc[subset_responder.idxmin()['duration_direct']]['Responder_loc'][1]
        lon_Responder = subset_responder.iloc[subset_responder.idxmin()['duration_direct']]['Responder_loc'][0]
        print('Duration for Responders found')
        

        Vector_df = pd.DataFrame(Vector_loc)
        Vector_df.rename(columns = {'coordinates':'Vector_loc'}, inplace = True)
        Vector_df['Patient_lon'] = Patient.loc[0, ('longitude')]
        Vector_df['Patient_lat'] = Patient.loc[0, ('latitude')]
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
        lat_Vector = subset_vector.iloc[subset_vector.idxmin()['duration']]['Vector_loc'][1]
        lon_Vector =subset_vector.iloc[subset_vector.idxmin()['duration']]['Vector_loc'][0]
        loc_Patient = subset_vector.iloc[subset_vector.idxmin()['duration']]['Patient_loc']
        print('Duration for Vectors found')
        
        dict = {'Patient_loc':[loc_Patient], 
                'Responder_lon':lon_Responder, 'Responder_lat':lat_Responder, 
                'duration_Responder':fastest_Responder, 'AED_lon':'No AED', 
                'AED_lat':'No AED','duration_AED':'No AED',
                'Vector_lon':lon_Vector, 'Vector_lat':lat_Vector,
                'duration_Vector':fastest_Vector}
        df = pd.DataFrame(dict)
        
        return df

    # Function to simulate survival probability.
    # Used in fastest_comparisson to decide which responder should be send directly.
    def survival_probability(self, x, z):
        return 0.9 **(x/60)* 0.97**(z/60)
    
    # Function to build a data frame with the fastest direct, indirect and vector duration.
    def fastest_comparisson(self, Patient, Vector_loc, Responder_loc, AED_loc):
        Patient = pd.DataFrame(Patient)
        Patient = Patient.transpose()
        Patient = Patient.reset_index(drop = True)
        Responder_df = pd.DataFrame(Responder_loc)
        Responder_df.rename(columns = {'coordinates':'Responder_loc'}, inplace = True)
        Responder_df['Patient_lon'] = Patient.loc[0, ('longitude')]
        Responder_df['Patient_lat']  = Patient.loc[0, ('latitude')]
        Responder_df['Patient_loc'] = list(zip(Responder_df['Patient_lon'],Responder_df['Patient_lat']))
        Responder_df['dist_patient'] = Responder_df.apply(lambda row: geopy.distance.distance(row['Responder_loc'], row['Patient_loc']).meters, axis=1)
        # only keep the 5 closest responders. keep='all' so that more that all responders with the 5 lowest values are kept.        
        # This is done to minimize the requests to the API (max = 2000 a day)        
        subset_responder = Responder_df.nsmallest(5, 'dist_patient', keep='all')
        subset_responder['duration_direct']=[self.directions([i, Patient_cood])['duration'] for i,
                                             Patient_cood in zip(subset_responder['Responder_loc'], subset_responder['Patient_loc'])]
        # select the fastest overall time
        # reset the index of the subset to make indexing possible again
        subset_responder = subset_responder.reset_index(drop = True)
        print('Duration for responders found')

        AED_df = pd.DataFrame(AED_loc)
        AED_df.rename(columns = {'coordinates':'AED_coordinates'}, inplace = True)
        df_merged = pd.merge(subset_responder.assign(key=1), AED_df.assign(key=1),
                        on='key').drop('key', axis=1)
        df_merged['dist_AED'] = df_merged.apply(lambda row: geopy.distance.distance(row['Responder_loc'], row['AED_coordinates']).meters, axis=1)
        
        df_merged['duration_through_AED']=[self.directions([df_merged['Responder_loc'][i], df_merged['AED_coordinates'][i],
                                                            df_merged['Patient_loc'][i]])['duration']  if df_merged['dist_AED'][i] < 700 else 5000 for i in range(len(df_merged['dist_AED']))]
        print('Duration for AEDs found')

         # coordinate of the Responder with the fastest direct time and the one with the fastest time through an AED
        coord_direct = df_merged.iloc[df_merged.idxmin()['duration_direct']]['Responder_loc']
        coord_AED = df_merged.iloc[df_merged.idxmin()['duration_through_AED']]['Responder_loc']
        # coordinates of the AED with the second fastest route
        subset = df_merged[(df_merged['duration_direct']>df_merged.min()['duration_direct'])]
        # Reset the index for later indexing to work.
        subset = subset.reset_index(drop=True)
        
        # Starting decision rule to decide who collects the AED
        # Parameters for caculating the survival probability
        # t_a = Total time for fastest responder to arrive with AED
        # t_b = Total time for second fastest responder to arrive with AED
        # t_a_no_aed = Time for fastest responder to start CPR without AED
        t_a = df_merged[df_merged['duration_direct']==df_merged.min()['duration_direct']].min()['duration_through_AED'] 
        t_b = subset.iloc[subset.idxmin()['duration_through_AED']]['duration_through_AED']  
        t_a_no_aed = df_merged.min()['duration_direct']  
        # t_b_CPR = Time of CPR until second fastest responder arrives
        t_b_CPR = (t_b-t_a_no_aed)

        # Calculate survival probabilities with function survival_probability
        # - 0.9 ** (x/60) * 0.97 ** (z/60)
        # - x is the time without CPR, z is the time with CPR in minutes

        # Fastest responder going for the AED
        # - z equals zero as no one does any CPR
        surv_A = self.survival_probability(t_a, 0)
        # Second fastest responder arriving with AED
        # - time until AED arrives minus time CPR arrives is the time without CPR
        surv_B = self.survival_probability(t_b_CPR, t_a_no_aed)

        # Now check if the fastest through AED is the same as the fastest direct 
        if coord_direct == coord_AED:
            print('Fastest Direct and Indirect are the same')
            # Find best strategy which is the maximal survival chances
            best_strategy = max(surv_A, surv_B)
            if best_strategy == surv_A:
                # - Second fastest direct time will be send directly  
                # - Fastest direct and AED responder will be send through the AED
                coord_direct = (df_merged.iloc[df_merged.drop_duplicates(subset=['Responder_loc']).nsmallest(2,'duration_direct').index[1]]['Responder_loc'])
                coord_AED =  df_merged.iloc[df_merged.idxmin()['duration_direct']]['Responder_loc']
                AED_coordinates = df_merged.iloc[df_merged.idxmin()['duration_direct']]['AED_coordinates']
                fastest_Responder = df_merged.iloc[df_merged.drop_duplicates(subset=['Responder_loc']).nsmallest(2,'duration_direct').index[1]]['duration_direct']
                fastest_AED = df_merged[df_merged['duration_direct']==df_merged.min()['duration_direct']].min()['duration_through_AED']

            else:
                # - Fastes direct responder will be send directly
                # - Second fastest through AED responder will be send through the AED                
                coord_direct = coord_direct
                coord_AED = subset.iloc[subset.idxmin()['duration_through_AED']]['Responder_loc']
                AED_coordinates = subset.iloc[subset.idxmin()['duration_through_AED']]['AED_coordinates']
                fastest_Responder = df_merged.iloc[df_merged.idxmin()['duration_direct']]['duration_direct']
                fastest_AED = subset.iloc[subset.idxmin()['duration_through_AED']]['duration_through_AED']     
        else:
            print('Fastest Direct and Indirect are not the same')
            # If the fastest direct responder and thorugh AED responder are different:
            # - Take the fastest responders for both
            coord_direct = coord_direct
            coord_AED = coord_AED
            fastest_Responder = df_merged.iloc[df_merged.idxmin()['duration_direct']]['duration_direct']
            fastest_AED = df_merged.iloc[df_merged.idxmin()['duration_through_AED']]['duration_through_AED']        
    
        lat_Responder = coord_direct[1]
        lon_Responder = coord_direct[0]
        lat_AED = coord_AED[1]
        lon_AED = coord_AED[0]
        
        Vector_df = pd.DataFrame(Vector_loc)
        Vector_df.rename(columns = {'coordinates':'Vector_loc'}, inplace = True)
        Vector_df['Patient_lon'] = Patient.loc[0, ('longitude')]
        Vector_df['Patient_lat'] = Patient.loc[0, ('latitude')]
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
        lat_Vector = subset_vector.iloc[subset_vector.idxmin()['duration']]['Vector_loc'][1]
        lon_Vector =subset_vector.iloc[subset_vector.idxmin()['duration']]['Vector_loc'][0]
        loc_Patient = subset_vector.iloc[subset_vector.idxmin()['duration']]['Patient_loc']
        print('Duration for Vectors found')
        
        dict = {'Patient_loc':[loc_Patient], 
                'Responder_lon':lon_Responder, 'Responder_lat':lat_Responder, 
                'duration_Responder':fastest_Responder, 'AED_lon':lon_AED, 
                'AED_lat':lat_AED,'duration_AED':fastest_AED,
                'Vector_lon':lon_Vector, 'Vector_lat':lat_Vector,
                'duration_Vector':fastest_Vector}
        df = pd.DataFrame(dict)
        
        return df
