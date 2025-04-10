import openrouteservice
from openrouteservice import client
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import geometry
import time
import geopy.distance
import random
from Simulation_Routing import RoutingSimulation
from Simulation_Routing_Matrix import RoutingSimulationMatrix
from Simulation_Routing_Matrix_copy import RoutingSimulationMatrixSec
from Simulation_Routing_Matrix_Batch import RoutingSimulationMatrixBatch
from tqdm import tqdm

# Exception class
#------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------

class NotAcceptableDeclineRate(Exception):
    """Raised when no responder accepts the request."""
    def __init__(self, distribution, message="Distribution must sum to 1"):
        self.distribution = distribution
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.distribution} -> {self.message}'


class NoAEDResponderAcceptedError(Exception):
    """Raised when no responder through AED accepts the request."""
    pass

class NotAcceptableInput(Exception):
    '''
    Raised when distribution in opening hours does not add to 1
    '''

    def _number_input(self, input_type, input_value, lower_limit, upper_limit):
        return f'Given {input_type} is: {input_value} -> But value should be between {lower_limit} and {upper_limit}'

    def _str_input(self, input_value, possible_list):
        return f'Given input is: {input_value} -> But value should be one of the following or a combination of these {possible_list}'


#--------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------



class Simulation:
    def __init__(self, ip):
        self.IP = ip
        self.CLIENT_ORS = openrouteservice.Client(base_url=f'http://{self.IP}:8080/ors')
        self.AEDs = pd.read_csv("Data/filtered_AED_loc.csv")
        self.AMBULANCES = pd.read_parquet("Data/ambulance_locations.parquet.gzip")
        self.VECTORS = self.__clean_vector()
        self.STAT_SEC_GEOMETRIES = gpd.read_file("C:/Users/leonw/Downloads/first_responder_generation.gpkg")
        intervention = pd.read_excel("C:/Users/leonw/OneDrive - KU Leuven/Documents/GitHub/AED-Route-Optimization-MDA-Project/Data/interventions_new.xlsx")
        intervention.rename(columns = {'longitude_intervention':'longitude', 'latitude_intervention':'latitude'}, inplace = True)
        intervention['coordinates'] = list(zip(intervention['longitude'], intervention['latitude']))
        self.PATIENTS = intervention.copy()
        self.AED_ISO = gpd.read_file('C:/Users/leonw/OneDrive - KU Leuven/Documents/GitHub/AED-Route-Optimization-MDA-Project/Data/temp.gpkg', layer='AED_data')


    def __clean_vector(self):
        amb_loc = self.AMBULANCES
        amb_loc = amb_loc[amb_loc['province']=='Vlaams-Brabant']
        # Only keep rows that contain 30 as all postla code of Leuven start with 30
        amb_loc = amb_loc[amb_loc['departure_location'].str.contains('30')]
        amb_loc = amb_loc.reset_index(drop = True)
        df_ambulances = pd.DataFrame()
        df_ambulances['Name'] = amb_loc['departure_location_number']
        df_ambulances['longitude'] = amb_loc['longitude']
        df_ambulances['latitude'] = amb_loc['latitude']

        #Googled the only Pit in Leuven
        df_pit = pd.DataFrame()
        df_pit['Name'] = "PVLEUV01A"
        df_pit['longitude'] = 4.6716518
        df_pit['latitude'] = 50.8791702

        # Googled the latitude and longitude
        df_mug = pd.DataFrame()
        df_mug['Name'] = 322
        df_mug['longitude'] = 4.6690603
        df_mug['latitude'] = 50.8784361

        df_vectors = pd.concat([df_ambulances, df_pit, df_mug], ignore_index = True)
        df_vectors['coordinates'] =  list(zip(df_vectors.loc[:,'longitude'], df_vectors.loc[:,'latitude']))
        df_vectors = df_vectors[['longitude', 'latitude','coordinates']]
        return df_vectors

    def _generate_cfrs(self, time_of_day = "day", proportion = 0.01):
        stat_sec_geometries = self.STAT_SEC_GEOMETRIES
        if time_of_day == "day":
            cfr_counts = "total_daytime_CFRs"
        elif time_of_day == "night":
            cfr_counts = "total_nighttime_CFRs"
        else:
            raise ValueError("Invalid value for 'time_of_day'. Please choose either 'day' or 'night'.")
        sample = stat_sec_geometries.sample_points(size = (stat_sec_geometries[cfr_counts] * proportion).round().astype(int))
        sampled_points_gdf = gpd.GeoDataFrame(geometry=sample.explode(), crs=stat_sec_geometries.crs)
        sampled_points_gdf = sampled_points_gdf.to_crs(epsg=4326)
        sampled_points_gdf['latitude'] = sampled_points_gdf['geometry'].y
        sampled_points_gdf['longitude'] = sampled_points_gdf['geometry'].x
        return sampled_points_gdf

    
    def simulation_run(self, decline_rate, max_number_responder, opening_hour, filter_values, time_of_day, proportion, dist_responder = 600, dist_AED = 600, dist_Vector = 600):
        # If no responders are available, raise an exception
        exception_input = NotAcceptableInput()
        # Fix: Corrected condition for valid range
        if not (0 <= decline_rate <= 1):
            raise NotAcceptableInput(exception_input._number_input("decline_rate", decline_rate, 0, 1))
        
        if not (0 <= proportion <= 1):
            raise NotAcceptableInput(exception_input._number_input("proportion", proportion, 0, 1))

        if not (0 <= opening_hour <= 24):
            raise NotAcceptableInput(exception_input._number_input("opening hour",opening_hour, 0, 24))

        possible_list = ["Yes", "Private", "Company"]
        if filter_values is not None and not set(filter_values).issubset(possible_list):
            raise NotAcceptableInput(exception_input._str_input(filter_values, possible_list))
        
        df_final = pd.DataFrame(columns = ['patient_loc', 'responder_loc', 'duration_Responder',
                                           'Indirect_Responder_loc', 'aed_loc', 'duration_AED',
                                           'vector_loc', 'duration_Vector', 'prob_vec', 'prob_resp'])

        
        responders = self._generate_cfrs(time_of_day, proportion)
        responders = responders.reset_index(drop=True)
        ip = self.IP
        routing = RoutingSimulationMatrixBatch(ip)

        for _, patient in tqdm(self.PATIENTS.iterrows(), total=self.PATIENTS.shape[0]):
            try:
                df = routing.fastest_time(patient, responders, self.VECTORS, decline_rate, 
                                        max_number_responder, opening_hour, filter_values, dist_responder, dist_AED, dist_Vector)
                
                # Handle missing durations safely
                df['duration_AED'] = df['duration_AED'].replace('No AED', float(10000))
                df['duration_Responder'] = df['duration_Responder'].replace('No responder', float(10000))

                duration_responder = df['duration_Responder'].iloc[0] if 'duration_Responder' in df.columns else None
                duration_AED = df['duration_AED'].iloc[0] if 'duration_AED' in df.columns else None
                duration_vector = df['duration_Vector'].iloc[0] if 'duration_Vector' in df.columns else None

                prob_resp, prob_vec = self.__probability_survival(duration_responder, duration_AED, duration_vector)
                df['prob_vec'] = prob_vec
                df['prob_resp'] = prob_resp

                df_final = pd.concat([df_final, df])
                df_final = df_final.reset_index(drop=True)

            except Exception as e:
                # Skip this patient and log the error
                print(f"[Warning] Error with patient: {patient.get('id', 'unknown')}. Skipping. Error: {str(e)}")
                continue
        
        return df_final
    
        # only used to compare, so we do not need to check if vector or responder arrives first, just comapre their prop of survival
        # this is the probability of survival immedaiatly after cpr or aed chock, not the survival after 24 hours... 
        # info on survival rates from: https://www.sciencedirect.com/science/article/pii/S0019483219304080
        # and from here: https://jamanetwork.com/journals/jama/fullarticle/196200
        # and from here: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2600120/
    def __probability_survival(self, duration_responder, duration_AED, duration_vector, decrease_with_cpr = 0.97, decrease_no_cpr = 0.9):
        # explanation of parameters 
        # duration_responder: time it takes for responder without aed to arrive, integer
        # duration_AED: for aed to arrive, integer
        # duration_vector: for vector to arrive, integer
        # decrease_with_cpr: decrease in survival if cpr is perfromed  
        # decrease_no_cpr: decrease in survival if cpr is not performed 

        prob_resp = 1 # starting probability to survive with responder
        prob_vec = 1 # starting probability to survive with vector 

        # calcualte probability of survival with vector, probability decreases with decrease_no_cpr per minute 
        time_vec_min = duration_vector/60 
        prob_vec = prob_vec * decrease_no_cpr ** time_vec_min

        # also add vector to calcualtion since we never have only first responder 
        fastest_aed = min(duration_AED, duration_vector) # get the fastest of the aed and vector that can arrive with aed 

        if(fastest_aed <= duration_responder): # if AED/vector is faster or as fast as first responder, no cpr is started
            time_aed_min = fastest_aed/60 # time to aed arrives in min 
            prob_resp = prob_resp * decrease_no_cpr ** time_aed_min
        else: # if responder is faster than aed 
            # time without cpr in min
            time_no_cpr = duration_responder/60
            # time with cpr before aed arrives in min 
            time_with_cpr = (fastest_aed-duration_responder)/60
            # decrease in survival when cpr is started after a while   
            prob_resp = prob_resp * decrease_no_cpr ** time_no_cpr * decrease_with_cpr**time_with_cpr
        return prob_resp, prob_vec 

    
        
    def __survival(self, responders, decline_rate, max_number_responder, incident_time):
        df = self.simulation_run(responders, decline_rate, max_number_responder, incident_time)
        # find instances where duration_Responder is 'No responder' or duration_AED is "No AED" and replace them with 10000
        df['duration_Responder'] = df['duration_Responder'].replace('No responder', float(10000))
        df['duration_AED'] = df['duration_AED'].replace('No AED', float(10000))
        df_prob = pd.DataFrame(columns=['prob_resp', 'prob_vec'])
        for index, row in df.iterrows():
            prob_resp, prob_vec = self.probability_survival(float(row['duration_Responder']), float(row['duration_AED']), row['duration_Vector'])
            # add calcualted probabilities to df 
            df_prob.loc[index] = {'prob_resp': prob_resp, 'prob_vec': prob_vec}

        return df_prob
    
    def average_surv_prob(self, responders, decline_rate, max_number_responder, incident_time, number_of_sim = 10):
        i= 0
        while i < number_of_sim:
            df_prob = self.survival(responders, decline_rate, max_number_responder, incident_time)
            df_avg = df_prob.add(df_avg * (i-1))/i
            df_avg = df_avg.reset_index(drop = True)
            i =+ 1
        
        return df_avg


    def plot_simulation(self, simulation_result):
            # color palette and liine style
            COLOR_SCALE = ["#feb24c", "#000000", "#7fcdbb", "#2c7fb8"]
            LINE_STYLE = [":","-","--","-."]
            # Define margin around annotation
            PAD = 3
            y_offset = 0.01  # Adjust based on your data range

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            # Hide default legend
            ax.legend().remove()
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            # Data series
            df_list = [simulation_result]
            columns = ["prob_resp", "prob_vec"]
            labels = ["Responder", "Vector Real"]

            # Plot each series
            for idx, (df, column, label) in tqdm(enumerate(zip(df_list, columns, labels)), total=len(df_list)):
                color = COLOR_SCALE[idx]
                
                # Plot line with markers
                ax.plot(df.index, df[column], color=color, marker="o", markersize=2.5, lw=1.2, label=label)

                # Add annotation
                y_end_value = df[column].iloc[-1]
                ax.text(
                    df.index[-1] + PAD, y_end_value + y_offset, label,
                    color=color, fontsize=12, va="center"
                )

                # Add arrow from line to annotation
                ax.arrow(
                    df.index[-1], y_end_value,
                    PAD, 0, clip_on=False, color="gray"
                )

            # Labels and title
            ax.set_xlabel("Patient Index")
            ax.set_ylabel("Survival Probability [%]")
            ax.set_title("Survival Probability")
