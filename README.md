# MDA_2024_Zambia

## FR_Gneration_Class.py
This class generates the first responders in Leuven (Data/Leuven.shp) based on the population density of different Leuven area Data/leuven.csv. The first responders are gnerated as a share of the total population.

## Preprocessing_Class.py
Cleans the given AED data in Leuven and finds their coordinates.

## Routing_Class.py
Class that finds the optimal first responder that is send to the patient directly and through the AED. These optimal responders are found by maximizing the survival chances of the patient.

## Simulation_Class.py
Class to simulate responder dispatch and their comparisson to vectors. 

