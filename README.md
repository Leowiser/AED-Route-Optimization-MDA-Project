# MDA_2024_Zambia
This is the repository of the Zambia group of the course Modern Data Analytics at the KU Leuven. The goal of the Project was to develop an App that optimizes and visualizes routes of community first responders in the city of Leuven, to maximize survival chances in the case of a cardiac arrest.

The repository includes all files neede to create and run the App.

It also includes all files to reproduce the report.

## Preprocessing_Class.py
Cleans the given AED data in Leuven and finds their coordinates.

## FR_Genration_Class.py
This class generates the first responders in Leuven (Data/Leuven.shp) based on the population density of different Leuven area Data/leuven.csv. The first responders are genrated as a share of the total population.

The class is used in the app.py as well as to create the data for the simulations of the effect of our App.

## Routing_Class.py
Class that finds the optimal first responders that are send to the patient directly and through the AED. These optimal responders are found by maximizing the survival chances of the patient.

## app.py
This is the resulting app to visualize the best routes of the direct and indirect first responders. It uses the Routing_class as well as the FR_Generation_Class and enables the user to take any location inside Leuven as well as changing the proportion of first responders inside the city to see optimal routing in different scenarios.

------

# Simulation of Survival Chances

## Simulation_Class.py
Class to simulate responder dispatch and compares their response time to the one of the vectors. 

## Survival_prob_Class_py
Class to find the survival chances of a patient based on the arrival of the responders and vectors. 

### Simulations_Survivability
Uses the Simulation_Class.py simulation class to simulate the time that was needed for first responders to reach all observed 81 cardiac arrest from the year 2022-2023. The simulations are conducted with 0.5% and 1% of the population as responders.

### sim_plot.ipynb
Utilizes the Survival_prob_Class.py probability_survival function to calculate the survival of the simulated patients from Simulations_Survivability. 

------

# Extra Folders
## Test_Classes
Folder with tests for the route and survival probability class.

## Data
Folder including all the data needed and produced to run the app.
