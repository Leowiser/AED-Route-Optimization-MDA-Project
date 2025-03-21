# MDA_2024_Zambia
This is the repository of the Zambia group of the course Modern Data Analytics at the KU Leuven. The goal of the Project was to develop an App that optimizes and visualizes routes of community first responders in the city of Leuven, to maximize survival chances in the case of a cardiac arrest.

The repository includes all files needed to create and run the App. It also includes all files to reproduce the report.

**!!! The App can only be used once per minute !!!**

[Click here to acces the App](https://aed-route-optimization-mda-project.onrender.com/)

## Preprocessing_Class.py
Cleans the given AED data in Leuven and finds their coordinates.

## FR_Genration_Class.py
This class generates first responders in Leuven using two datasets: a geodataset of the statistical sectors of Leuven (Data/Leuven.shp) and a dataset containing the population information for these statistical sectors (Data/leuven.csv). The number of first responders is generated as a share of the total population. These first responders are then spread among the statistical sectors of Leuven according the to sectors' population density. Areas with higher population density will receive have a higher density of first responders.

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

### data_interventions
Cleans the interventions data to only include the cardiac arrests in Leuven on which we want to conduct the simulation.

### Simulations_Survivability
**!! Must be run over serveral days to not strain the API to much !!**

Uses the Simulation_Class.py simulation class to simulate the time that was needed for first responders to reach all observed 81 cardiac arrest from the year 2022-2023. The simulations are conducted with 0.5% and 1% of the population as responders.

### sim_plot.ipynb
Utilizes the Survival_prob_Class.py probability_survival function to calculate the survival of the simulated patients from Simulations_Survivability. 

------

# Extra Folders
## Test_Classes
Folder with tests for the route and survival probability class.

## Data
Folder including all the data needed and produced to run the app.
