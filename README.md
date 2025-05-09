# Citizen First Responder System - From Bystander to Hero
This is the repository of the Citizen First Responder (CFR) System for Leuven project. The goal of the Project was to simulate the effect of a CFR system on the survival chances in the case of an out of hospital cardiac arrest.

The repository includes all files needed to create and run the simulations. It also includes all files to reproduce the report.



----------------------------
## Important preprocessing steps
### Preprocessing_Class.py
Cleans the given AED data in Leuven and finds their coordinates.

### Data_Isochones_AED.py
Cleans the AED data and adds a 10 minute walking isochrone around the AED location for a quicker calculation of the optimal responders during the simulations.
Produces Data/temp.gpkg

### leuven_CFR_generation.ipynb
Defines the distribution of day and nighttime population of Leuven. Also gives the sectors to generate the CFRs from. (Data/first_responder_generation.gpkg)


## Simulation of Survival Chances

### Simulation_Routing_Matrix_Batch.py
Class RoutingSimulationMatrixBatch to simulate and optimize responder dispatch and compares their response time to the quickest vector. 

### Simulating.py
Class that uses RoutingSimulationMatrixBatch and iterates through every patient of a given dataset to give back the optimal CFRs and the estimated survival rate of the patients. 
Uses Data/first_responder_generation.gpkg to generate different CFR distributions

### data_interventions
Cleans the interventions data to only include the cardiac arrests in Leuven on which we want to conduct the simulation.

### sim_script.py
Is the notebook to run the simulations.

-----------

## Visualization Folder
### sim_plot.ipynb
Utilizes the Survival_prob_Class.py probability_survival function to calculate the survival of the simulated patients from Simulations_Survivability. 

------

## Extra Folders
### Test_Classes
Folder with tests for the route and survival probability class.

### Data
Folder including all the data needed and produced to run the app.

### Obsolete
Includes all outdated versions that are no longer used
\\


------
## App folder

The folder **App** includes an app that could be a first visualization of an implementation of the system, however:
**!!! The App can only be used once per minute !!!**

[Click here to acces the App](https://aed-route-optimization-mda-project.onrender.com/)
*(No longer running)*

### FR_Genration_Class.py
This class generates first responders in Leuven using two datasets: a geodataset of the statistical sectors of Leuven (Data/Leuven.shp) and a dataset containing the population information for these statistical sectors (Data/leuven.csv). The number of first responders is generated as a share of the total population. These first responders are then spread among the statistical sectors of Leuven according the to sectors' population density. Areas with higher population density will receive have a higher density of first responders.

The class is used in the app.py as well as to create the data for the simulations of the effect of our App.

### Routing_Class.py
Class that finds the optimal first responders that are send to the patient directly and through the AED. These optimal responders are found by maximizing the survival chances of the patient.

### app.py
This is the resulting app to visualize the best routes of the direct and indirect first responders. It uses the Routing_class as well as the FR_Generation_Class and enables the user to take any location inside Leuven as well as changing the proportion of first responders inside the city to see optimal routing in different scenarios.

------
