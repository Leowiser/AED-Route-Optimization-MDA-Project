
#required packages
import pandas as pd
import geopandas as gpd
from pyproj import CRS

class FR_Generation:

    def __init__(self):
        
    def load_data():
        # read in the geospatial data for the statistical sectors of Belgium
        gdf = gpd.read_file("statistical_sectors.shp")
        # reading in statistical sector data which contains populations counts
        pop_df = pd.read_csv("statistical_sector_data.csv")
             
    def stat_sec_proportions(location="Leuven",coord_system=4326):
        # convert to preferred coordinate system
        gdf = gdf.to_crs(epsg=coord_system)
        # drop unnecessary multilingual labels
        gdf_wanted_columns = ["CS01012022","T_SEC_NL", "T_NIS6_NL", "T_MUN_NL", 
                                    "T_ARRD_NL", "T_PROVI_NL", "T_REGIO_NL", 
                                    "C_COUNTRY", 'M_AREA_HA', 'geometry']
        gdf_nl = gdf[gdf_wanted_columns]
        # subsetting by all statistical sectors in the municipality of choice
        gdf_location = gdf_nl[gdf_nl["T_MUN_NL"]==location]
        # selecting only the useful columns
        pop_df = pop_df[["CD_SECTOR", "TOTAL", "TX_DESCR_SECTOR_NL", "TX_DESCR_NL"]]
        # subsetting for sectors in chosen municipality
        pop_df = pop_df[pop_df["TX_DESCR_NL"]==location]
        # dropping incidence of residents whose sector was unknown
        pop_df = pop_df[pop_df["TX_DESCR_SECTOR_NL"]!="NIET TE LOKALISEREN IN EEN SECTOR"]
        # merge dataframe with population info with the geodataframe
        gdf_pop_location = pd.merge(right=pop_df,left=gdf_location, right_on="CD_SECTOR", left_on="CS01012022")
        # drop unwanted columns
        gdf_pop_location = gdf_pop_location.drop(["TX_DESCR_SECTOR_NL", "TX_DESCR_NL", "CS01012022"], axis = 1)
        # calculating population density for visualisation purposes
        gdf_pop_location["pop_density"] = gdf_pop_location["TOTAL"]/gdf_pop_location["M_AREA_HA"]
        # calculating total population of chosen municipality
        location_pop = sum(gdf_pop_location["TOTAL"])
        # calculating proportion of the population residing in each sector
        gdf_pop_location["pop_proportion"] = gdf_pop_location["TOTAL"]/location_pop
        
    # function which takes as input the desired proporiton of the population who will 
    # act as first responders, and outputs a proportional number of FRs in each sector    
    def generate_FRS(proportion=0.005):
        gdf_pop_location["fr_per_sector"]=round(gdf_pop_location["pop_proportion"]*proportion*location_pop).astype(int)
        # manipulating the output of our function so that it may be used to generate the
        # random points (first responders) for each sector
        fr_array = gdf_pop_location["fr_per_sector"].to_numpy()
        # generating random points
        gdf_pop_location["fr_loc"] = gdf_pop_location.sample_points(size=fr_array, method="uniform", rng=1)
        # getting each fr loc as a seperate point
        frs = gdf_pop_location["fr_loc"].explode()
        # getting the fr locs in coordinate form, in order to calculate the taxicab metric
        fr_coords=frs.get_coordinates()

    