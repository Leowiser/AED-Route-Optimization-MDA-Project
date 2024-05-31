#required packages
import pandas as pd
import geopandas as gpd
from pyproj import CRS
import warnings
## Silencing some pesky warnings
warnings.filterwarnings("ignore", message="`unary_union` returned None due to all-None GeoSeries. In future, `unary_union` will return 'GEOMETRYCOLLECTION EMPTY' instead.", category=FutureWarning)
warnings.filterwarnings("ignore", message="Currently, index_parts defaults to True, but in the future, it will default to False to be consistent with Pandas. Use `index_parts=True` to keep the current behavior and True/False to silence the warning.", category=FutureWarning)

class FR_Generation:
    
    def load_data():
        # read in the geospatial data for the statistical sectors of Belgium
        gdf = gpd.read_file("Data/leuven.shp")
        # reading in statistical sector data which contains populations counts
        pop_df = pd.read_csv("Data/leuven.csv")
        return gdf, pop_df
             
    def stat_sec_proportions(gdf, pop_df, coord_system=4326):      
        # convert to preferred coordinate system
        gdf = gdf.to_crs(epsg=coord_system)
        # drop unnecessary multilingual labels
        gdf_wanted_columns = ["CS01012022","T_SEC_NL", "T_NIS6_NL", "T_MUN_NL", 
                                    "T_ARRD_NL", "T_PROVI_NL", "T_REGIO_NL", 
                                    "C_COUNTRY", 'M_AREA_HA', 'geometry']
        gdf_nl = gdf[gdf_wanted_columns]
        ## Now working with the population dataset
        # selecting only the useful columns
        pop_df = pop_df[["CD_SECTOR", "TOTAL", "TX_DESCR_SECTOR_NL", "TX_DESCR_NL"]]
        # merge population dataframe with the geodataframe
        pop_gdf = pd.merge(left=gdf_nl, right=pop_df, left_on="CS01012022", right_on="CD_SECTOR")
        # drop unwanted columns
        pop_gdf = pop_gdf.drop(["TX_DESCR_SECTOR_NL", "TX_DESCR_NL", "CS01012022"], axis = 1)
        # calculating population density for visualisation purposes
        pop_gdf["pop_density"] = pop_gdf["TOTAL"]/pop_gdf["M_AREA_HA"]
        # calculating total population of chosen municipality
        location_pop = sum(pop_gdf["TOTAL"])
        # calculating proportion of the population residing in each sector
        pop_gdf["pop_proportion"] = pop_gdf["TOTAL"]/location_pop
        return pop_gdf, location_pop
        
    # function which takes as input the desired proportion of the population who will 
    # act as first responders, and outputs a proportional number of FRs in each sector    
    def generate_FRs(pop_gdf, location_pop, proportion=0.005):
        pop_gdf["fr_per_sector"]=round(pop_gdf["pop_proportion"]*proportion*location_pop).astype(int)
        # manipulating the output of our function so that it may be used to generate the
        # random points (first responders) for each sector
        fr_array = pop_gdf["fr_per_sector"].to_numpy()
        # generating random points
        pop_gdf["fr_loc"] = pop_gdf.sample_points(size=fr_array, method="uniform", rng=1, index_parts=True)
        # getting each fr loc as a seperate point
        frs = pop_gdf["fr_loc"].explode()
        # getting the fr locs in coordinate form, in order to calculate the taxicab metric etc.
        fr_coords=frs.get_coordinates()
        fr_coords = fr_coords.reset_index(drop=True)
        responder = pd.DataFrame({'longitude':fr_coords["x"], 'latitude':fr_coords["y"]})
        return responder
    