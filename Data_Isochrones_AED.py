# Class for routing betweeen responders and patients through AED locations

import openrouteservice
from openrouteservice import client
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import geometry
import time



df_aeds = pd.read_csv("C:/Users/leonw/Downloads/aed(Sheet1)(7).csv", encoding='unicode_escape')

ip =  "3.123.42.170"
Client_ors = openrouteservice.Client(base_url=f'http://{ip}:8080/ors')


def find_isochrone(Midpoint, profile = "foot-walking", threshold = 600):
    # Midpoint must be a tuple
    # AEDS must be a dataframe with columns (This is gathered in another file) named latitude and longitude
    # profile by default is walking by foot
    # set the parameters for conducting an isochrone search
    isochrones_parameters = {
      'locations': [Midpoint],
      'profile': profile,
      'range_type': 'time',
      'range': [threshold] # 10 minutes away (600 seconds)
    }

    # personal API key for openrouteservices
    ors_client_custom = Client_ors

    # Get the area that can be reached in 10 minutes walking.
    # This creates an isochrone wich is the whole area surounding the patient that is reachable in 10 minutes by foot.
    # 10 minutes are taken as most research assumes that 8 minutes is the longest time
    # first responders should take. As we assume the responders to walk faster than normal,
    # a 10 minute radius should be fine.
    isochrone = ors_client_custom.isochrones(**isochrones_parameters)

    return isochrone


def isochrones_AEDs(AED_df):
    list_position = []
    list_iso = []
        
    for i in range(len(AED_df)):
        coord = (AED_df['longitude'].iloc[i], AED_df['latitude'].iloc[i])  # Fixed tuple creation
        isochrone = find_isochrone(coord, profile="foot-walking", threshold=600)
            
        poly = geometry.Polygon(isochrone['features'][0]['geometry']['coordinates'][0])
        list_position.append(coord)
        list_iso.append(poly)
        time.sleep(0)

    df_AED_iso = pd.DataFrame(list(zip(list_position, list_iso)), columns=['AED', 'Iso'])
        
    AED_iso = gpd.GeoDataFrame(df_AED_iso, geometry='Iso', crs="EPSG:4326")  # Fixed CRS
    return AED_iso

df_iso_AED = df_aeds[["latitude","longitude"]].copy()
df_iso_AED = df_iso_AED.dropna(subset = ['latitude', 'longitude'])
df_iso_AED = isochrones_AEDs(df_iso_AED)
df_AED = df_aeds[["latitude","longitude","available","Checked","Opens", "Closes"]].copy()
df_AED = df_AED.dropna(subset = ['latitude', 'longitude'])

coord_list = []
for i in range(len(df_AED)):
        coord = (df_AED['longitude'].iloc[i], df_AED['latitude'].iloc[i])
        coord_list.append(coord)

df_AED['AED'] = coord_list
AED_df = df_iso_AED.merge(df_AED, on='AED')

# Drop the original tuple column if not needed
AED_df.drop(columns=['AED'], inplace=True)
AED_df.to_file('C:/Users/leonw/OneDrive - KU Leuven/Documents/GitHub/AED-Route-Optimization-MDA-Project/Data/temp.gpkg', layer='AED_data', driver='GPKG')

#######################################
### Visualization of the Isochrones ###
#######################################

import folium
import geopandas as gpd
from shapely.geometry import Point

# Assuming 'gdf' is your GeoDataFrame with 'AED' as points and 'Iso' as isochrones (polygons)

gdf = df_iso_AED


# Convert AED column to Points if not already in geometry format
if not isinstance(gdf['AED'].iloc[0], Point):
    gdf['AED'] = gdf['AED'].apply(lambda coords: Point(coords) if isinstance(coords, (list, tuple)) else coords)

# Set AED as the active geometry column in the GeoDataFrame
gdf = gdf.set_geometry('AED')

# Initialize a folium map centered around the average AED location
m = folium.Map(location=[gdf['AED'].y.mean(), gdf['AED'].x.mean()], zoom_start=13)

# Add isochrones (polygons) to the map
for _, row in gdf.iterrows():
    folium.GeoJson(
        row['Iso'],
        style_function=lambda x: {
            'fillColor': 'blue',
            'color': 'blue',
            'weight': 2,
            'fillOpacity': 0.3
        }
    ).add_to(m)

# Add AED points to the map
for _, row in gdf.iterrows():
    folium.Marker(
        location=[row['AED'].y, row['AED'].x],
        popup="AED",
        icon=folium.Icon(color='red', icon='heartbeat', prefix='fa')
    ).add_to(m)

# Save and display the map
m.save("aed_isochrones_map.html")





