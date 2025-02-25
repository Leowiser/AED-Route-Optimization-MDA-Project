# Class for routing betweeen responders and patients through AED locations

import openrouteservice
from openrouteservice import client
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import geometry
import time



df_aeds = pd.read_csv("OneDrive - KU Leuven/Documents/GitHub/AED-Route-Optimization-MDA-Project/filtered_AED_loc.csv")

Client_ors = openrouteservice.Client(key='5b3ce3597851110001cf624802e069d6633748a5ae4e9842334f1dc2')


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
        time.sleep(1.7)

    df_AED_iso = pd.DataFrame(list(zip(list_position, list_iso)), columns=['AED', 'Iso'])
        
    AED_iso = gpd.GeoDataFrame(df_AED_iso, geometry='Iso', crs="EPSG:4326")  # Fixed CRS
    return AED_iso


df_iso_AED = isochrones_AEDs(df_aeds)

df_iso_AED.to_file("C:/Users/leonw/OneDrive - KU Leuven/Documents/GitHub/AED-Route-Optimization-MDA-Project/Data/AED_polygons.shp", driver="ESRI Shapefile")



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





