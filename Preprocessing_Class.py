#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import requests
import time
from shapely.geometry import Point
import geopandas as gpd

class AEDPreprocessor:
    def __init__(self, aed_file_path, mapping_file_path, api_key):
        self.aed_file_path = aed_file_path
        # We introduce an external mapping file for filling missing street number data
        self.mapping_file_path = mapping_file_path
        self.api_key = api_key
    
    def load_data(self):
        # Load AED location data from parquet file
        self.AED_loc = pd.read_parquet(self.aed_file_path)
        # Load AED mapping data from CSV file
        self.AED_mapping = pd.read_csv(self.mapping_file_path)
        self.AED_mapping['address'] = self.AED_mapping['address'].str.title()
    
    def clean_data(self):
        # Correct values in column 'number' with one decimal point, leave NaN as it is
        self.AED_loc.loc[:, 'number'] = self.AED_loc['number'].apply(lambda x: str(int(float(x))) if (pd.notnull(x) and str(x).replace('.', '').isdigit()) else x)
        
        # Filter out rows with NaNs in 'province' and 'municipality'
        # Since none of them is in Leuven
        self.AED_loc = self.AED_loc[self.AED_loc['province'].notna() & self.AED_loc['municipality'].notna()]
        
        # Create a list of Leuven and its submunicipalities for filtering
        leuven_submunicipalities = ['Leuven', 'Heverlee', 'Kessel-Lo', 'Wilsele', 'Wijgmaal', 'Korbeek-Lo', 'Haasrode']
        self.filtered_AED_loc = self.AED_loc[self.AED_loc['municipality'].isin(leuven_submunicipalities)].copy()
        
        # Standardize 'public' column values
        public_mapping = {'Non-Nee': 'N', 'Oui-Ja': 'Y'}
        self.filtered_AED_loc.loc[:, 'public'] = self.filtered_AED_loc['public'].replace(public_mapping)
    
    def fill_missing_info(self):
        # Standardize 'address' column in filtered AED data
        self.filtered_AED_loc.loc[:, 'address'] = self.filtered_AED_loc['address'].str.replace(r'\s*\)+$', '', regex=True).str.title()
        
        # Fill in 'public' and 'number' info from the mapping file
        for index, row in self.filtered_AED_loc[self.filtered_AED_loc['public'].isna()].iterrows():
            address = row['address']
            if address in self.AED_mapping['address'].values:
                self.filtered_AED_loc.loc[self.filtered_AED_loc['address'] == address, 'public'] = 'Y'
        
        for index, row in self.filtered_AED_loc[self.filtered_AED_loc['public'].isna()].iterrows():
            address = row['address']
            if self.filtered_AED_loc.loc[self.filtered_AED_loc['address'] == address, 'public'].notna().any():
                public_value = self.filtered_AED_loc.loc[self.filtered_AED_loc['address'] == address, 'public'].dropna().iloc[0]
                self.filtered_AED_loc.loc[index, 'public'] = public_value
            # We treat the rest as private devices 
            else:
                self.filtered_AED_loc.loc[self.filtered_AED_loc['address'] == address, 'public'] = 'N'
    
    def generate_complete_address(self):
        # Generate 'complete_address' column
        # This is for precise API querry
        self.filtered_AED_loc.loc[:, 'complete_address'] = self.filtered_AED_loc.apply(
            lambda row: ' '.join([
                (str(row['number']) if pd.notna(row['number']) else ''),
                (row['address'] if pd.notna(row['address']) else '')
            ]).strip() + ', ' + ', '.join([
                (row['municipality'] if pd.notna(row['municipality']) else ''),
                (row['province'] if pd.notna(row['province']) else '')
            ]).strip(', ') + ', Belgium',
            axis=1
        )
    
    def geocode_addresses(self):
        # Geocode addresses to get latitude and longitude
        base_url = "https://maps.googleapis.com/maps/api/geocode/json"
        latitudes, longitudes = [], []
        
        for address in self.filtered_AED_loc['complete_address']:
            full_url = f"{base_url}?address={requests.utils.quote(address)}&key={self.api_key}"
            response = requests.get(full_url)
            response_json = response.json()
            if response_json['status'] == 'OK':
                latitude = response_json['results'][0]['geometry']['location']['lat']
                longitude = response_json['results'][0]['geometry']['location']['lng']
                latitudes.append(latitude)
                longitudes.append(longitude)
            else:
                latitudes.append(None)
                longitudes.append(None)
            time.sleep(1)
        self.filtered_AED_loc.loc[:, 'latitude'] = latitudes
        self.filtered_AED_loc.loc[:, 'longitude'] = longitudes
        self.filtered_AED_loc.dropna(subset=['latitude', 'longitude'], inplace=True)
    
    def to_geodataframe(self):
        # Convert to GeoDataFrame
        self.filtered_AED_loc['geometry'] = self.filtered_AED_loc.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
        self.aed_gdf = gpd.GeoDataFrame(self.filtered_AED_loc, geometry='geometry')
    
    def preprocess(self):
        self.load_data()
        self.clean_data()
        self.fill_missing_info()
        self.generate_complete_address()
        self.geocode_addresses()
        self.to_geodataframe()
        return self.aed_gdf


# In[ ]:




