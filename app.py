import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output, State
import plotly.express as px
from Routing_Class import route

app = dash.Dash(__name__, title='Zambia MDA Project', external_stylesheets=[dbc.themes.BOOTSTRAP])

route = route()

# Generate 40 random lat and long for responders
N = 40
lon_resp = np.random.uniform(4.69, 4.71, N)
lat_resp = np.random.uniform(50.85, 50.88, N)

# Create dataframe for responders
responder = pd.DataFrame({'longitude': lon_resp, 'latitude': lat_resp})

# Data of AED-s
df_aed = pd.read_csv('filtered_AED_loc.csv')    # Cleaned dataset with the locations of the AEDs 
AED_longitudes = df_aed['longitude'].values
AED_latitudes = df_aed['latitude'].values
aed = list(zip(AED_longitudes, AED_latitudes))    # Zip to one tuple 
aed = pd.DataFrame(aed, columns=['longitude', 'latitude'])    # Transform aed to dataframe with headings

# Coordinates of Leuven, Belgium
lat = 50.8798
lon = 4.7005

# Create a map figure
fig = px.scatter_mapbox(aed, lat="latitude", lon="longitude", color_discrete_sequence=["green"])
fig.update_traces(marker=dict(size=7))

# Add responders to the map
fig.add_trace(go.Scattermapbox(
    lat=responder['latitude'],
    lon=responder['longitude'],
    mode='markers',
    marker=go.scattermapbox.Marker(
        size=7,
        color='blue',
        symbol='circle',
        opacity=0.8,
    ),
    name='Responder'
))

fig.add_trace(go.Scattermapbox(
    lat=[lat],
    lon=[lon],
    mode='markers',
    marker=go.scattermapbox.Marker(
        size=10,
        color='red',  # Color of the new point
        symbol='circle',  # Symbol of the new point
        opacity=0.8,
    ),
    name='Patient'  # Name of the new point for legend
))

# Update layout to use mapbox style and set initial zoom level
fig.update_layout(
    mapbox_style="carto-positron",
    mapbox_center={"lat": lat, "lon": lon},
    mapbox_zoom=13,
    height=500,
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    legend=dict(orientation='h', yanchor='bottom')
)

# lat-long user inputs
latitude_input = dcc.Input(
    id='latitude_input',
    type='number',
    value=lat,
    step=0.0001,
    debounce=True,
    style={'width': '100%'}
)

longitude_input = dcc.Input(
    id='longitude_input',
    type='number',
    value=lon,
    step=0.0001,
    debounce=True,
    style={'width': '100%'}
)

app.layout = dbc.Container(
    [
        html.Div(
            children=[
                html.H1(children='AED & Responder route optimization'),
                html.H2(children='Please wait while the figure updates...', id='id_title')
            ],
            style={'textAlign': 'center', 'color': 'black'}
        ),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Row(html.Label("Patient latitude")),
                        dbc.Row(latitude_input),
                        dbc.Row(html.Label("Patient longitude")),
                        dbc.Row(longitude_input),
                        html.Div(style={'height': '50px'}),
                        dbc.Row(html.Label("Click button to calculate route")),
                        dbc.Row(html.Button('Click me', id='button'))
                    ],
                    md=2
                ),
                dbc.Col(dcc.Graph(id="id_graph", figure=fig), md=10),
            ],
            align="center",
        ),
    ],
    fluid=True,
)

@app.callback(
    [Output('id_title', 'children'),
     Output('id_graph', 'figure')],
    [Input('button', 'n_clicks')],
    [State('latitude_input', 'value'),
     State('longitude_input', 'value')]
)
def update_chart(n_clicks, latitude_value, longitude_value):
    if n_clicks is None:
        # Initial rendering
        return 'Coordinates of the patient (latitude, longitude): (' + str(lat) + ',' + str(lon) + ')', fig

    # Get coordinates of the optimal first-and second responder's and AED's location
    best_coordinates = route.send_responders((longitude_value, latitude_value), responder, aed)
    coord_direct = best_coordinates['coord_direct']
    coord_AED = best_coordinates['coord_AED']
    AED_coordinates = best_coordinates['AED_coordinates']

    # Get both routes
    direct_route = route.directions([coord_direct, (longitude_value, latitude_value)])
    AED_route = route.directions([coord_AED, AED_coordinates, (longitude_value, latitude_value)])

    # Get a dataframe of the description of the route for plotting
    # To transform the route into usable data frame for plotting with the get_coordinates function
    df_latlong_direct = route.get_coordinates(direct_route['coordinates'])
    df_latlong_AED = route.get_coordinates(AED_route['coordinates'])

    # "Reset" the plot to keep only the AED locations
    fig.data = [fig.data[0], fig.data[1]]
    # Plot the new patient location (after user changed input)
    updated_fig = fig.update_traces(lat=[latitude_value], lon=[longitude_value], selector=dict(name='Patient'))
    updated_fig.update_layout(mapbox_center={"lat": latitude_value, "lon": longitude_value})    # center plot around the new coordinates

    # plot the direct way - first responder to patient
    direct_trace = px.line_mapbox(df_latlong_direct, lat="lat", lon="lon").data[0]
    direct_trace.line.width = 4
    direct_trace.line.color = 'darkblue'
    updated_fig.add_trace(direct_trace)
    # Add the route through the AED
    AED_trace = px.line_mapbox(df_latlong_AED, lat='lat', lon='lon').data[0]
    AED_trace.line.width = 4
    AED_trace.line.color = 'orange'
    updated_fig.add_trace(AED_trace)

    # Add marker for the first responder's initial location
    beginning_direct = go.Scattermapbox(
        lat=[df_latlong_direct['lat'].iloc[0]],
        lon=[df_latlong_direct['lon'].iloc[0]],
        mode='markers',
        name='Route of first responder',
        marker=go.scattermapbox.Marker(
            size=10,
            color='darkblue'
        ),
        text='First responder direct',  # Text to display when hovering over the marker
        hoverinfo='text'
    )

    # Add markers for the first responder that takes the route through the AED
    beginning_AED = go.Scattermapbox(
        lat=[df_latlong_AED['lat'].iloc[0]],
        lon=[df_latlong_AED['lon'].iloc[0]],
        mode='markers',
        name='Route of responder through AED',
        marker=go.scattermapbox.Marker(
            size=10,
            color='orange'
        ),
        text='Start responder through AED',
        hoverinfo='text'
    )

    # Add marker for the Patient
    Patient = go.Scattermapbox(
        lat=[df_latlong_AED['lat'].iloc[-1]],
        lon=[df_latlong_AED['lon'].iloc[-1]],
        mode='markers',
        name='Patient',
        marker=go.scattermapbox.Marker(
            size=15,
            color='red'
        ),
        text='Patient',
        hoverinfo='text'
    )

    # Add a marker for the AED
    AED_marker = go.Scattermapbox(
        lat=[AED_coordinates[1]],
        lon=[AED_coordinates[0]],
        mode='markers',
        name='AED',
        marker=go.scattermapbox.Marker(
            size=15,
            color='green'
        ),
        text='AED device',
        hoverinfo='text'
    )

    # Add the markers to the figure
    updated_fig.add_trace(beginning_direct)
    updated_fig.add_trace(beginning_AED)
    updated_fig.add_trace(Patient)
    updated_fig.add_trace(AED_marker)

    return 'Coordinates of the patient (latitude, longitude): (' + str(latitude_value) + ',' + str(longitude_value) + ')', updated_fig

if __name__ == '__main__':
    app.run_server(debug=True)
