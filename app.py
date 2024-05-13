import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output
from Routing_Class import route
import plotly.express as px

app = dash.Dash(__name__, title='Zambia MDA Project', external_stylesheets=[dbc.themes.BOOTSTRAP])

# Generate 40 random lat and long for responders
N = 40
lon_resp = np.random.uniform(4.69, 4.71, N)
lat_resp = np.random.uniform(50.85, 50.88, N)

# Create dataframe for responders
responder = pd.DataFrame({'longitude': lon_resp, 'latitude': lat_resp})

# Generate 10 random lat and long for AEDs
N = 10
lon_aed = np.random.uniform(4.69, 4.71, N)
lat_aed = np.random.uniform(50.85, 50.88, N)

# Create dataframe for AEDs
aed = pd.DataFrame({'longitude': lon_aed, 'latitude': lat_aed})

# Coordinates of Leuven, Belgium
lat = 50.8798
lon = 4.7005

# Create a map figure
fig = px.scatter_mapbox(aed, lat="latitude", lon="longitude", color_discrete_sequence=["green"])
fig.update_traces(marker=dict(size=7))
'''
# Create a map figure
fig = go.Figure(go.Scattermapbox(
    lat=[lat],
    lon=[lon],
    mode='markers',
    marker=go.scattermapbox.Marker(
        size=14
    ),
))
'''

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

# fig = px.scatter_mapbox(aed, lat="latitude", lon="longitude", zoom=3, height=300, color_discrete_sequence=["green"])
# fig.update_traces(marker=dict(size=7))

# Update layout to use mapbox style and set initial zoom level
fig.update_layout(
    mapbox_style="carto-positron",
    mapbox_center={"lat": lat, "lon": lon},
    mapbox_zoom=13,
    height=500,
    margin={"r": 0, "t": 0, "l": 0, "b": 0}
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
                html.H2(children='...', id='id_title')
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
                        dbc.Row(longitude_input)
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
    Output('id_title', 'children'),
    Output('id_graph', 'figure'),
    [Input('latitude_input', 'value'),
     Input('longitude_input', 'value')
     ]
)
def update_chart(latitude_value, longitude_value):
    updated_fig = fig.update_traces(lat=[latitude_value], lon=[longitude_value], selector=dict(name='Patient'))
    updated_fig.update_layout(mapbox_center={"lat": latitude_value, "lon": longitude_value})    # center plot around the new coordinates
    # test = route()
    # updated_fig = fig.update_traces(test.send_responders((longitude_value, latitude_value), responder, aed))
    return 'Coordinates of the patient (latitude, longitude): (' + str(latitude_value) + ',' + str(longitude_value) +')', updated_fig

if __name__ == '__main__':
    app.run_server(debug=True)