import dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from dash.dependencies import Input, Output


app = dash.Dash(__name__,title='Zambia MDA Project', external_stylesheets=[dbc.themes.BOOTSTRAP])

# Coordinates of Leuven, Belgium
lat = 50.8798
lon = 4.7005

# Create a map figure
fig = go.Figure(go.Scattermapbox(
    lat=[lat],
    lon=[lon],
    mode='markers',
    marker=go.scattermapbox.Marker(
        size=14
    ),
))

# Update layout to use mapbox style and set initial zoom level
fig.update_layout(
    mapbox_style="open-street-map",
    mapbox_center={"lat": lat, "lon": lon},
    mapbox_zoom=12
)


# lat-long user inputs
latitude_input = dcc.Input(
    id='latitude_input',
    type='number',
    value=lat,
    step = 0.0001,
    debounce = True,
    style={'width': '100%'}
)

longitude_input = dcc.Input(
    id='longitude_input',
    type='number',
    value=lon,
    step = 0.0001,
    debounce = True,
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
    Output('id_title','children'),
    Output('id_graph','figure'),
    [Input('latitude_input', 'value'),
     Input('longitude_input','value')
     ]
)

def update_chart(latitude_value, longitude_value):
    updated_fig = fig.update_traces(lat=[latitude_value], lon=[longitude_value])
    return 'Coordinates of the patient (latitude, longitude): (' + str(latitude_value) + ',' + str(longitude_value) +')', updated_fig

if __name__ == '__main__':
    app.run_server(debug=True)