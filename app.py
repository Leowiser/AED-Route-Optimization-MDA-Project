import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from dash.dependencies import Input, Output, State
import plotly.express as px
from Routing_Class import route
from FR_Generation_Class import FR_Generation as fr

app = dash.Dash(__name__, title='Zambia MDA Project', external_stylesheets=[dbc.themes.BOOTSTRAP])

# Initiate a route object from the Routing_class
route = route()

# Generate a realistic dispersion of first responders using FR_Generation_Class
gdf, pop_df = fr.load_data()
pop_gdf, location_pop = fr.stat_sec_proportions(gdf, pop_df)

def generate_responders(proportion):
    return fr.generate_FRs(pop_gdf, location_pop, proportion=proportion)

# Initial proportion for responders
initial_proportion = 0.005
responder = generate_responders(initial_proportion)

# Data of AED-s
df_aed = pd.read_csv('filtered_AED_loc.csv')  # Cleaned dataset with the locations of the AEDs
AED_longitudes = df_aed['longitude'].values
AED_latitudes = df_aed['latitude'].values
aed = list(zip(AED_longitudes, AED_latitudes))  # Zip to one tuple
aed = pd.DataFrame(aed, columns=['longitude', 'latitude'])  # Transform aed to dataframe with headings

# Coordinates of Leuven, Belgium
lat = 50.8798
lon = 4.7005

# Create a map figure visualizing the AEDs
fig = go.Figure(go.Scattermapbox(
    lat=aed['latitude'],
    lon=aed['longitude'],
    mode='markers',
    marker=go.scattermapbox.Marker(
        size=7,
        color='green',
    ),
    name='AEDs'
))

# Add responders to the map
fig.add_trace(go.Scattermapbox(
    lat=responder['latitude'],
    lon=responder['longitude'],
    mode='markers',
    marker=go.scattermapbox.Marker(
        size=4,
        color='blue',
        symbol='circle',
        opacity=0.8,
    ),
    name='Responders'
))

# Add a "default" patient to the middle of Leuven
fig.add_trace(go.Scattermapbox(
    lat=[lat],
    lon=[lon],
    mode='markers',
    marker=go.scattermapbox.Marker(
        size=10,
        color='red',
        symbol='circle',
        opacity=0.8,
    ),
    name='Patient'
))

# Update layout for nicer design
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

# Slider for adjusting the proportion of responders - user input
proportion_slider = dcc.Slider(
    id='proportion_slider',
    min=0.001,
    max=0.01,
    step=0.0001,  # Make the slider smoother
    value=initial_proportion,
    marks=None,
    tooltip={"placement": "bottom", "always_visible": True}
)

# Design the app
app.layout = dbc.Container(
    [
        html.Div(
            children=[
                html.H1(children='AED & Responder route optimization'),
                html.H2(children='Default title 1', id='id_title'),
                html.H2(children='Default title 2', id='id_title2')
            ],
            style={'textAlign': 'center', 'color': 'black'}
        ),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Row(html.Label("Responder proportion")),
                        dbc.Row(proportion_slider),
                        html.Div(style={'height': '50px'}),
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
     Output('id_title2', 'children'),
     Output('id_graph', 'figure')],
    [Input('button', 'n_clicks'),
     Input('proportion_slider', 'value')],
    [State('latitude_input', 'value'),
     State('longitude_input', 'value')]
)
def update_chart(n_clicks, proportion, latitude_value, longitude_value):
    # Determine which input triggered the callback
    ctx = dash.callback_context

    if not ctx.triggered:
        # Default state of the app - without any callback triggered yet
        return f'Adjust the responder ratio and patient coordinate parameters on the left', 'Click the button on the bottom after changing patient coordinates!', fig

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'proportion_slider':
        # Update responders based on the new proportion
        global responder
        responder = generate_responders(proportion)

        # Update the figure with new responders
        fig.data = [fig.data[0]]    # First, keep only the AED locations
        
        # Add the responders again to the map now based on the updated responder proportion
        fig.add_trace(go.Scattermapbox(
            lat=responder['latitude'],
            lon=responder['longitude'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=4,
                color='blue',
                symbol='circle',
                opacity=0.8,
            ),
            name='Responders'
        ))

        # Add a "default" patient to the middle of Leuven 
        fig.add_trace(go.Scattermapbox(
            lat=[lat],
            lon=[lon],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=10,
                color='red',
                symbol='circle',
                opacity=0.8,
            ),
            name='Patient'
        ))
        
        # Update layout so that the map is centered at the middle of Leuven
        fig.update_layout(mapbox_center={"lat": lat, "lon": lon})

        return f'Responders are generated assuming {round(proportion*100, 2)}% of the population is a first responder','Click the button on the bottom after changing patient coordinates!', fig

    if trigger_id == 'button':
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
        fig.data = [fig.data[0]]    # First, keep only the AED locations

        # Add responders back to the map
        fig.add_trace(go.Scattermapbox(
            lat=responder['latitude'],
            lon=responder['longitude'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=4,
                color='blue',
                symbol='circle',
                opacity=0.8,
            ),
            name='Responders'
        ))

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
            text='First responder direct',
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
            name='AED to pick up',
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

        return f'Time to arrive for direct responder: {round(direct_route["duration"]/60, 1)} mins', f'Time to arrive for responder through AED: {round(AED_route["duration"]/60, 1)} mins', updated_fig

if __name__ == '__main__':
    app.run_server(debug=True)
