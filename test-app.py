import dash
from dash import html

# Create a Dash application instance
app = dash.Dash(__name__)

# Define the layout of your application
app.layout = html.Div("Hello, Dash!")

# Ensure the application is callable
if __name__ == '__main__':
    app.run_server()