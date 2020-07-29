import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import requests, base64
from io import BytesIO
import plotly.graph_objs as go
from collections import Counter
import plotly.express as px
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

"""Navbar"""

PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

nav_item = dbc.NavItem(dbc.NavLink('GitHub', href='https://github.com/karenbennis/Xy'))

# Dropdown menu with links to our portfolios
dropdown = dbc.DropdownMenu(children=[
    dbc.DropdownMenuItem("Blake's Portfolio"),
    dbc.DropdownMenuItem("Helen's Portfolio"),
    dbc.DropdownMenuItem("Jasmeer's Portfolio"),
    dbc.DropdownMenuItem("Karen's Portfolio"),
    ],
    nav=True,
    in_navbar=True,
    label='Team Portfolios'
)

navbar = dbc.Navbar(
    dbc.Container(
    [
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                    dbc.Col(dbc.NavbarBrand("Yelp Sentiment Analysis", className="ml-2")),
                ],
                align="center",
                no_gutters=True,
            ),
            href="https://plot.ly",
        ),
        dbc.NavbarToggler(id="navbar-toggler"),
        dbc.Collapse(dbc.Nav([nav_item, dropdown], className='ml-auto', navbar=True), id="navbar-collapse", navbar=True),
    ],
    ),
    color="dark",
    dark=True,
    className='mb-5'
)

"""Navbar End"""

#############################################################
"""App Components"""

DropdownApp = dbc.DropdownMenu(
    label="Select a Model",
    children=[
        dbc.DropdownMenuItem("Naive Bayes"),
        dbc.DropdownMenuItem("Logistic Regression"),
        dbc.DropdownMenuItem("Neural Net"),
        dbc.DropdownMenuItem("Outside Model")
    ],
)

text_input = html.Div(
    [
        dbc.Label("Try it Yourself"),
        dbc.Input(id="input", placeholder="Type a review...", type="text"),
        html.Br(),
        html.P(id="output"),
    ]
)


""" Final Layout Render"""
app.layout = html.Div(
    [navbar, DropdownApp, text_input]
)


# # Set the file path
# file_path = 'uniform_yelp.csv'

# # Read in data from csv
# df = pd.read_csv(file_path)

# stars_count = df.stars.value_counts()

# star_count_df = pd.DataFrame(stars_count).sort_index()

# fig = px.bar(star_count_df, y=["stars"], barmode="group")

# app.layout = html.Div(children=[
#     html.H1(children='Data Exploration'),

#     html.Div(children='''
#         An interactive web app to explore Machine Learning with Natural Language Processing.
#     '''),

#     dcc.Graph(
#         id='example-graph',
#         figure=fig
#     )
# ])

"""App Callback"""

# Navbar
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)

def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# Text Input App
@app.callback(Output("output", "children"), [Input("input", "value")])
def output_text(value):
    return value


"""End App Callback"""

if __name__ == '__main__':
    app.run_server(debug=True)