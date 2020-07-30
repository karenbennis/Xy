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
# Body
#############################################################

"""App Components"""

## Machine learning dropdown app ##
dropdown_app = dbc.DropdownMenu(
    label="Select a Model",
    children=[
        dbc.DropdownMenuItem("Naive Bayes"),
        dbc.DropdownMenuItem("Logistic Regression"),
        dbc.DropdownMenuItem("Neural Net"),
        dbc.DropdownMenuItem("Outside Model")
    ],
)

## Machine learning text area app ##
text_input = html.Div(
    [
        #dbc.Label("Try it Yourself"),
        dbc.Textarea(id="input", placeholder="Write a review...",),
        html.Br(),
        html.P(id="output"),
    ]
)

## Dataset dropdown app ##

data_set_dropdown = dbc.DropdownMenu(
    label="Select Dataset",
    children=[
        dbc.DropdownMenuItem("Unbalanced"),
        dbc.DropdownMenuItem("Balanced"),
        dbc.DropdownMenuItem("Scaled"),
    ],
)


## Graph one app ##

# Set the file path
file_path = 'uniform_yelp.csv'

# Read in data from csv
df = pd.read_csv(file_path)

# Get the number of times each star rating appears in dataset
stars_count = df.stars.value_counts()
star_count_df = pd.DataFrame(stars_count).sort_index()

# Initialize bar chart object
fig = px.bar(star_count_df, y=["stars"], barmode="group")

# Create the plot
explore_graph_app = dcc.Graph(
        id='explore-graph',
        figure=fig
    )

## Tab one app for Graph ##
tab1_app = dbc.Card(
    dbc.CardBody(
        [
            explore_graph_app
        ]
    ),
    className="mt-3",
)

## Tab two app for Graph ##
tab2_app = dbc.Card(
    dbc.CardBody(
        [
            html.P("tab two :(", className="card-text")
        ]
    ),
    className="mt-3",
)

## Parent tabs app ##
parent_tab_app = dbc.Tabs(
    [
        dbc.Tab(tab1_app, label="Tab 1"),
        dbc.Tab(tab2_app, label="Tab 2"),
    ]
)


"""Cards"""

# Card 1
card = dbc.Card(
    [
        dbc.CardImg(src="/static/images/ml_text_banner.jpeg", top=True),
        dbc.CardBody(
            [
                html.H4("What the Dilly with That There Machine Learnin' Stuff", className="card-title"),
                html.P(
                    "An app allowing users to test different machine learning models' "
                    "ability to classify the sentiment of their review.",
                    className="card-text",
                ),
                dropdown_app,
                html.Br(),
                text_input,
                dbc.Button("Predict Rating", color="primary", id="open", style={'margin':'auto','width':'100%'}),
            #     dbc.Modal(
            #     [
            #         dbc.ModalHeader("Try it Yourself!"),
            #         dbc.ModalBody(dropdown_app),
            #         dbc.ModalBody(text_input),
            #         dbc.ModalFooter(
            #             dbc.Button("Close", id="close", className="ml-auto")
            #         ),
            #     ],
            #     id="modal",
            # )
            ]
        ),
    ],
    style={"width": "45rem"},
)

# Card 2

card_two = dbc.Card(
    [
        dbc.CardBody(
            [
                html.H4("Data Exploration", className="card-title"),
                data_set_dropdown,
                parent_tab_app,
                                
            ]
        ),
    ],
    style={"width": "45rem"},
)

"""Body"""
body = html.Div(
    dbc.Row([
        dbc.Col(html.Div(card)),
        dbc.Col(html.Div(card_two))
    ])
)

""" Final Layout Render"""
app.layout = html.Div(
    [navbar, body]
)

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

# # Modal 1
# @app.callback(
#     Output("modal", "is_open"),
#     [Input("open", "n_clicks"), Input("close", "n_clicks")],
#     [State("modal", "is_open")],
# )
# def toggle_modal(n1, n2, is_open):
#     if n1 or n2:
#         return not is_open
#     return is_open

"""End App Callback"""

if __name__ == '__main__':
    app.run_server(debug=True)