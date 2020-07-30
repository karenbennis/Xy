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
        dbc.DropdownMenuItem("Naive Bayes", id="Naive-button"),
        dbc.DropdownMenuItem("Logistic Regression", id="logistic-button"),
        dbc.DropdownMenuItem("Neural Net", id="neu-button"),
        dbc.DropdownMenuItem("Outside Model", id="outside-button")
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

data_set_dropdown = html.Div(
    [
        dcc.Dropdown(
        id='dataset_dropdown',
        options=[
            {'label': 'Balanced Dataset', 'value': '1'},
            {'label': 'Unbalanced Dataset', 'value': '2'},
            {'label': 'Scaled Dataset', 'value': '3'}
        ],
        value='1'
    ),
    html.Div(id='dd-output-container')
    ],
)


## Graph one app ##

# Set the file path
file_path = 'uniform_yelp.csv'

# Read in data from csv
df = pd.read_csv(file_path)
df['length'] = df['text'].apply(len)
df = df.loc[df['length'] < 3200]
# Create df for each star
one_star_df = df.loc[df['stars'] == 1]
two_star_df = df.loc[df['stars'] == 2]
three_star_df = df.loc[df['stars'] == 3]
four_star_df = df.loc[df['stars'] == 4]
five_star_df = df.loc[df['stars'] == 5]



# Get the number of times each star rating appears in dataset
stars_count = df.stars.value_counts()
star_count_df = pd.DataFrame(stars_count).sort_index()

# Initialize bar chart object
fig = px.bar(star_count_df, y=["stars"], barmode="group")

# # Create the plot
explore_graph_app = dcc.Graph(
        id='explore_graph',
        figure=fig
    )

## Graph two app ##
# Create plot adding each figure individually so the color of each marker can be changed

trace_1 = go.Box(x=one_star_df['stars'], y=one_star_df['length'], marker_color='royalblue', boxmean='sd')
trace_2 = go.Box(x=two_star_df['stars'], y=two_star_df['length'], marker_color='royalblue', boxmean='sd')
trace_3 = go.Box(x=three_star_df['stars'], y=three_star_df['length'], marker_color='royalblue', boxmean='sd')
trace_4 = go.Box(x=four_star_df['stars'], y=four_star_df['length'], marker_color='royalblue', boxmean='sd')
trace_5 = go.Box(x=five_star_df['stars'], y=five_star_df['length'], marker_color='royalblue', boxmean='sd')

data = [trace_1, trace_2, trace_3, trace_4, trace_5]
box_layout = go.Layout(
    title = "Length of Review per Star Rating"
)

fig2 = go.Figure(data=data, layout=box_layout)

# Create the boxplot
box_plot_graph = dcc.Graph(
    figure=fig2
)


## Tab one app for Graph ##
tab1_app = dbc.Card(
    dbc.CardBody(
        [
            explore_graph_app
        ]
    ),
    className="mt-3",
    id='first_tab',
)

## Tab two app for Graph ##
tab2_app = dbc.Card(
    dbc.CardBody(
        [
            box_plot_graph
        ]
    ),
    className="mt-3",
    id='second_tab',
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

# # Dataset Dropdown
# @app.callback(
#     Output("explore_graph", "figure"),
#     [Input("dataset_dropdown", "value")],
# )
# def update_fig(value):
#     data=[]
#     if value:
#         file_path = 'yelp.csv'
#         df2 = pd.read_csv(file_path)
#         df2['length'] = df['text'].apply(len)
#         df2 = df.loc[df['length'] < 3200]
#         # Get the number of times each star rating appears in dataset
#         stars_count = df2.stars.value_counts()
#         star_count_df = pd.DataFrame(stars_count).sort_index()

#         # Initialize bar chart object
#         fig = px.bar(star_count_df, y=["stars"], barmode="group")
#         # Create the plot
#         dataset_callback = dcc.Graph(
#             figure=fig
#         )
#         data.append(dataset_callback)
#     return {'data': data}

"""End App Callback"""

if __name__ == '__main__':
    app.run_server(debug=True)