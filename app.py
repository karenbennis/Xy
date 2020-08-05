import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import requests, base64
from io import BytesIO
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from collections import Counter
import plotly.express as px
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import numpy as np

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

#### Get Data and  ML Models ####

### ML models ###

## Naive Bayes ##

# Import data for ml model
ml_file_path = 'static/Resources/ml_app_df.csv'
ml_input_df = pd.read_csv(ml_file_path)

# print(ml_input_df)

# Create object with review data
df_x = ml_input_df['cleaned']

# print(df_x)

# Create object with class(star) data
df_y = ml_input_df['class']

# print(df_y)

# Split data into training and testing
X_train,X_test,y_train,y_test = train_test_split(df_x,df_y,test_size=0.2,random_state=41)

# Initialize tfidf vectorizer
tfidf = TfidfVectorizer(min_df=1)

# Fit training data to vectorizer
X = tfidf.fit_transform(X_train)

# Initialize Niave Bayes object
mnb = MultinomialNB()

# Cast y_train as int
y_train = y_train.astype('int')

# Fit Naive Bayes model
mnb.fit(X, y_train)

# Transform X_test
X_test = tfidf.transform(X_test)

# Initialize predict object for testing model accuracy
pred = mnb.predict(X_test)

# Initialize y_test object for testing model accuracy
actual = np.array(y_test)

# For displaying full dataset prediction if we decide to add it at a later date
count=0
for i in range(len(pred)):
    if pred[i] == actual[i]:
        count+=1

## Logistic Regression ##

# Initialize logistic regression object
logr = LogisticRegression()

# Fit logistic regression model
logr.fit(X,y_train)

"""Navbar"""

yelp_logo = "static/images/yelp_logo.png"

nav_item = dbc.NavItem(dbc.NavLink('GitHub', href='https://github.com/karenbennis/Xy'))

# Dropdown menu with links to our portfolios
dropdown = dbc.DropdownMenu(children=[
    dbc.DropdownMenuItem("Blake's GitHub", href='https://github.com/blocrunx'),
    dbc.DropdownMenuItem("Helen's GitHub", href='https://github.com/Helen-Ly'),
    dbc.DropdownMenuItem("Jasmeer's GitHub", href='https://github.com/JasmeerSangha'),
    dbc.DropdownMenuItem("Karen's GitHub", href='https://github.com/karenbennis'),
    ],
    nav=True,
    in_navbar=True,
    label='Team Repositories'
)

navbar = dbc.Navbar(
    
    [
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    html.Img(src=yelp_logo, height="70px"),
                    dbc.NavbarBrand("NLP Sentiment Analysis of Yelp Reviews"),
                ],
                align="center",
                no_gutters=True,
                className='ml-auto'
            ),
            #href="https://plot.ly",
        ),
        dbc.NavbarToggler(id="navbar-toggler"),
        dbc.Collapse(dbc.Nav([nav_item, dropdown], className='ml-auto', navbar=True), id="navbar-collapse", navbar=True),
    ],
    color="dark",
    dark=True,
    className="ml-auto flex-nowrap mt-3 mt-md-0",
)

"""Navbar End"""

#############################################################
# Body
#############################################################

"""App Components"""

# ## Machine learning dropdown app ##
# dropdown_app = dbc.DropdownMenu(
#     label="Select a Model",
#     children=[
#                 dbc.DropdownMenuItem("Naive Bayes", id="naive-button"),
#                 dbc.DropdownMenuItem("Logistic Regression", id="logistic-button")
#             ],
# )

# ## Machine learning text area app ##
# text_input = html.Div(
#     [
#         #dbc.Label("Try it Yourself"),
#         dbc.Textarea(id="input", placeholder="Write a review...",),
#         html.Br(),
#         html.P(id="output"),
#     ]
# )

# ## Dataset dropdown app ##
# data_set_dropdown = html.Div(
#     [
#         dcc.Dropdown(
#         id='dataset_dropdown',
#         options=[
#             {'label': 'Balanced Dataset', 'value': '1'},
#             {'label': 'Unbalanced Dataset', 'value': '2'},
#             {'label': 'Scaled Dataset', 'value': '3'}
#         ],
#         value='1'
#     ),
#     html.Div(id='dd-output-container')
#     ],
# )
## Graph one app ##

# # Set the file path
# file_path = 'uniform_yelp.csv'

# # Read in data from csv
# df = pd.read_csv(file_path)

# # Create a length column
# df['length'] = df['text'].apply(len)

# # Cut out major outliers
# df = df.loc[df['length'] < 3200]

# # Create df for each star to be used in box plot
# one_star_df = df.loc[df['stars'] == 1]
# two_star_df = df.loc[df['stars'] == 2]
# three_star_df = df.loc[df['stars'] == 3]
# four_star_df = df.loc[df['stars'] == 4]
# five_star_df = df.loc[df['stars'] == 5]



# # Get the number of times each star rating appears in dataset
# stars_count = df.stars.value_counts()
# star_count_df = pd.DataFrame(stars_count).sort_index()

# Initialize bar chart object
#fig = px.bar(star_count_df, y=["stars"], barmode="group")

# # # Create the plot
# explore_graph_app = dcc.Graph(
#         id='explore_graph',
#         figure=fig
#     )


# Create plot adding each figure individually so the color of each marker can be changed

# trace_1 = go.Box(x=one_star_df['stars'], y=one_star_df['length'], marker_color='royalblue', boxmean='sd')
# trace_2 = go.Box(x=two_star_df['stars'], y=two_star_df['length'], marker_color='royalblue', boxmean='sd')
# trace_3 = go.Box(x=three_star_df['stars'], y=three_star_df['length'], marker_color='royalblue', boxmean='sd')
# trace_4 = go.Box(x=four_star_df['stars'], y=four_star_df['length'], marker_color='royalblue', boxmean='sd')
# trace_5 = go.Box(x=five_star_df['stars'], y=five_star_df['length'], marker_color='royalblue', boxmean='sd')

# data = [trace_1, trace_2, trace_3, trace_4, trace_5]
# box_layout = go.Layout(
#     title = "Length of Review per Star Rating"
# )

# fig2 = go.Figure(data=data, layout=box_layout)

# # Create the boxplot
# box_plot_graph = dcc.Graph(
#     figure=fig2
# )


# ## Tab one app for Graph ##
# tab1_app = dbc.Card(
#     dbc.CardBody(
#         [
#             explore_graph_app
#         ]
#     ),
#     className="mt-3",
#     id='first_tab',
# )

# ## Tab two app for Graph ##
# tab2_app = dbc.Card(
#     dbc.CardBody(
#         [
#             box_plot_graph
#         ]
#     ),
#     className="mt-3",
#     id='second_tab',
# )

# ## Parent tabs app ##
# parent_tab_app = dbc.Tabs(
#     [
#         dbc.Tab(tab1_app, label="Tab 1"),
#         dbc.Tab(tab2_app, label="Tab 2"),
#     ]
# )


## pie chart

# Read in pie chat data
unbalanced_pie = pd.read_csv('static/Resources/unbalanced_pie.csv')
balanced_pie_df = pd.read_csv('static/Resources/balanced_pie.csv')

# Set labels for pie charts
ub_labels = unbalanced_pie['type']
ub_values = unbalanced_pie['value']
# balanced_labels = balanced_pie_df['type']
balanced_labels = ['Positive words', 'Negative words', 'Neutral words']
balanced_values = balanced_pie_df['value']

# Create pie charts

# fig3 = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
# fig3.add_trace(go.Pie(labels=ub_labels, values=ub_values, hole=0.3, pull= [0.2, 0.2, 0], name="Unbalanced Singular Word Sentiment"),
#             1, 1)
# fig3.add_trace(go.Pie(labels=balanced_labels, values=balanced_values, hole=0.3, pull= [0.2, 0.2, 0], name="Balanced Singular Word Sentiment"),
#             1, 2)

"""Cards"""

# Card 1
card = dbc.Card(
    [
        dbc.CardImg(src="/static/images/ml.jpeg", top=True),
        dbc.CardBody(
            [
                html.P("We applied natural language processing (NLP) and machine learning techniques to identify sentiment to "
                "classify Yelp reviews for our project."),
                html.H4("Try it yourself!", className="card-title"),
                html.P(
                    "Type a review, then select a machine learning model, and see how it predicts your sentiment.",
                    className="card-text",
                ),
                html.Div(
                    [
                        dbc.DropdownMenu(
                            id= "model-dropdown",
                            label="Select a model",
                            children=
                                    [
                                        dbc.DropdownMenuItem("Naive Bayes", id="naive-button"),
                                        dbc.DropdownMenuItem("Logistic Regression", id="logistic-button")
                                    ],
                        ),
                        html.Br(),
                        dbc.Textarea(id="input_area", placeholder="Write a review...",),
                        html.Br(),
                        
                        dbc.Button("Predict sentiment", color="secondary", id="predict-button", ),
                                # dbc.Button("Clear prediction to try again", color="secondary", id="clear-button", style={'margin':'auto','width':'50%',
                                #     'padding-right':'10px'} )
                            
                    ],
                ),
                html.Br(),
                html.P(id="output"),        
            ]
        ),
    ],
    style={"width": "40rem"},
)

# Card 2
card_two = dbc.Card(
    [
        dbc.CardBody(
            
            [
                html.H4("Data Exploration", className="card-title"),
                html.P("Select a dataset to explore."),
                # dcc.Dropdown(
                #     id='dataset_dropdown',
                #     options=
                #             [
                #                 {'label': 'Unbalanced Dataset', 'value': 0},
                #                 {'label': 'Balanced Dataset', 'value': 1},
                #             ],
                #     label= 'Select a dataset',
                # ),
                dbc.DropdownMenu(
                    label="Select a dataset",
                    children=[
                        dbc.DropdownMenuItem("Unbalanced Dataset", id ="ub_dropdown"),
                        dbc.DropdownMenuItem("Balanced Dataset", id ="b_dropdown"),
                    ],
                ),
                ## Parent tabs app ##
                html.Br(),
                dbc.Tabs(
                    [
                        dbc.Tab(label="Star Distribution",
                            id='tab_one',
                            children= 
                                    [
                                        html.Div(
                                        dcc.Graph(id='tab_one_graph'),
                                        )
                                    ],  
                        ),             
                        dbc.Tab(label="Length vs Rating",
                            id='tab_two',
                            children=
                                    [
                                        html.Div(
                                        dcc.Graph(id='tab_two_graph')
                                        )
                                    ]
                        ),
                        dbc.Tab(label="Word Sentiment",
                            id='tab_three',
                            children=
                                    [
                                        html.Div(
                                        dcc.Graph(id='tab_three_graph')
                                        )
                                    ]
                        ),

                    ]
                ),
                html.Br(), 
                #html.Br(),                            
            ]
        ),
        # dbc.Button("Show graphs side by side", color="secondary", id="show-both", style={'margin':'left','width':'auto','padding-left':'10px'}),
        # dbc.Modal(
        #     [
        #         dbc.ModalHeader("Header"),
        #         dbc.ModalBody(id='mod-bod'),
        #         dbc.ModalFooter(
        #         dbc.Button("Close", id="close", className="ml-auto")
        #         ),
        #     ],
        #     id="modal",
        # ),
        #html.Br(),
    ],
    #style={"width": "40rem"},
)

# card_three = dbc.Card(
#     [
#         dbc.CardBody(
#             [
#                 html.Link(href="href='https://fonts.googleapis.com/css?family=Caudex", rel="stylesheet"),
#                 html.H3("Summary"),
#                 html.P("We applied natural language processing (NLP) and machine learning techniques to identify sentiment to "
#                 "classify Yelp reviews for our project.")
#             ], style={"font-family":"Caudex","font-size": "18px"}
#         ),
#     ], style={"width": "40rem"},
# )

"""Body"""
body = html.Div(
    [
        dbc.Card(
                    [
                        dbc.CardBody(
                            [
                        
                                dbc.Row([
                                    dbc.Col(html.Div(card)),
                                    dbc.Col(html.Div(card_two))
                                    ]
                                )
                            ]
                        )
                    ], style = {"padding-left":"10px", "padding-right":"10px"},
        ),
    ],style = {"padding-left":"10px", "padding-right":"10px"},
)

""" Final Layout Render"""
app.layout = html.Div(
    [navbar, body,]
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
@app.callback(
            Output("output", "children"), 
            [Input("input_area", "value"),
            Input("naive-button", "n_clicks_timestamp"),
            Input("logistic-button", "n_clicks_timestamp"),
            Input("predict-button", "n_clicks_timestamp"),
            Input("input_area", "n_clicks_timestamp")],
            
            )

def output_text(value, n1, n2, n3, n4):

    if n3 and n1 and not n2:
        if n3 > n1 and n3 > n4:
                 
            # Assign text area input to a variable
            dash_input = [value]

            # Transform input
            dash_input = tfidf.transform(dash_input)
            
            # User sentiment prediction for naive bayes
            user_prediction = mnb.predict(dash_input)
            if user_prediction == 1:
                user_prediction = 'The Naive Bayes model predicted your review was positive.'
            else:
                user_prediction = "The Naive Bayes model predicted your review was negative."
            #print(user_prediction)
            
            return user_prediction
    
    elif n3 and n2 and not n1:
        if n3 > n2 and n3 > n4:

            # Assign text area input to a variable
            dash_input = [value]

            # Transform input
            dash_input = tfidf.transform(dash_input)

            # User sentiment prediction for logistic regression
            user_prediction = logr.predict(dash_input)
            if user_prediction == 1:
                user_prediction = 'The Logistic Regression model predicted your review was positive.'
            else:
                user_prediction = "The Logistic Regression model predicted your review was negative."
            
            return user_prediction
    elif n3 and n2 and n1:
        if n3 > n2 and n3 > n1 and n3 > n4:
            if n2 > n1:
                # Assign text area input to a variable
                dash_input = [value]

                # Transform input
                dash_input = tfidf.transform(dash_input)

                # User sentiment prediction for logistic regression
                user_prediction = logr.predict(dash_input)
                if user_prediction == 1:
                    user_prediction = 'The Logistic Regression model predicted your review was positive.'
                else:
                    user_prediction = "The Logistic Regression model predicted your review was negative."
                
                return user_prediction
            
            if n1 > n2 : 
                # Assign text area input to a variable
                dash_input = [value]

                # Transform input
                dash_input = tfidf.transform(dash_input)
                
                # User sentiment prediction for naive bayes
                user_prediction = mnb.predict(dash_input)
                if user_prediction == 1:
                    user_prediction = 'The Naive Bayes model predicted your review was positive.'
                else:
                    user_prediction = "The Naive Bayes model predicted your review was negative."
                #print(user_prediction)
            
                return user_prediction
    else:
        return " "

@app.callback(
    Output("model-dropdown", "label"),
    [Input("naive-button", "n_clicks_timestamp"),
    Input("logistic-button", "n_clicks_timestamp"),],
)

def update_dropdown_logistic_label(n1, n2):
    if n2 and n1:
        if n1 < n2:
            label = 'Logistic'
        elif n2 < n1:
            label = 'Naive Bayes'
    elif n1 and not n2:
        label = 'Naive Bayes'
    elif n2 and not n1:
            label = 'Logistic'
    else:
        label = 'Select a model'
    return label

        # naive_true = False
    # logistic_true = False
    # ctx = dash.callback_context
    # if not ctx.triggered:
    #     return " "
    # # elif ctx.triggered and n3:
    # #     button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    # #     return value
    # elif ctx.triggered and n1 and n3:


    #     return value
    
    # elif ctx.triggered and n2 and n3:

    #     return value

# @app.callback(
    
#     [Output("predict-button", "n_clicks"),
#     Output("naive-button", "n_clicks"),],
#     [Input("clear-button", "n_clicks"),]
    
# )   

# def clear_clicks(n1):
#     return [0,0]

# Dataset Dropdown
@app.callback(
    [Output("tab_one_graph", "figure"),
    Output("tab_two_graph", "figure")],
    [Input("ub_dropdown", "n_clicks_timestamp"),
    Input("b_dropdown", "n_clicks_timestamp")],
)
def update_fig(n1, n2):
    if not n1 and not n2:
        dataset = 0
    if n1 and not n2:
        dataset = 0
    if n2 and not n1:
        dataset = 1
    if n1 and n2:
        if n1 > n2:
            dataset = 0
        else:
            dataset = 1


    int_dataset = int(dataset)
    dfs = []
    file_path1 = 'yelp.csv'
    file_path2 = 'uniform_yelp.csv'
    df1 = pd.read_csv(file_path1)
     # Create length column containing length of reviews
    df1['length'] = df1['text'].apply(len)
    
    # Drop extreme outliers
    df1 = df1.loc[df1['length'] < 3200]
    
    # Get the number of times each star rating appears in dataset
    new_stars_count = df1.stars.value_counts()
    new_star_count_df = pd.DataFrame(new_stars_count).sort_index()
    
    # Set up second Dataframe 
    df2 = pd.read_csv(file_path2)
    # Create length column containing length of reviews
    df2['length'] = df2['text'].apply(len)
    
    # Drop extreme outliers
    df2 = df2.loc[df2['length'] < 3200]
    
    # Get the number of times each star rating appears in dataset
    df2_new_stars_count = df2.stars.value_counts()
    df2_new_star_count_df = pd.DataFrame(df2_new_stars_count).sort_index()

    dfs.append(new_star_count_df)
    dfs.append(df2_new_star_count_df)

    # Create df for each star to be used in box plot
    tab_two_dfs = [df1, df2]
    df = tab_two_dfs[int_dataset]

    one_star_df = df.loc[df['stars'] == 1]
    two_star_df = df.loc[df['stars'] == 2]
    three_star_df = df.loc[df['stars'] == 3]
    four_star_df = df.loc[df['stars'] == 4]
    five_star_df = df.loc[df['stars'] == 5]

    # Create trace object for each star rating 
    trace_1 = go.Box(x=one_star_df['stars'], y=one_star_df['length'], marker_color='#54478c', boxmean='sd', name='1 star')
    trace_2 = go.Box(x=two_star_df['stars'], y=two_star_df['length'], marker_color='#048ba8', boxmean='sd', name='2 star')
    trace_3 = go.Box(x=three_star_df['stars'], y=three_star_df['length'], marker_color='#16db93', boxmean='sd', name='3 star')
    trace_4 = go.Box(x=four_star_df['stars'], y=four_star_df['length'], marker_color='#f29e4c', boxmean='sd', name='4 star')
    trace_5 = go.Box(x=five_star_df['stars'], y=five_star_df['length'], marker_color='#efea5a', boxmean='sd', name='5 star')
    
    # Assign trace objects to list 
    data = [trace_1, trace_2, trace_3, trace_4, trace_5]

    # Create box plot layout object
    box_layout = go.Layout(
    title = "Length of Review per Star Rating",
    xaxis_title="Star Rating",
    yaxis_title="Length of Review"
    )
    fig_data=[]
    ds = dfs[int_dataset]
    ds['x'] = ds.index
    fig = px.bar(ds, x='x', y="stars", barmode="group", labels={"x":"Star Rating", "stars":"Number of Reviews"}, color="x")
    fig2 = go.Figure(data=data, layout=box_layout)
    fig_data.append(fig2)
    #print(ds)
    return [fig, fig2]

# # Modal
# @app.callback(
#     Output("modal", "is_open"),
#     [Input("show-both", "n_clicks"),
#     Input("close", "n_clicks"),
#     Input("dataset_dropdown", "value")],
#     [State("modal", "is_open")],
# )
# def toggle_modal(n1, n2, value, is_open):
#     if n2 1:
#         return not is_open
#     elif: n2 and value
#     else:

#     return is_open

# Dataset Dropdown
@app.callback(
    Output("tab_three_graph", "figure"),
        [Input("ub_dropdown", "n_clicks_timestamp"),
        Input("b_dropdown", "n_clicks_timestamp")],
)
def show_pies(n1, n2):
    #fig3 = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
    if not n1 and not n2:
        fig3 = go.Figure(data=[go.Pie(labels=ub_labels, values=ub_values, pull= [0.2, 0.2, 0], name="Unbalanced")])
        fig3.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    
    if n1 and not n2:
        fig3 = go.Figure(data=[go.Pie(labels=ub_labels, values=ub_values, pull= [0.2, 0.2, 0], name="Unbalanced")])
        fig3.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    
    if n1 and n2:
        if n1 > n2:
            fig3 = go.Figure(data=[go.Pie(labels=ub_labels, values=ub_values, pull= [0.2, 0.2, 0], name="Unbalanced")])
            fig3.update_layout(margin=dict(t=0, b=0, l=0, r=0))

        if n2 > n1:
            fig3 = go.Figure(data=[go.Pie(labels=ub_labels, values=balanced_values, pull= [0.2, 0.2, 0], name="Balanced")])
            fig3.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    
    else:
        fig3 = go.Figure(data=[go.Pie(labels=ub_labels, values=balanced_values, pull= [0.2, 0.2, 0], name="Balanced")])
        fig3.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    print(n1)
    return fig3


# """End App Callback"""

if __name__ == '__main__':
    app.run_server(debug=True)