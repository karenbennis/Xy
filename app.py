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

# Navbar image
yelp_logo = "static/images/yelp_logo.png"

# Link to projects master GitHub repo
nav_item = dbc.NavItem(dbc.NavLink('GitHub', href='https://github.com/karenbennis/Xy'))

# Dropdown menu with links to our portfolios
dropdown = dbc.DropdownMenu(children=
    [
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
                    # Tag to display navbar logo
                    html.Img(src=yelp_logo, height="70px"),
                    # Navbar header with project name
                    dbc.NavbarBrand("NLP Sentiment Analysis of Yelp Reviews"),
                ],
                align="center",
                no_gutters=True,
                className='ml-auto'
            ),
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

"""Cards"""

# Card 1
card = dbc.Card(
    [
        # Add machine learning image
        dbc.CardImg(src="/static/images/ml.jpeg", top=True),
        
        # Body of card
        dbc.CardBody(
            [
                # Paragraph
                html.P("We applied natural language processing (NLP) and machine learning techniques to identify sentiment to "
                "classify Yelp reviews for our project."),
                # Header
                html.H4("Try it yourself!", className="card-title"),
                # Paragraph
                html.P(
                    "Select a machine learning model, type a review, and click 'predict sentiment' to see the model classifies your review."
                    " Get Creative!",
                    className="card-text",
                ),
                # div containing drop down menu and menu items
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
                    ],
                ),
                html.Br(),
                html.P(id="output"),        
            ]
        ),
    ],
    style={"width": "40rem"},
)

# Card 2 - to be used on right side of screen(Data Exploration)
card_two = dbc.Card(
    [
        dbc.CardBody(
            
            [
                html.H4("Data Exploration", className="card-title"),
                html.P("From the dropdown menu, select a dataset to explore. Each tab will update dynamically and"
                    " display insights gleaned from the data exploratoin process."),                
                # Dropdown component for dataset selection options
                dbc.DropdownMenu(
                    id="dataset-dropdown-menu",
                    label="Select a dataset",
                    children=[
                        dbc.DropdownMenuItem("Unbalanced Dataset", id ="ub_dropdown"),
                        dbc.DropdownMenuItem("Balanced Dataset", id ="b_dropdown"),
                    ],
                ),
                ## Parent tabs component ##
                html.Br(),
                dbc.Tabs(
                    [
                        # Tab component to display bar graph
                        dbc.Tab(label="Star Distribution",
                            id='tab_one',
                            children= 
                                    [
                                        html.Div(
                                            # Bar Graph, returned by callback
                                            dcc.Graph(id='tab_one_graph'),
                                        )
                                    ],  
                        ), 
                        # Tab component to display boxplot           
                        dbc.Tab(label="Length vs Rating",
                            id='tab_two',
                            children=
                                    [
                                        html.Div(
                                        # Box plot, returned by callback
                                        dcc.Graph(id='tab_two_graph')
                                        )
                                    ]
                        ),
                        # Tab component to display pie charts
                        dbc.Tab(label="Word Sentiment",
                            id='tab_three',
                            children=
                                    [
                                        html.Div(
                                            # Pie chart, returned by callback
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
    ],
    #style={"width": "40rem"},
)

"""Body"""

# This is where page components are combined and added to a body object to pass to the app layout 
# function along with with the navbar
# navbar
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
                
            
                return user_prediction
    else:
        return " "

# Callback to update model selection dropdown label
@app.callback(
    Output("model-dropdown", "label"),
    [Input("naive-button", "n_clicks_timestamp"),
    Input("logistic-button", "n_clicks_timestamp"),],
)

def update_dropdown_logistic_label(n1, n2):
    if n2 and n1:
        if n1 < n2:
            label = 'Logistic Regression'
        elif n2 < n1:
            label = 'Naive Bayes'
    elif n1 and not n2:
        label = 'Naive Bayes'
    elif n2 and not n1:
            label = 'Logistic Regression'
    else:
        label = 'Select a model'
    return label

# Callback to update dataset dropdown label
@app.callback(
    Output("dataset-dropdown-menu", "label"),
    [Input("ub_dropdown", "n_clicks_timestamp"),
    Input("b_dropdown", "n_clicks_timestamp")],
)

def update_dataset_dropdown_label(n1, n2):
    if n2 and n1:
        if n1 < n2:
            label = 'Balanced Dataset'
        elif n2 < n1:
            label = 'Unbalanced Dataset'
    elif n1 and not n2:
        label = 'Unbalanced Dataset'
    elif n2 and not n1:
            label = 'Balanced Dataset'
    else:
        label = 'Select a model'
    return label

# Callback to dynamically update bar chart and boxplot in card 2
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
        fig3.update_layout(margin=dict(t=1, b=0, l=0, r=0))
    
    if n1 and not n2:
        fig3 = go.Figure(data=[go.Pie(labels=ub_labels, values=ub_values, pull= [0.2, 0.2, 0], name="Unbalanced")])
        fig3.update_layout(margin=dict(t=1, b=0, l=0, r=0))
    
    if n1 and n2:
        if n1 > n2:
            fig3 = go.Figure(data=[go.Pie(labels=ub_labels, values=ub_values, pull= [0.2, 0.2, 0], name="Unbalanced")])
            fig3.update_layout(margin=dict(t=1, b=0, l=0, r=0))

        if n2 > n1:
            fig3 = go.Figure(data=[go.Pie(labels=ub_labels, values=balanced_values, pull= [0.2, 0.2, 0], name="Balanced")])
            fig3.update_layout(margin=dict(t=1, b=0, l=0, r=0))
    
    else:
        fig3 = go.Figure(data=[go.Pie(labels=ub_labels, values=balanced_values, pull= [0.2, 0.2, 0], name="Balanced")])
        fig3.update_layout(margin=dict(t=1, b=0, l=0, r=0))
    print(n1)
    return fig3


# """End App Callback"""

if __name__ == '__main__':
    app.run_server(debug=True)