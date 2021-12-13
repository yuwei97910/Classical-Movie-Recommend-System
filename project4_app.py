########################################################################
# STAT 542: Project 4
# Movies Recommendation System

### YuWei Lai
### 2021.12.12
########################################################################

#%%
from os import system
import pandas as pd
import numpy as np
import pickle

import dash
import dash_table
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import plotly.express as px
import plotly.graph_objects as go

import flask

########################################################################
# Use Surprise for recommendation
from surprise import KNNBasic
from surprise import SVD
from surprise import Reader
from surprise import Dataset
from surprise import accuracy
from collections import defaultdict
from dash.exceptions import PreventUpdate
reader = Reader(rating_scale=(1, 5))

########################################################################
##########
# Set the path to image
image_url = "https://liangfgithub.github.io/MovieImages/"
static_image_route = '/static/'

# Import the pre-labled dataset for system 1:
movies_system_1 = pd.read_csv('movies_system_1.csv')
movies_system_1['Image'] = [image_url + str(x) + '.jpg?raw=true' for x in movies_system_1['MovieID']]

# load the pickled models
# filename = 'model_knn.pk'
# with open(filename, 'rb') as f:
#     model_knn = pickle.load(f)

filename = 'model_svd.pk'
with open(filename, 'rb') as f:
    model_svd = pickle.load(f)

##########
# For deploy
external_stylesheets = [dbc.themes.YETI]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

##########
df = px.data.stocks() 

colors = {
    'h1': '#222222',
    'h2': '#2E5481',
    'h3': '#151515',
    'text': '#151515'
}
########################################################################
########################################################################
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f8f8",
}
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

IMAGE_ALIGN_STYLE = {
    'align':'bottom', 
    'display': 'block', 
    'width': '100%', 
    'height':'100%'
}

H3_BOLD_STYLE = {'textAlign':'left', 'marginTop':10, 'marginBottom':20,'font-weight': 'bold', 'color': colors['h3']}

BUTTON_STYLE = {'font-size': '16px;', 'border-radius': '4px;', 'background-color':'e7e7e7', 'color': '151515'}
IMORTANT_BUTTON_STYLE = {'font-size': '24px;', 'border-radius': '4px;', 'background-color':'#6A8EB7', 'color': 'white'}

TEXT_ALIGN_STYLE_1 = {'textAlign':'Bottom', 'marginTop':5, 'marginBottom':10, 'height':'50px'}
TEXT_ALIGN_STYLE_2 = {'display': 'block', 'textAlign':'Bottom', 'marginTop':10, 'marginBottom':5, 'height':'60px'}

########################################################################

########################################################################
def initiate_rating_section(movies_system_1=movies_system_1):
    n = 4
    movies_sel = movies_system_1.sample(n=n)
    
    movies_sel = movies_sel.loc[:,['MovieID', 'Title', 'Image']]
    movies_sel['Rating'] = [0] * n

    presenting_df = pd.DataFrame()
    presenting_df['Order'] = list(range(0, n))
    presenting_df['MovieID'] = [x for x in movies_sel['MovieID']]

    movie_store_df = movies_sel.to_json(date_format='iso', orient='split')
    presenting_df = presenting_df.to_json(date_format='iso', orient='split')
    data_dict = {'movie_store_df': movie_store_df, 'presenting_df': presenting_df}

    children = [
        html.H3(id='rating_title', children='Step 1: Please rate the movies as more as possible', style=H3_BOLD_STYLE),
        html.Div(children = [
            html.Table([
                html.Tr(children=[
                    html.Th([html.Tr(movies_sel.iloc[0, 1], id='rating_t_0', style=TEXT_ALIGN_STYLE_2),
                        html.Th(html.Img(id='rating_i_0', src=movies_sel.iloc[0, 2], style=IMAGE_ALIGN_STYLE), style={'vertical-align':'bottom'}), # Col 2: Image
                    ]),
                    html.Th([html.Tr(movies_sel.iloc[1, 1], id='rating_t_1', style=TEXT_ALIGN_STYLE_2),
                        html.Th(html.Img(id='rating_i_1', src=movies_sel.iloc[1, 2], style=IMAGE_ALIGN_STYLE), style={'vertical-align':'bottom'}), # Col 2: Image
                    ]),
                    html.Th([html.Tr(movies_sel.iloc[2, 1], id='rating_t_2', style=TEXT_ALIGN_STYLE_2),
                        html.Th(html.Img(id='rating_i_2', src=movies_sel.iloc[2, 2], style=IMAGE_ALIGN_STYLE), style={'vertical-align':'bottom'}), # Col 2: Image
                    ]),
                    html.Th([html.Tr(movies_sel.iloc[3, 1], id='rating_t_3', style=TEXT_ALIGN_STYLE_2),
                        html.Th(html.Img(id='rating_i_3', src=movies_sel.iloc[3, 2], style=IMAGE_ALIGN_STYLE), style={'vertical-align':'bottom'}), # Col 2: Image
                    ]),
                ]),
                html.Tr(children=[
                    html.Th(
                        dcc.Slider(
                            id='rating_slider_0',
                            min=1,
                            max=5,
                            step=1,
                            value=0,
                            marks={
                                1: {'label': '1','style':{'color': '#151515'}},
                                2: {'label': '2','style':{'color': '#151515'}},
                                3: {'label': '3','style':{'color': '#151515'}},
                                4: {'label': '4','style':{'color': '#151515'}},
                                5: {'label': '5', 'style':{'color': '#151515'}}
                            },
                            tooltip={"placement": "top", "always_visible": True},
                        )
                    ),
                    html.Th(
                        dcc.Slider(
                            id='rating_slider_1',
                            min=1,
                            max=5,
                            step=1,
                            value=0,
                            marks={
                                1: {'label': '1','style':{'color': '#151515'}},
                                2: {'label': '2','style':{'color': '#151515'}},
                                3: {'label': '3','style':{'color': '#151515'}},
                                4: {'label': '4','style':{'color': '#151515'}},
                                5: {'label': '5', 'style':{'color': '#151515'}}
                            },
                            tooltip={"placement": "top", "always_visible": True},
                        )
                    ),
                    html.Th(
                        dcc.Slider(
                            id='rating_slider_2',
                            min=1,
                            max=5,
                            step=1,
                            value=0,
                            marks={
                                1: {'label': '1','style':{'color': '#151515'}},
                                2: {'label': '2','style':{'color': '#151515'}},
                                3: {'label': '3','style':{'color': '#151515'}},
                                4: {'label': '4','style':{'color': '#151515'}},
                                5: {'label': '5', 'style':{'color': '#151515'}}
                            },
                            tooltip={"placement": "top", "always_visible": True},
                        )
                    ),
                    html.Th(
                        dcc.Slider(
                            id='rating_slider_3',
                            min=1,
                            max=5,
                            step=1,
                            value=0,
                            marks={
                                1: {'label': '1','style':{'color': '#151515'}},
                                2: {'label': '2','style':{'color': '#151515'}},
                                3: {'label': '3','style':{'color': '#151515'}},
                                4: {'label': '4','style':{'color': '#151515'}},
                                5: {'label': '5', 'style':{'color': '#151515'}}
                            },
                            tooltip={"placement": "top", "always_visible": True},
                        )
                    ),
                ]),
            ], style={'align':'left', 'table-layout': 'fixed', 'width': '100%', 'height':'100%'}),
        ]),
        html.Div(children = [
            html.Button('Skip these movies', id='skip_set_button', n_clicks=0, style=BUTTON_STYLE)
            ], style={'textAlign':'Bottom', 'marginTop':40, 'marginBottom':5, 'height':'60px'}),
        html.Hr(),
        dcc.Store(id='stored_df', data=data_dict),
    ]
    return children



########################################################################
def rating_p_update(p, value, data, movies_system_1=movies_system_1):
    movie_store_df = data['movie_store_df']
    presenting_df = data['presenting_df']
    movie_store_df = pd.read_json(movie_store_df, orient='split')

    presenting_df = pd.read_json(presenting_df, orient='split')
    now_id = presenting_df.iloc[p, 1] # the position p MovieID

    movie_store_df.loc[movie_store_df.MovieID == now_id, 'Rating'] = value

    movies_sel = movies_system_1.sample(n=1)
    new_id = movies_sel['MovieID'].values
    while new_id in movie_store_df.loc[:, 'MovieID'].tolist():
        movies_sel = movies_system_1.sample(n=1)
        new_id = movies_sel['MovieID'].values

    movies_sel = movies_sel.loc[:,['MovieID', 'Title', 'Image']]
    movie_store_df = movie_store_df.append(movies_sel)
    presenting_df.iloc[p, 1] = new_id # the position p MovieID

    # print(movie_store_df.loc[:,['Title', 'Rating']])

    movie_store_df = movie_store_df.to_json(date_format='iso', orient='split')
    presenting_df = presenting_df.to_json(date_format='iso', orient='split')
    data_dict = {'movie_store_df': movie_store_df, 'presenting_df': presenting_df}

    return data_dict, movies_sel.iloc[0, 1], movies_sel.iloc[0, 2]  # movie_sel had only one row !!!

@app.callback([
    
    dash.dependencies.Output('stored_df', 'data'),

    dash.dependencies.Output('rating_t_0', 'children'),
    dash.dependencies.Output('rating_i_0', 'src'),
    dash.dependencies.Output('rating_slider_0', 'value'),

    dash.dependencies.Output('rating_t_1', 'children'),
    dash.dependencies.Output('rating_i_1', 'src'),
    dash.dependencies.Output('rating_slider_1', 'value'),

    dash.dependencies.Output('rating_t_2', 'children'),
    dash.dependencies.Output('rating_i_2', 'src'),
    dash.dependencies.Output('rating_slider_2', 'value'),

    dash.dependencies.Output('rating_t_3', 'children'),
    dash.dependencies.Output('rating_i_3', 'src'),
    dash.dependencies.Output('rating_slider_3', 'value'),
    ],
    [dash.dependencies.Input('stored_df', 'data'),
    dash.dependencies.Input('skip_set_button', 'n_clicks'),

    dash.dependencies.Input('rating_t_0', 'children'),
    dash.dependencies.Input('rating_i_0', 'src'),
    dash.dependencies.Input('rating_slider_0', 'value'),

    dash.dependencies.Input('rating_t_1', 'children'),
    dash.dependencies.Input('rating_i_1', 'src'),
    dash.dependencies.Input('rating_slider_1', 'value'),

    dash.dependencies.Input('rating_t_2', 'children'),
    dash.dependencies.Input('rating_i_2', 'src'),
    dash.dependencies.Input('rating_slider_2', 'value'),

    dash.dependencies.Input('rating_t_3', 'children'),
    dash.dependencies.Input('rating_i_3', 'src'),
    dash.dependencies.Input('rating_slider_3', 'value'),
    ])
def rating_0_update(stored_df, skip_nclicks, t_0, i_0, s_0, t_1, i_1, s_1, t_2, i_2, s_2, t_3, i_3, s_3, movies_system_1=movies_system_1):
    ctx = dash.callback_context
    click_result = ctx.triggered[0]['prop_id'].split('.')[0]
    print(click_result)

    if click_result == 'rating_slider_0':
        p = 0
        stored_df, t_p, i_p = rating_p_update(p, s_0, stored_df, movies_system_1)
        return stored_df, t_p, i_p, 0, t_1, i_1, 0, t_2, i_2, 0, t_3, i_3, 0

    elif click_result == 'rating_slider_1':
        p = 1
        stored_df, t_p, i_p = rating_p_update(p, s_1, stored_df, movies_system_1)
        return stored_df, t_0, i_0, 0, t_p, i_p, 0, t_2, i_2, 0, t_3, i_3, 0

    elif click_result == 'rating_slider_2':
        p = 2
        stored_df, t_p, i_p = rating_p_update(p, s_2, stored_df, movies_system_1)
        return stored_df, t_0, i_0, 0, t_1, i_1, 0, t_p, i_p, 0, t_3, i_3, 0

    elif click_result == 'rating_slider_3':
        p = 3
        stored_df, t_p, i_p = rating_p_update(p, s_2, stored_df, movies_system_1)
        return stored_df, t_0, i_0, 0, t_1, i_1, 0, t_2, i_2, 0, t_p, i_p, 0

    elif click_result == 'skip_set_button':
        stored_df, t_0, i_0 = rating_p_update(0, 0, stored_df, movies_system_1)
        stored_df, t_1, i_1 = rating_p_update(0, 0, stored_df, movies_system_1)
        stored_df, t_2, i_2 = rating_p_update(0, 0, stored_df, movies_system_1)
        stored_df, t_3, i_3 = rating_p_update(0, 0, stored_df, movies_system_1)
        return stored_df, t_0, i_0, 0, t_1, i_1, 0, t_2, i_2, 0, t_3, i_3, 0

########################################################################
def get_top_n(predictions, n=20):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

@app.callback(dash.dependencies.Output('result_display', 'children'),
    [dash.dependencies.Input('get_recommend', 'n_clicks'), 
    dash.dependencies.Input('stored_df', 'data')])
def display_system_2_result(n_clicks, stored_df, movies_system_1=movies_system_1):
    ctx = dash.callback_context
    click_result = ctx.triggered[0]['prop_id'].split('.')[0]
    print(click_result)

    if click_result != 'get_recommend':
        return html.Div(html.H3('Try New Recommendation ?', style={'marginTop':30, 'marginBottom':20}))

    movie_store_df = stored_df['movie_store_df']
    presenting_df = stored_df['presenting_df']

    uid = 'User'
    user_test = pd.read_json(movie_store_df, orient='split')
    user_test['UserID'] = uid
    user_test.fillna(0, inplace=True)

    # For Summary
    columns = [{'name': col, 'id': col} for col in user_test.loc[:,['Title','Rating']].columns]
    data = user_test.loc[:,['Title','Rating']].to_dict(orient='records')

    # Get the prediction by the pre-trained model
    user_test = Dataset.load_from_df(user_test[['UserID', 'MovieID', 'Rating']], reader)
    user_test = user_test.build_full_trainset()
    user_test = user_test.build_testset()

    user_pred = model_svd.test(user_test)
    recommend_result = get_top_n(user_pred, n=10)
    result_id = [iid for (iid, _) in recommend_result[uid]]

    sel_recommend_movie = movies_system_1[movies_system_1["MovieID"].isin(result_id)]
    print(result_id)
    if len(result_id) <=5:
        children=[
            html.Div(children=[
                html.H3("Top recommendations for you!", style={'marginTop':30, 'marginBottom':20}),
                html.Table([html.Th([
                        html.Tr(str('{}'.format(x['Title'])), style={'marginTop':30}), 
                        html.Th(html.Img(src=x['Image'], style=IMAGE_ALIGN_STYLE), style={'vertical-align':'bottom'})
                    ])for i, x in sel_recommend_movie.iterrows()],
                style={'table-layout': 'auto', 'width': '100%', 'height':'100%'}),
                ]),
                html.Hr(style={'marginTop':60}),
                html.H6('Based on these movies you rated:', style=H3_BOLD_STYLE),
                dash_table.DataTable(data=data, columns=columns)
            ]
    else:
        children=[
            html.Div(children=[
                html.H3("Top recommendations for you!", style={'marginTop':30, 'marginBottom':20}),
                html.Table([html.Th([
                        html.Tr(str('{}'.format(x['Title'])), style={'textAlign':'Bottom', 'marginTop':10, 'marginBottom':5, 'height':'60px'}), 
                        html.Th(html.Img(src=x['Image'], style=IMAGE_ALIGN_STYLE), style={'vertical-align':'bottom'})
                    ])for i, x in sel_recommend_movie.iloc[0:5,].iterrows()],
                style={'table-layout': 'auto', 'width': '100%', 'height':'100%'}),
                ]),
            html.Hr(),
            html.Div(children=[
                html.Table([html.Th([
                        html.Tr(str('{}'.format(x['Title'])), style={'textAlign':'Bottom', 'marginTop':10, 'marginBottom':5, 'height':'60px'}), 
                        html.Th(html.Img(src=x['Image'], style=IMAGE_ALIGN_STYLE), style={'vertical-align':'bottom'})
                    ])for i, x in sel_recommend_movie.iloc[5:,].iterrows()],
                style={'table-layout': 'auto', 'width': '100%', 'height':'100%'}),
                ]),
            html.Hr(style={'marginTop':60}),
            html.H6('Based on these movies you rated:', style=H3_BOLD_STYLE),
            dash_table.DataTable(data=data, columns=columns)
            ]
    return children

########################################################################
# -----------
# Update the classical recommendations
@app.callback(dash.dependencies.Output(component_id='recommend_table', component_property='children'),
    [dash.dependencies.Input(component_id='genre_dropdown', component_property= 'value')])
def classic_recommend_update(genre_dropdown, movies_system_1=movies_system_1, image_url=image_url):
    input_genre = [genre_dropdown]
    sel_row = [True if x in r else False for x in input_genre for r in movies_system_1['Genres']]
    movies_system_1_sel = movies_system_1.loc[sel_row, :].copy()

    ### Output for classic movies
    sel_class_movie = movies_system_1_sel.sort_values(by=['classic_score'], ascending=False).iloc[0:5,:].loc[:,['MovieID', 'Title', 'Image']]
    sel_trend_movie = movies_system_1_sel.sort_values(by=['trending_score'], ascending=False).iloc[0:5,:].loc[:,['MovieID', 'Title', 'Image']]

    return html.Div(children=[
        html.Div(children=[
            html.H3("### Top 5 Classic Movies", style=H3_BOLD_STYLE),
            html.Table([html.Th([
                                html.Tr(x['Title'], style=TEXT_ALIGN_STYLE_1), 
                                html.Th(html.Img(src=x['Image'], style=IMAGE_ALIGN_STYLE), style={'vertical-align':'bottom'})
                            ])for i, x in sel_class_movie.iterrows()],
                        style={'table-layout': 'auto', 'width': '100%', 'height':'100%'}),
            html.Hr(),
            html.H3("### Top 5 Trending Movies", style=H3_BOLD_STYLE),
            html.Table([html.Th([
                                html.Tr(x['Title'], style=TEXT_ALIGN_STYLE_1), 
                                html.Th(html.Img(src=x['Image'], style=IMAGE_ALIGN_STYLE), style={'vertical-align':'bottom'})
                            ])for i, x in sel_trend_movie.iterrows()],
                        style={'table-layout': 'auto', 'width': '100%', 'height':'100%'}),
            ]),
        ])

@app.server.route('{}<image_path>.png'.format(static_image_route))
def serve_image(image_path):
    image_name = '{}'.format(image_path)
    # if image_name not in list_of_images:
    #     raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
    return flask.send_from_directory(image_path, image_name)

########################################################################
system_1 = html.Div(id = 'system_1', children = [
    
    html.H2(children='The Recommender System 1:  Classics and New Trends', style={
        'textAlign': 'left',
        'font-weight': 'bold',
        'color': colors['h2'],
        'marginTop':5, 'marginBottom':10
    }),
    html.H3(children='Please select your favorite movie genre.', style={
        'textAlign': 'left',
        'color': colors['h3'], 'marginTop':20, 'marginBottom': 20
    }),
    dcc.Dropdown(id = 'genre_dropdown', 
        options = [
            ##### ["Children's", 'Film-Noir', 'Drama', 'Thriller', 'Action', 'War', 
            # 'Documentary', 'Sci-Fi', 'Animation', 'Western', 
            # 'Mystery', 'Fantasy', 'Musical', 'Comedy', 'Crime', 'Adventure', 'Horror', 'Romance']
            {'label': "Children's Movies", 'value': "Children's"},
            {'label': 'Film-Noir', 'value': 'Film-Noir'},
            {'label': 'Drama', 'value': 'Drama'},
            {'label': 'Thriller', 'value': 'Thriller'},
            {'label': 'Action', 'value': 'Action'},
            {'label': 'War', 'value': 'War'},
            {'label': 'Documentary', 'value': 'Documentary'},
            {'label': 'Sci-Fi', 'value': 'Sci-Fi'},
            {'label': 'Animation', 'value': 'Animation'},
            {'label': 'Western', 'value': 'Western'},
            {'label': 'Mystery', 'value': 'Mystery'},
            {'label': 'Fantasy', 'value': 'Fantasy'},
            {'label': 'Musical', 'value': 'Musical'},
            {'label': 'Comedy', 'value': 'Crime'},
            {'label': 'Adventure', 'value': 'Adventure'},
            {'label': 'Horror', 'value': 'Horror'},
            {'label': 'Romance', 'value': 'Romance'},
            ],
        value = "Children's"),

    html.Div(id='recommend_table'),
    html.Div(id='trend_table'),
    ], style=CONTENT_STYLE)

# -----------
system_2_topic = html.Div(id='syste_2_topic',children=[
        html.H2('The Recommender System 2: User based Recommendations', style={
        'textAlign': 'left',
        'font-weight': 'bold',
        'color': colors['h2']}),
    ], style=CONTENT_STYLE)
rating_section = html.Div(id='rating_section', children=initiate_rating_section(), style=CONTENT_STYLE)
result_display_section = html.Div(id='result_display_section', children=[
        html.H3(id='present_title', children='Step 2: Get your recommendations', style=H3_BOLD_STYLE),
        html.Button('Get recommendations!', id='get_recommend', n_clicks=0, style=IMORTANT_BUTTON_STYLE),
        html.Div(id='result_display', children=''),
    ], style=CONTENT_STYLE)

# -----------
sidebar = html.Div([
        html.H2("Movies Recommend System", className="sidebar"),
        html.Hr(),
        html.H4("STAT542: Project 4, Fall 2021"),
        dbc.Nav([
                # html.Div(id="multitab_layout", children=multitab_layout),
                # dbc.NavLink("Home", href="/", active="exact"),
                # dbc.NavLink("Page 1", href="/page-1", active="exact"),
                # dbc.NavLink("Page 2", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
        html.H6("Created by Yu-Wei Lai", style = {'font-weight': 'bold'}),
        html.H4("Description", style = {'font-weight': 'bold', 'marginTop':20, 'marginBottom':20}),
        html.P('There are two recommender systems in the app. The first system will recommend movies based on the genre you selected. The system two is based on the movies you rated. Therefore, when using system two, you need to rate several movies which are randomly generated.'),
        html.H4("Instruction", style = {'font-weight': 'bold', 'marginTop':20, 'marginBottom':20}),
    ],style=SIDEBAR_STYLE)
# -----------
app.layout = html.Div([
    dcc.Location(id="url"), 
    html.Div(id='main_content', children=[
            html.H1(id = 's1_h1', children = 'Movies Recommender System', style = {'textAlign':'center','font-weight': 'bold', 'marginTop':20, 'marginBottom':20, 'block-size': '2em', 'background-color':'#6A8EB7', 'color':'#F0EDEC', 'padding': '10px 0', "margin-left": "4rem"}),
            dcc.Tabs(id='system_tabs', children=[
                dcc.Tab(label='Recommendation System 1', children = system_1, style={'font-weight': 'bold'}),
                dcc.Tab(label='Recommendation System 2', children = [system_2_topic, rating_section, result_display_section], style={'font-weight': 'bold'}),
            ], style=CONTENT_STYLE),
        ]),
    sidebar,
    ])

########################################################################
# Main
if __name__ == '__main__': 
    app.run_server(port=8000, host='127.0.0.1')

    ### For deploy
    # app.run_server(debug=True)
# %%
