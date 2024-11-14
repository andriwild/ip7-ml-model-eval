
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px


def management_layout():
    layout = dbc.Container(
            style={"marginTop": "20px"},
            children=[
                dbc.Row([
                    dbc.Col([html.H1("Management")], className="mb-2"),
                    ]),
                dbc.Row([
                    dbc.Col([
                        html.H3("Projekt Board", className="d-inline-block me-3"),
                        html.A(html.I(className=f"fa fa-arrow-up-right-from-square"), href="https://github.com/users/andriwild/projects/2/views/1", target="_blank", className="mb-1"),
                        ]),
                    ], className="mb-5"),

                dbc.Row([
                    dbc.Col([
                        html.H3("Planning"),
                        html.Img(src="static/images/time_planning.png", className="mx-auto d-block", style={'width': '60%', 'height': 'auto'}),
                        ])
                    ]),
                ]
            )
    return layout
