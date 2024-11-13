from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px

def research_layout(devices_df, camera_df):

    device_colors = px.colors.qualitative.Pastel

    layout = dbc.Container(
            style={"marginTop": "20px"},
            children=[
                dbc.Row([
                    dbc.Col([html.H1("Research")], className="mb-2")
                    ]),
                dbc.Row([
                    dbc.Col([html.H3("Paper")], className="mb-2")
                    ]),
                dbc.Row([
                    dbc.Col([html.H5("Benchmarking Deep Learning Models for Object Detection on Edge Computing Devices")], width=11),
                    dbc.Col([html.A("Link", href="https://arxiv.org/abs/2409.16808")])
                    ], className="mb-5"),
                ]
            )

    return layout
