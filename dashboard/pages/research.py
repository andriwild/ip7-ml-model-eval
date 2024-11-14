from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px

def research_layout():

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
                    dbc.Col([html.A(html.I(className=f"fa fa-arrow-up-right-from-square me-2"), href="https://arxiv.org/abs/2409.16808", target="_blank")])
                    ], className="mb-5"),
                dbc.Row([
                    dbc.Col([html.H5("Automated Analysis for Urban Biodiversity Monitoring")], width=11),
                    dbc.Col([html.A(html.I(className=f"fa fa-arrow-up-right-from-square me-2"), href="https://wullt.ch/documents/p8.pdf", target="_blank")])
                    ], className="mb-5"),
                ]
            )


    return layout
