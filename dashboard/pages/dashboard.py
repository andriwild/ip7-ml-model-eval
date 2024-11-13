from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def dashboard_layout(inference_df):
    # Get unique values for controls
    devices = inference_df['device'].unique()
    accelerators = inference_df['accelerator'].unique()
    
    models = inference_df['model'].unique()
    frameworks = inference_df['framework'].unique()
    batch_sizes = inference_df['batch_size'].unique()

    layout = dbc.Container(
            style={"marginTop": "20px"},
            children=[
                dbc.Row([
                    dbc.Col([
                        html.Label('Devices', className="h6"),
                        dbc.Checklist(
                            id='device-checklist',
                            options=[{'label': dev, 'value': dev} for dev in devices],
                            value=list(devices),
                            inline=False,
                            persistence=True,
                            style={"marginBottom": "20px"}
                            ),
                        ], width=1),
                    dbc.Col([
                        html.Label('Accelerators', className="h6"),
                        dbc.Checklist(
                            id='accelerator-checklist',
                            options=[{'label': html.Div(acc, style={"marginLeft": "10px"}) if acc else 'None', 'value': acc} for acc in accelerators],
                            value=list(accelerators),
                            inline=False,
                            persistence=True,
                            labelStyle={"display": "flex", "align-items": "center"},
                            ),
                        ], width=2),
                    dbc.Col([
                        html.Label('Models', className="h6"),
                        dbc.Stack(
                            direction="horizontal",
                            style={"alignItems": "baseline"},
                            gap=4,
                            children=[
                                dbc.Checklist(
                                    id='model-dropdown-v5',
                                    options=[{'label': mod, 'value': mod} for mod in models if "5" in mod],
                                    value=[m for m in models if "5" in m],
                                    persistence=True,
                                    ),
                                dbc.Checklist(
                                    id='model-dropdown-v8',
                                    options=[{'label': mod, 'value': mod} for mod in models if "8" in mod],
                                    value=[m for m in models if "8" in m],
                                    persistence=True,
                                    ),
                                dbc.Checklist(
                                    id='model-dropdown-v10',
                                    options=[{'label': mod, 'value': mod} for mod in models if "10" in mod],
                                    value=[m for m in models if "10" in m],
                                    persistence=True,
                                    ),
                                ])

                        ], width=5),
                    dbc.Col([
                        html.Label('Frameworks', className="h6"),
                        dbc.Checklist(
                            id='framework-checklist',
                            options=[{'label': html.Div(fw, style={"marginLeft": "10px"}), 'value': fw} for fw in frameworks],
                            value=list(frameworks),
                            inline=False,
                            persistence=True,
                            labelStyle={"display": "flex", "align-items": "center"},
                            ),
                        ], width=2),
                    dbc.Col([
                        html.Label('Batch Sizes', className="h6"),
                        dbc.RadioItems(
                            id='batch-radio',
                            options=[{'label': html.Div(str(bs), style={"marginLeft": "10px"}), 'value': bs} for bs in sorted(batch_sizes)],
                            value=batch_sizes[0],
                            persistence=True,
                            labelStyle={"display": "flex", "align-items": "center"},
                            ),
                        ], width=2),
                    ], className="mb-4"),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='performance-bar-chart', style={'height': '600px'})
                ])
            ]),
        dbc.Row([
            dbc.Col([
                html.Label('Unit', className="h6"),
                dbc.RadioItems(
                    id='unit-radio',
                    options=[{'label': html.Div(unit, style={"marginLeft": "10px"}), 'value': unit} for unit in ['time', 'fps']],
                    value="time",
                    inline=True,
                    labelStyle={"display": "flex", "align-items": "center"},
                    ),
                ], width=2),
            ])
    ])
    return layout


