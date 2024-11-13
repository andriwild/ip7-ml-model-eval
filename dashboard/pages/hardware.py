from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px

def hardware_layout(devices_df, camera_df):

    device_colors = px.colors.qualitative.Pastel
    devices_df = devices_df.sort_values('cost')
    camera_df = camera_df.sort_values('cost')

    layout = dbc.Container(
            style={"marginTop": "20px"},
            children=[
                dbc.Row([
                    dbc.Col([html.H1("Hardware")], className="mb-2")
                    ]),
                dbc.Row([
                    dbc.Col(html.H3("Edge Devices"), className="mb-2")
                    ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(
                            id='fps-bar-chart',
                            figure=px.bar(
                                devices_df,
                                x='name',
                                y='cost',
                                color='name',
                                labels={'cost': 'CHF', 'name': 'Name'},
                                title='Cost of Edge Devices',
                                color_discrete_sequence=device_colors
                                ).update_layout(template='ggplot2')
                            )
                        ])
                    ], className="mb-5"),
                dbc.Row([
                    dbc.Col([
                        dbc.ListGroup(
                            [
                                dbc.ListGroupItem(
                                    [
                                        html.Div(
                                            [
                                                html.H5(device["name"], className="mb-1"),
                                                html.Small(f"{str(device['cost'])} CHF", className="text-muted"),
                                                ],
                                            className="d-flex w-100 justify-content-between",
                                            ),
                                        html.A("link", href=device["link"], target="_blank", className="mb-1"),
                                        ]
                                    )
                                 for device in devices_df.to_dict("records")]
                            )
                        ])
                    ], className="mb-5"),

                dbc.Row(dbc.Col([html.H5("Erkenntnisse")], className="mb-2")),
                dbc.Row([
                    html.Ul([
                        html.Li("Accelerator von Google Coral hat sporadisch Probleme mit der Datenübertragung."),
                        html.Li("Modelle für die Hailo Accelerator müssen speziell angepasst werden (.hef)."),
                        html.Li("Inferenz auf dem BeagleY-AI konnte (noch) nicht auf der eingebauten TPU durchgeführt werden."),
                        html.Li("Die über die M.2 Schnittstelle angeschlossene TPU können auf dem Raspberry Pi 4 nicht verwendet werden (keine Schnittstelle)."),
                        html.Li("Google Coral M.2 Dual TPU hat nach erfolgreicher Inbetriebnahme nicht mehr funktioniert."),
                        html.Li("Support für die Python Bibliothek von Google Coral Devices eingestellt")
                        ]),
                ], className="mb-5"),
                dbc.Row([
                    dbc.Col(html.H3("Edge Cameras"))
                    ], className="mb-2"),
                dbc.Row([
                    dbc.Col([
                        dbc.ListGroup(
                            [
                                dbc.ListGroupItem(
                                    [
                                        html.Div(
                                            [
                                                html.H5(device["name"], className="mb-1"),
                                                #html.Small(f"{str(device['cost'])} CHF", className="text-muted"),
                                                ],
                                            className="d-flex w-100 justify-content-between",
                                            ),
                                        #html.A("link", href=device["link"], target="_blank", className="mb-1"),
                                        ]
                                    )
                                for device in camera_df.to_dict("records")]
                            )
                        ])
                    ], className="mb-5"),
                ]
            )

    return layout
