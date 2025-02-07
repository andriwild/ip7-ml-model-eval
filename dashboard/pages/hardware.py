from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px

def hardware_layout(devices_df, camera_df, filtered_df):

    device_colors = px.colors.qualitative.Pastel
    devices_df = devices_df.sort_values('cost')
    camera_df = camera_df.sort_values('cost')

    print(filtered_df)


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
                                devices_df[devices_df['type'] == 'device'].sort_values('name'),
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
                    dbc.Col(html.H3("Accelerators"), className="mb-2")
                    ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(
                            id='fps-bar-chart',
                            figure=px.bar(
                                devices_df[devices_df['type'] == 'accelerator'].sort_values('name'),
                                x='name',
                                y='cost',
                                color='name',
                                labels={'cost': 'CHF', 'name': 'Name'},
                                title='Cost of Accelerators',
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
                                                html.A(html.I(className=f"fa fa-arrow-up-right-from-square me-2"), href=device["link"], target="_blank", className="mb-1"),
                                                ],
                                            className="d-flex w-100 justify-content-between",
                                            ),
                                                html.Small(f"{str(device['cost'])} CHF", className="text-muted"),
                                        ]
                                    )
                                 for device in devices_df.to_dict("records")]
                            )
                        ])
                    ], className="mb-5"),

                dbc.Row(dbc.Col([html.H5("Erkenntnisse")], className="mb-2")),
                dbc.Row([
                    html.Ul([
                        html.Li("Google Coral accelerator has sporadic data transmission issues."),
                        html.Li("Models for the Hailo accelerator need to be specially adapted (.hef)."),
                        html.Li("Inference on the BeagleY-AI could not (yet) be performed on the built-in TPU."),
                        html.Li("The TPU's connected via the M.2 interface cannot be used on the Raspberry Pi 4 (no interface)."),
                        html.Li("Google Coral M.2 Dual TPU stopped working after successful deployment."),
                        html.Li("Support for the Python library of Google Coral devices has been discontinued.")
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

                dbc.Row([
                    dcc.Graph(
                        id='model-time-bar-chart2',
                        figure=px.scatter(
                            filtered_df,
                            x='time',
                            y='total_cost',
                            title='Raspberry Pi 5 - Mitwelten ML Pipeline',
                            color='device_framework',
                            labels={
                                'time': 'Time (ms)',
                                'total_cost': 'CHF'
                                },
                            color_discrete_sequence=device_colors
                            ).update_layout( 
                                            template='ggplot2',
                                            yaxis_range=[50, 220])
                            .update_traces(marker_size=10)
                            )
                    ])

                ]

            )

    return layout
