from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px


def ml_layout(mw_pytorch_df, mw_tflite_df, frameworks_df, mw_pipeline_df):
    device_colors = px.colors.qualitative.Pastel
    mw_pytorch_df = mw_pytorch_df.sort_values(by=["threads", "batch_size"])
    mw_pytorch_df['thread_batchsize'] = mw_pytorch_df.apply(lambda row: f"threads: {int(row['threads'])}, batch size: {int(row['batch_size'])}", axis=1)

    mw_tflite_df = mw_tflite_df.sort_values(by=["threads"])
    mw_tflite_df['thread_batchsize'] = mw_tflite_df.apply(lambda row: f"threads: {int(row['threads'])}, batch size: {int(row['batch_size'])}", axis=1)

    layout = dbc.Container(
            style={"marginTop": "20px"},
            children=[
                dbc.Row(dbc.Col(html.H1("ML"), className="mb-5")),
                dbc.Row(dbc.Col(html.H3("Frameworks"), className="mb-2")),
                dbc.Row(
                    className="mb-5",
                    children= dbc.Col(
                        dbc.ListGroup(
                            [
                                dbc.ListGroupItem([
                                        html.Div([
                                                html.H5(device["name"], className="mb-1"),
                                                html.A(html.I(className=f"fa fa-arrow-up-right-from-square me-2"), href=device["link"], target="_blank", className="mb-1"),
                                                ],
                                            className="d-flex w-100 justify-content-between",
                                            ),
                                        html.P(device["description"], className="mb-1"),
                                        ]
                                    )
                                for device in frameworks_df.to_dict("records")]
                            )
                        ),
                    ),
                dbc.Row(dbc.Col(html.H3("Mitwelten Pipeline"), className="mb-2")),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Img(src="static/images/flowers.jpg", style={'width': '100%', 'height': 'auto'}),
                            width=4
                            ),
                        dbc.Col(
                            html.Img(src="static/images/annotator2.png", style={'width': '100%', 'height': 'auto'}),
                            width=4
                            ),
                        dbc.Col(
                            html.Img(src="static/images/pollinators.png", style={'width': '100%', 'height': 'auto'}),
                            width=4
                            ),
                        ],
                    className="mt-4"
                    ),

                dbc.Row(dbc.Col(html.H4("PyTorch"), className="mb-2 mt-5")),
                dbc.Row(dbc.Col(html.P("Flower Model(640 yolov5n mAP.5:.95 = 0.685) - Pollinator Model(480 yolov5s mAP.5:.95 = 0.6575)"), className="mb-2")),

                dbc.Row([
                    dcc.Graph(
                        id='model-time-bar-chart',
                        figure=px.scatter(
                            mw_pytorch_df,
                            x='pipeline',
                            y='n_flowers',
                            title='Raspberry Pi 5 - Mitwelten ML Pipeline',
                            color='thread_batchsize',
                            labels={
                                'pipeline': 'Time (sec)',
                                'n_flowers': 'Number of Flowers'
                                },
                            color_discrete_sequence=device_colors
                            ).update_layout(
                                template='ggplot2').add_vline(x=15, line_width=2, line_dash="dash", line_color="red")
                            )
                    ], className="mb-5"),
                dbc.Row(dbc.Col(html.H4("TFLite"), className="mb-2")),
                dbc.Row(dbc.Col(html.P("Flower Model(640 yolov5n mAP.5:.95 = 0.583) Pollinator Model(480 yolov5s mAP.5:.95 = ???)"), className="mb-2")),
                dbc.Row([
                    dcc.Graph(
                        id='model-time-bar-chart',
                        figure=px.scatter(
                            mw_tflite_df,
                            x='pipeline',
                            y='n_flowers',
                            title='Raspberry Pi 5 - Mitwelten ML Pipeline',
                            color='thread_batchsize',
                            labels={
                                'pipeline': 'Time (sec)',
                                'n_flowers': 'Number of Flowers'
                                },
                            color_discrete_sequence=device_colors
                            ).update_layout(
                                template='ggplot2').add_vline(x=15, line_width=2, line_dash="dash", line_color="red")
                            )
                    ]),
                dbc.Row([
                    dcc.Graph(
                        id='model-time-bar-chart2',
                        figure=px.scatter(
                            mw_pipeline_df,
                            x='pipeline',
                            y='n_flowers',
                            title='Raspberry Pi 5 - Mitwelten ML Pipeline',
                            color='model',
                            labels={
                                'pipeline': 'Time (sec)',
                                'n_flowers': 'Number of Flowers'
                                },
                            color_discrete_sequence=device_colors
                            ).update_layout( 
                                            template='ggplot2',
                                            xaxis_range=[0, 16]
                                            )
                            .add_vline(x=15, line_width=2, line_dash="dash", line_color="red")
                            )
                    ])
                ]
            )

    return layout
