from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px



def analysis_layout(filtered_df):

    device_colors = px.colors.qualitative.Pastel

    layout = dbc.Container(
            style={"marginTop": "20px"},
            children=[
                dbc.Row([
                    dbc.Col([html.H1("Analysis"), html.P("based on Dashboard filter!")], className="mb-2")
                    ]),
                # Maximize Performance (FPS)
                dbc.Row([
                    dbc.Col(html.H3("Maximize Performance (FPS)"), className="mb-2")
                    ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(
                            id='fps-bar-chart',
                            figure=px.bar(
                                filtered_df.sort_values('fps', ascending=False).head(10),
                                x='model',
                                y='fps',
                                color='device_framework',
                                barmode='group',
                                hover_data=['framework', 'accelerator'],
                                labels={'fps': 'FPS', 'model': 'Model'},
                                title='Top 10 Configurations by FPS',
                                color_discrete_sequence=device_colors
                                ).update_layout(template='ggplot2')
                            )
                        ])
                    ]),
                # Maximize Accuracy
                dbc.Row([
                    dbc.Col(html.H3("Maximize Accuracy"), className="mb-2")
                    ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(
                            id='map-bar-chart',
                            figure=px.bar(
                                filtered_df.sort_values('mAP 50-95', ascending=False).head(10),
                                x='model',
                                y='mAP 50-95',
                                color='device_framework',
                                barmode='group',
                                hover_data=['framework', 'accelerator'],
                                labels={'mAP 50-95': 'mAP 50-95', 'model': 'Model'},
                                title='Top 10 Configurations by Accuracy (mAP)',
                                color_discrete_sequence=device_colors
                                ).update_layout(template='ggplot2')
                            )
                        ])
                    ]),
                # Optimize Cost per FPS
                dbc.Row([
                    dbc.Col(html.H3("Optimize Cost per FPS"), className="mb-2")
                    ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(
                            id='cost-per-fps-bar-chart',
                            style={'height': '600px'},
                            figure=px.bar(
                                filtered_df.sort_values('cost_per_fps').head(10),
                                x='model',
                                y='cost_per_fps',
                                color='device_framework',
                                barmode='group',
                                hover_data=['framework', 'accelerator'],
                                labels={'cost_per_fps': 'Cost per FPS', 'model': 'Model'},
                                title='Top 10 Configurations by Cost Efficiency',
                                color_discrete_sequence=device_colors
                                ).update_layout(template='ggplot2')
                            )
                        ])
                    ]),
        # Balance Performance and Accuracy
        dbc.Row([
            dbc.Col(html.H3("Balance Performance and Accuracy"), className="mb-2")
            ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id='performance-accuracy-scatter',
                    figure=px.scatter(
                        filtered_df,
                        x='fps',
                        y='mAP 50-95',
                        size='total_cost',
                        color='device_framework',
                        hover_data=['model', 'framework', 'accelerator'],
                        labels={'fps': 'FPS', 'mAP 50-95': 'mAP 50-95'},
                        title='Performance vs. Accuracy',
                        color_discrete_sequence=device_colors
                        ).update_layout(template='ggplot2')
                    )
                ])
            ]),
        # Minimize Costs
        dbc.Row([
            dbc.Col(html.H3("Minimize Costs"), className="mb-2")
            ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id='cost-bar-chart',
                    figure=px.bar(
                        filtered_df.sort_values('total_cost').head(10),
                        x='model',
                        y='total_cost',
                        color='device_framework',
                        barmode='group',
                        hover_data=['framework', 'accelerator'],
                        labels={'total_cost': 'Total Cost', 'model': 'Model'},
                        title='Top 10 Configurations by Lowest Cost',
                        color_discrete_sequence=device_colors
                        ).update_layout(template='ggplot2')
                    )
                ])
            ]),
    ])
    return layout
