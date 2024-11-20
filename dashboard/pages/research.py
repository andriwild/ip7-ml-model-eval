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
                    dbc.Col([html.A(html.I(className=f"fa fa-arrow-up-right-from-square me-2"), href="https://arxiv.org/abs/2409.16808", target="_blank")]),
                    dbc.Row([html.P("Energy Consumption vs Inference Time:", className="fw-bold")]),
                    dbc.Row([html.P("... energy consumption and inference time are largely linearly correlated for various models.")]),
                    dbc.Row([html.P("Energy Consumption vs Accuracy:", className="fw-bold")]),
                    dbc.Row([html.P("... accuracy remains consistent across all devices, with a minor reduction on the Jetson Orin Nano. Jetson Orin Nano is the most energy-efficient device, while Pi3 is the least efficient.")]),

                    dbc.Row([html.P("Inference Time vs Accuracy:", className="fw-bold")]),
                    dbc.Row([html.P("For the YOLO8 models, the accuracy remains stable across most devices, except for the Pis with TPUs, which show a significant reduction in accuracy due to the compression process required for execution on these platforms.")]),

                    ], className="mb-5"),
                dbc.Row([
                    dbc.Col([html.H5("Automated Analysis for Urban Biodiversity Monitoring")], width=11),
                    dbc.Col([html.A(html.I(className=f"fa fa-arrow-up-right-from-square me-2"), href="https://wullt.ch/documents/p8.pdf", target="_blank")])
                    ], className="mb-5"),
                dbc.Row([
                    dbc.Col([html.H5("Projekt Board")], width=11),
                    dbc.Col([html.A(html.I(className=f"fa fa-arrow-up-right-from-square me-2"), href="https://github.com/users/andriwild/projects/2/views/1", target="_blank")])
                    ], className="mb-5"),
                ]
    )
#https://www.hackster.io/news/benchmarking-tensorflow-and-tensorflow-lite-on-raspberry-pi-5-b9156d58a6a2
#https://wiki.seeedstudio.com/benchmark_on_rpi5_and_cm4_running_yolov8s_with_rpi_ai_kit/
#https://wiki.seeedstudio.com/tutorial_of_ai_kit_with_raspberrypi5_about_yolov8n_object_detection/

    return layout
