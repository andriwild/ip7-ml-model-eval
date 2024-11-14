import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pages.analysis import analysis_layout
from pages.dashboard import dashboard_layout
from pages.hardware import hardware_layout
from pages.research import research_layout
from pages.ml import ml_layout
from pages.management import management_layout

# Load data
file_folder = 'data/'
inference_df = pd.read_csv(file_folder + 'rpi/inference_benchmark.csv')
devices_df = pd.read_csv( file_folder + 'devices.csv')
camera_df = pd.read_csv( file_folder + 'cameras.csv')
criteria_df = pd.read_csv( file_folder + 'criteria.csv')
map_df = pd.read_csv( file_folder + 'map.csv')
frameworks_df = pd.read_csv(file_folder + 'frameworks.csv')
mw_pytorch_df = pd.read_csv( file_folder + 'mw/pytorch/rpi5_mw_pytorch.csv')
mw_tflite_df = pd.read_csv( file_folder + 'mw/tflite/rpi5_mw_tflite.csv')

batch_sizes = inference_df['batch_size'].unique()

inference_df['fps'] = 1000 / inference_df['time']

#Make model names uppercase for consistency
inference_df['model'] = inference_df['model'].str.upper()
map_df['model'] = map_df['model'].str.upper()

inference_df = inference_df.merge(map_df, on=['model', "framework"], how='left')

devices_df['device'] = devices_df['device'].str.lower()
device_costs = devices_df.set_index('device')['cost'].to_dict()

inference_df['device'] = inference_df['device'].str.lower()
inference_df['device_cost'] = inference_df['device'].map(device_costs)

inference_df['accelerator_cost'] = inference_df['accelerator'].map(device_costs)
inference_df['accelerator_cost'] = inference_df['accelerator_cost'].fillna(0)

inference_df['total_cost'] = inference_df['device_cost'] + inference_df['accelerator_cost']

inference_df['cost_per_fps'] = inference_df['total_cost'] / inference_df['fps']
filtered_df = inference_df

filtered_df['device_framework'] = filtered_df.apply(lambda row: f"{row['device']} {row['accelerator']} {row['framework']}", axis=1)

# Initialize the app
stylesheets = [
    dbc.themes.LUX,
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"  # Font Awesome CSS
    ]

app = dash.Dash(
        __name__, 
        external_stylesheets=stylesheets,
        suppress_callback_exceptions=True, 
        title="Model Eval")

# Create navbar
navbar = dbc.NavbarSimple(
    brand='IP7 - Model Evaluation Dashboard',
    brand_href='/',
    color='primary',
    dark=True,
    children=[
        dbc.NavItem(dbc.NavLink("Dashboard", href="/", active=True, id='dashboard-link')),
        dbc.NavItem(dbc.NavLink("Analysis", href="/analysis", id='analysis-link')),
        dbc.NavItem(dbc.NavLink("Hardware", href="/hardware", id='hardware-link')),
        dbc.NavItem(dbc.NavLink("ML", href="/ml", id='ml-link')),
        dbc.NavItem(dbc.NavLink("Research", href="/research", id='research-link')),
        dbc.NavItem(dbc.NavLink("Management", href="/management", id='management-link')),
    ]
)



# App layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
])

# Update page content based on URL
@app.callback(
        Output('page-content', 'children'),
        Output('dashboard-link', 'active'),
        Output('analysis-link', 'active'),
        Output('hardware-link', 'active'),
        Output('ml-link', 'active'),
        Output('research-link', 'active'),
        Output('management-link', 'active'),
        Input('url', 'pathname')
        )

def display_page(pathname):
    n_pages = 6
    active = [False] * n_pages

    match(pathname):
        case '/':   
            active[0] = True
            return dashboard_layout(inference_df), *active
        case '/analysis':
            active[1] = True
            return analysis_layout(filtered_df), *active
        case '/hardware':
            active[2] = True
            return hardware_layout(devices_df, camera_df), *active
        case '/ml':
            active[3] = True
            return ml_layout(mw_pytorch_df, mw_tflite_df, frameworks_df), *active
        case '/research':
            active[4] = True
            return research_layout(), *active
        case '/management':
            active[5] = True
            return management_layout(), *active




@app.callback(
    Output('performance-bar-chart', 'figure'),
    Input('device-checklist', 'value'),
    Input('accelerator-checklist', 'value'),
    Input('model-dropdown-v5', 'value'),
    Input('model-dropdown-v8', 'value'),
    Input('model-dropdown-v10', 'value'),
    Input('framework-checklist', 'value'),
    Input('batch-radio', 'value'),
    Input('unit-radio', 'value')
)
def update_bar_chart(
        selected_devices, 
        selected_accelerators, 
        selected_models_v5, 
        selected_models_v8, 
        selected_models_v10, 
        selected_frameworks, 
        selected_batch,
        unit
        ):
    selected_models = [*selected_models_v5, *selected_models_v8, *selected_models_v10]
    print(selected_models)
    global filtered_df
    filtered_df = inference_df[
        (inference_df['device'].isin(selected_devices)) &
        (inference_df['accelerator'].isin(selected_accelerators)) &
        (inference_df['model'].isin(selected_models)) &
        (inference_df['framework'].isin(selected_frameworks)) &
        (inference_df['batch_size'] == selected_batch)
    ]

    if filtered_df.empty:
        fig = px.bar(title='No data available for the selected filters.')
        return fig


    # Only for sorting
    filtered_df['yolo_version'] = filtered_df['model'].str.extract('(\d+)', expand=False).astype(int)
    filtered_df['yolo_variant'] = filtered_df['model'].str.extract(r'\d+([a-zA-Z])')

    size_order = {
        'n': 1,   # nano
        's': 2,   # small
        'm': 3,   # medium
        'l': 4,   # large
        'x': 5    # extra large
    }

    filtered_df['variant_order'] = filtered_df['yolo_variant'].map(size_order)
    filtered_df = filtered_df.sort_values(by=["yolo_version", "variant_order"])


    # Generate color mapping
    device_colors = px.colors.qualitative.Pastel

    fig = px.bar(
        filtered_df,
        x='model',
        y=unit,
        color='device_framework',
        barmode='group',
        hover_data=['framework', 'accelerator', 'batch_size', 'fps'],
        labels={'time': 'Time (ms)', 'model': 'Model'},
        title='Model Performance on Different Devices and Frameworks (640x640)',
        color_discrete_sequence=device_colors
    ).update_layout(template='ggplot2')

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)

