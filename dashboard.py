import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import colorsys
import re


# Load the data
df = pd.read_csv('data/rpi/inference_benchmark.csv')


# Initialize the app with a Bootstrap stylesheet
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Get unique values for controls
devices = df['device'].unique()
models = df['model'].unique()
frameworks = df['framework'].unique()
batch_sizes = df['batch_size'].unique()

# Adjust controls based on the number of entities
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Model Performance Dashboard"), className="mb-2")
    ]),
    dbc.Row([
        dbc.Col([
            html.Label('Select Devices'),
            dcc.Checklist(
                id='device-checklist',
                options=[{'label': html.Div(dev, style={"marginLeft": "10px"}), 'value': dev} for dev in devices],
                value=list(devices),
                inline=False,
                labelStyle={"display": "flex", "align-items": "center"},
            ),
        ], width=2),
        dbc.Col([
            html.Label('Select Models'),
            dcc.Dropdown(
                id='model-dropdown',
                options=[{'label': mod, 'value': mod} for mod in models],
                value=list(models),
                multi=True
            ),
        ], width=4),
        dbc.Col([
            html.Label('Select Frameworks'),
            dcc.Checklist(
                id='framework-checklist',
                options=[{'label': html.Div(fw, style={"marginLeft": "10px"}), 'value': fw} for fw in frameworks],
                value=list(frameworks),
                inline=False,
                labelStyle={"display": "flex", "align-items": "center"},
            ),
        ], width=2),
        dbc.Col([
            html.Label('Select Batch Sizes'),
            dcc.RadioItems(
                id='batch-radio',
                options=[{'label': html.Div(str(bs), style={"marginLeft": "10px"}), 'value': bs} for bs in sorted(batch_sizes)],
                value=batch_sizes[0],
                inline=True,
                labelStyle={"display": "flex", "align-items": "center"},
            ),
        ], width=2),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='performance-bar-chart', style={'height': '600px'})
        ])
    ]),
    ], fluid=True, style={"height": "100vh", "margin": "20px"})

def get_device_color_map(devices):
    # Use pastel colors from Plotly
    pastel_colors = px.colors.qualitative.Pastel
    device_colors = {}
    for i, device in enumerate(devices):
        device_colors[device] = pastel_colors[i % len(pastel_colors)]
    return device_colors

def get_framework_shades(base_color, frameworks):
    # Convert base_color from 'rgb(r,g,b)' to RGB tuple
    m = re.match(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', base_color)
    if m:
        r, g, b = m.groups()
        r = int(r) / 255.0
        g = int(g) / 255.0
        b = int(b) / 255.0
    else:
        # If not matched, default to gray
        r, g, b = (0.5, 0.5, 0.5)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    n = len(frameworks)
    shades = []
    for i in range(n):
        # Adjust lightness
        new_l = l * (1 - (i / (n * 1.5)))  # Reduce lightness for each shade
        new_r, new_g, new_b = colorsys.hls_to_rgb(h, new_l, s)
        # Convert back to 'rgb(r,g,b)'
        new_r = int(new_r * 255)
        new_g = int(new_g * 255)
        new_b = int(new_b * 255)
        shades.append(f'rgb({new_r},{new_g},{new_b})')
    return shades

@app.callback(
    Output('performance-bar-chart', 'figure'),
    Input('device-checklist', 'value'),
    Input('model-dropdown', 'value'),
    Input('framework-checklist', 'value'),
    Input('batch-radio', 'value')
)
def update_bar_chart(selected_devices, selected_models, selected_frameworks, selected_batch):
    filtered_df = df[
        (df['device'].isin(selected_devices)) &
        (df['model'].isin(selected_models)) &
        (df['framework'].isin(selected_frameworks)) &
        (df['batch_size'] == selected_batch)
    ]
    
    if filtered_df.empty:
        fig = px.bar(title='No data available for the selected filters.')
        return fig
    
    # Sort the devices and frameworks for consistent color mapping
    devices = filtered_df['device'].unique()
    frameworks = filtered_df['framework'].unique()
    
    # Generate color mapping
    device_colors = get_device_color_map(devices)
    color_discrete_map = {}
    for device in devices:
        base_color = device_colors[device]
        device_df = filtered_df[filtered_df['device'] == device]
        device_frameworks = device_df['framework'].unique()
        framework_shades = get_framework_shades(base_color, device_frameworks)
        for i, framework in enumerate(device_frameworks):
            key = f"{device}_{framework}"
            color_discrete_map[key] = framework_shades[i]
    
    # Add a combined key for color mapping
    filtered_df['device_framework'] = filtered_df.apply(lambda row: f"{row['device']}_{row['framework']}", axis=1)
    
    fig = px.bar(
        filtered_df,
        x='model',
        y='time',
        color='device_framework',
        barmode='group',
        hover_data=['device', 'framework', 'batch_size'],
        labels={'time': 'Time (ms)', 'model': 'Model'},
        title='Model Performance on Different Devices and Frameworks',
        color_discrete_map=color_discrete_map
    )
    
    # Update legend to show device and framework separately
    for trace in fig.data:
        device_framework = trace.name
        device, framework = device_framework.split('_')
        trace.name = f"{device} - {framework}"
        trace.legendgroup = device
        trace.hovertemplate = trace.hovertemplate.replace(device_framework, f"{device} - {framework}")
    
    # Update the legend title
    fig.update_layout(legend_title_text='Device - Frameworks')
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)

