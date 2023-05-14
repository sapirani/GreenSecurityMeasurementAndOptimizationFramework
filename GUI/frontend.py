import random

import dash
from dash import html, Output, Input, State, no_update, dcc

from assets.dash_custom_styles import *
import dash_bootstrap_components as dbc
from flask import Flask


amount_of_files = [1,2,3,4,5,6,7,8,9,10]
size_of_scans = [1,2,3,4,5,6,7,8,9,10]
energy_consumed_for_each_scan = [1,2,3,4,5,6,7,8,9,10]
number_of_available_experiments = len(energy_consumed_for_each_scan)


_BOOTSWATCH_BASE = "https://cdn.jsdelivr.net/npm/bootswatch@5.1.0/dist/"
server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[_BOOTSWATCH_BASE + "slate/bootstrap.css"])


# --- About button callback
@app.callback(
    Output("about_modal", "is_open"), Input("About_btn", "n_clicks"), State("about_modal", "is_open")
)
def on_button_about_click(n, is_open):
    if n:
        return True
    return no_update


# --- Scan button callback
@app.callback(
    Output("text_after_scan_btn", "children"),
    Output("number_of_scans", "data"),
    Output("performed_scan", "data"),
    Input("scan_btn", "n_clicks"),
    Input("energy_btn", "n_clicks")
)
def on_button_scan_click(n_clicks_scan, n_clicks_energy):
    #number_of_experiments = number_of_experiments + 1
    if "scan_btn" != dash.ctx.triggered_id:
        return "", n_clicks_scan, False
    if n_clicks_scan:
        return f"The button scan has been clicked {n_clicks_scan} times", n_clicks_scan, True
    return "", n_clicks_scan, False


# --- Energy button callback
@app.callback(
    Output("text_after_energy_btn", "children"),
    Output("total_files_in_scan", "children"),
    Output("total_size_of_scan", "children"),
    Output("total_scans", "children"),
    Input("energy_btn", "n_clicks"),
    State("performed_scan", "data"),
    State("number_of_scans", "data")
)
def on_button_energy_click(n_clicks_energy, performed_scan, number_of_current_scans):

    # do something only if energy button was clicked
    if "energy_btn" != dash.ctx.triggered_id:
        return "", f"Total Files: ", f"Total Size of Scanned Files: ", f"Total Scans: "

    # check if the scan ended successfully
    # TODO: when doing real scan, avoid the second condition only
    print(f"!!!!!!!!!!Perfomed scan {performed_scan}")
    if not performed_scan or (number_of_current_scans >= number_of_available_experiments):
        return "You should perform scan first", f"Total Files: ", \
            f"Total Size of Scanned Files: ", f"Total Scans: "

    # receive experiments results
    number_of_files = amount_of_files[number_of_current_scans - 1]
    size_of_all_files = size_of_scans[number_of_current_scans - 1]
    energy_consumption = energy_consumed_for_each_scan[number_of_current_scans - 1]

    if n_clicks_energy:
        return f"The Energy consumed in this scan is: {energy_consumption} mwh", \
            f"Total Files: {number_of_files}", \
            f"Total Size of Scanned Files: {size_of_all_files} GB", f"Total Scans: {number_of_current_scans}"
    return "", f"Total Files: ", f"Total Size of Scanned Files: ", f"Total Scans: "


# Generate about object and button
about_text = [
    'Change about text']
about_ul = html.Ul(id='my-list', children=[html.Li(i) for i in about_text])
about_body = html.Div([html.H3('Motivation'), html.Hr(), about_ul, html.Br(), html.H3('Architecture'), html.Hr(),
                       html.Img(src='assets/mssp arch.png', width='1150px')])
about_modal = dbc.Modal(
    [
        dbc.ModalHeader(html.H1("Green Security", style={'color': 'white'}), style={'backgroundColor': '#e30074'}),
        dbc.ModalBody(about_body)

    ],
    id="about_modal",
    scrollable=True,
    fullscreen=True,
    is_open=False,
)

about_btn = html.Div(dbc.Button("About", id='About_btn',n_clicks=0, color='primary', style=STYLE_I,
                disabled=False), style={'padding-left': '10px'})


# Main settings bar
main_settings_bar = dbc.Row(
    [dbc.Col(about_btn, width='auto', align='center')],
    style={'background-color': 'transparent'}, className="g-2")

content_toolbar = html.Div(
    [
        main_settings_bar
    ], style={'display': 'flex', 'flex-direction': 'row', 'background-color': '#272b30', 'padding-top': '10px'})
header_bar = html.Div(
    [
        html.Span([html.I(className="t_icon"), html.H1('Green Security', style=STYLE_TITLE),
                   html.I(className="user_icon")],
                  style={'margin': 0, 'backgroundColor': '#e30074', 'display': 'flex', 'flex-direction': 'row',
                         'height': '50px', 'padding-top': '10px', 'padding-bottom': '10px'}),
        content_toolbar,
        html.Hr()
    ],
    style=CONTENT_STYLE_NAVBAR
)

# Scan Button
scan_btn = dbc.Col([html.Div(dbc.Button("Scan", id='scan_btn', n_clicks=0, color='primary', style=STYLE_I,
                disabled=False), style={'padding-left': '10px'})])
after_scan_operation = dbc.Row([html.Div(id='text_after_scan_btn', style={'color': '#FFFFFF', 'fontSize': 25})])


# Energy Button
energy_btn = dbc.Col([html.Div(dbc.Button("Energy Consumed", id='energy_btn', n_clicks=0, color='primary', style=STYLE_I,
                disabled=False), style={'padding-left': '10px'})])
after_energy_operation = dbc.Row([html.Div(id='text_after_energy_btn', style={'color': '#FFFFFF', 'fontSize': 25})])



# Content of the page
content = dbc.Container([
    dbc.Row([
        html.Div(children=['Scan Files with Windows Defender'],
                 style={'textAlign': 'center', 'color': '#B4E17F', 'fontSize': 30})
    ]),
    dbc.Row([
        html.Div(children="Total Files: ", id="total_files_in_scan", style={'color': '#FFFFFF', 'fontSize': 25})
    ]),
    dbc.Row([
        html.Div(children="Total Size of Scanned Files: ", id="total_size_of_scan", style={'color': '#FFFFFF', 'fontSize': 25})
    ]),
    dbc.Row([
        html.Div(children="Total Scans: ", id="total_scans", style={'color': '#FFFFFF', 'fontSize': 25})
    ]),
    after_scan_operation, after_energy_operation,
    dbc.Row([scan_btn, energy_btn])])


"""content = html.Div(html.Div(children=['Content'],
                            style={'display': 'flex', 'flex-direction': 'row'}),
                   style={'display': 'flex', 'justify-content': 'center'})"""

app.layout = html.Div(children=[header_bar, content, about_modal,
                                dcc.Store(id='number_of_scans', data=0),
                                dcc.Store(id='performed_scan', data=False)])

if __name__ == '__main__':
    app.run_server(debug=False)