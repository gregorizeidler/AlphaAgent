"""
Real-Time Live Trading Dashboard
Interactive web-based dashboard using Plotly Dash
"""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Dash app
app = dash.Dash(
    __name__,
    title="AlphaAgent Live Dashboard",
    update_title="Live Trading..."
)

# Global state (in production, use Redis or database)
dashboard_state = {
    'portfolio_value': [10000],
    'timestamps': [datetime.now()],
    'positions': {},
    'actions': [],
    'rewards': [],
    'risk_score': 0,
    'current_regime': 'UNKNOWN',
    'alerts': []
}

# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🤖 AlphaAgent Live Trading Dashboard", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 10}),
        html.H4(id='live-status', 
                children="● LIVE", 
                style={'textAlign': 'center', 'color': '#27ae60', 'marginTop': 0})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),
    
    # Auto-refresh interval
    dcc.Interval(
        id='interval-component',
        interval=2000,  # Update every 2 seconds
        n_intervals=0
    ),
    
    # Top Row: Key Metrics
    html.Div([
        # Portfolio Value Card
        html.Div([
            html.H3("Portfolio Value", style={'color': '#34495e'}),
            html.H2(id='portfolio-value-display', children="$10,000", 
                    style={'color': '#27ae60', 'fontSize': '36px', 'fontWeight': 'bold'}),
            html.P(id='portfolio-change', children="+0.00%", style={'color': '#27ae60'})
        ], className='metric-card', style={
            'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px',
            'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'flex': '1', 'margin': '10px'
        }),
        
        # Sharpe Ratio Card
        html.Div([
            html.H3("Sharpe Ratio", style={'color': '#34495e'}),
            html.H2(id='sharpe-display', children="0.00", 
                    style={'color': '#3498db', 'fontSize': '36px', 'fontWeight': 'bold'}),
            html.P("Risk-Adjusted Return", style={'color': '#7f8c8d'})
        ], className='metric-card', style={
            'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px',
            'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'flex': '1', 'margin': '10px'
        }),
        
        # Risk Score Card
        html.Div([
            html.H3("Risk Score", style={'color': '#34495e'}),
            html.H2(id='risk-score-display', children="0", 
                    style={'fontSize': '36px', 'fontWeight': 'bold'}),
            html.P(id='risk-level', children="LOW", style={'fontWeight': 'bold'})
        ], className='metric-card', style={
            'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px',
            'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'flex': '1', 'margin': '10px'
        }),
        
        # Market Regime Card
        html.Div([
            html.H3("Market Regime", style={'color': '#34495e'}),
            html.H2(id='regime-display', children="UNKNOWN", 
                    style={'fontSize': '28px', 'fontWeight': 'bold'}),
            html.P(id='regime-confidence', children="Confidence: N/A")
        ], className='metric-card', style={
            'backgroundColor': 'white', 'padding': '20px', 'borderRadius': '10px',
            'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'flex': '1', 'margin': '10px'
        }),
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'}),
    
    # Middle Row: Charts
    html.Div([
        # Portfolio Value Chart
        html.Div([
            dcc.Graph(id='portfolio-chart', style={'height': '400px'})
        ], style={'flex': '2', 'margin': '10px', 'backgroundColor': 'white', 
                  'borderRadius': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'padding': '10px'}),
        
        # Actions Distribution
        html.Div([
            dcc.Graph(id='actions-chart', style={'height': '400px'})
        ], style={'flex': '1', 'margin': '10px', 'backgroundColor': 'white',
                  'borderRadius': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'padding': '10px'}),
    ], style={'display': 'flex', 'marginBottom': '20px'}),
    
    # Bottom Row: Positions & Alerts
    html.Div([
        # Current Positions
        html.Div([
            html.H3("📊 Current Positions", style={'color': '#2c3e50', 'marginBottom': '15px'}),
            html.Div(id='positions-table')
        ], style={'flex': '1', 'margin': '10px', 'backgroundColor': 'white',
                  'borderRadius': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'padding': '20px'}),
        
        # Alerts
        html.Div([
            html.H3("🚨 Alerts & Warnings", style={'color': '#2c3e50', 'marginBottom': '15px'}),
            html.Div(id='alerts-list')
        ], style={'flex': '1', 'margin': '10px', 'backgroundColor': 'white',
                  'borderRadius': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'padding': '20px'}),
    ], style={'display': 'flex', 'marginBottom': '20px'}),
    
    # Footer
    html.Div([
        html.P(f"AlphaAgent v1.0 | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
               style={'textAlign': 'center', 'color': '#7f8c8d'})
    ], style={'marginTop': '20px'})
    
], style={'backgroundColor': '#f5f6fa', 'padding': '20px', 'fontFamily': 'Arial, sans-serif'})


# Callbacks
@app.callback(
    [
        Output('portfolio-value-display', 'children'),
        Output('portfolio-change', 'children'),
        Output('portfolio-change', 'style'),
        Output('sharpe-display', 'children'),
        Output('risk-score-display', 'children'),
        Output('risk-score-display', 'style'),
        Output('risk-level', 'children'),
        Output('regime-display', 'children'),
        Output('regime-confidence', 'children'),
        Output('portfolio-chart', 'figure'),
        Output('actions-chart', 'figure'),
        Output('positions-table', 'children'),
        Output('alerts-list', 'children'),
    ],
    Input('interval-component', 'n_intervals')
)
def update_dashboard(n):
    """Update all dashboard components"""
    
    # Simulate data updates (in production, fetch from agent/broker)
    update_simulated_data(n)
    
    # Portfolio Value
    current_value = dashboard_state['portfolio_value'][-1]
    initial_value = dashboard_state['portfolio_value'][0]
    change_pct = ((current_value - initial_value) / initial_value) * 100
    
    portfolio_display = f"${current_value:,.2f}"
    change_display = f"{'+' if change_pct >= 0 else ''}{change_pct:.2f}%"
    change_style = {'color': '#27ae60' if change_pct >= 0 else '#e74c3c', 'fontSize': '18px'}
    
    # Sharpe Ratio
    if len(dashboard_state['rewards']) > 10:
        returns = np.diff(dashboard_state['portfolio_value']) / dashboard_state['portfolio_value'][:-1]
        sharpe = (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(252)
    else:
        sharpe = 0.0
    sharpe_display = f"{sharpe:.2f}"
    
    # Risk Score
    risk_score = dashboard_state['risk_score']
    risk_color = '#27ae60' if risk_score < 30 else '#f39c12' if risk_score < 70 else '#e74c3c'
    risk_style = {'color': risk_color, 'fontSize': '36px', 'fontWeight': 'bold'}
    risk_level = "LOW" if risk_score < 30 else "MEDIUM" if risk_score < 70 else "HIGH"
    
    # Market Regime
    regime = dashboard_state['current_regime']
    regime_color = {'BULL': '#27ae60', 'BEAR': '#e74c3c', 'SIDEWAYS': '#f39c12', 'HIGH_VOL': '#9b59b6'}
    
    # Portfolio Chart
    portfolio_fig = go.Figure()
    portfolio_fig.add_trace(go.Scatter(
        x=dashboard_state['timestamps'],
        y=dashboard_state['portfolio_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#3498db', width=2),
        fill='tozeroy',
        fillcolor='rgba(52, 152, 219, 0.1)'
    ))
    portfolio_fig.add_hline(y=initial_value, line_dash="dash", line_color="red", 
                            annotation_text="Initial", annotation_position="right")
    portfolio_fig.update_layout(
        title="Portfolio Value Evolution",
        xaxis_title="Time",
        yaxis_title="Value ($)",
        hovermode='x unified',
        template='plotly_white',
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    # Actions Chart
    if len(dashboard_state['actions']) > 0:
        actions_fig = go.Figure()
        actions_fig.add_trace(go.Histogram(
            x=dashboard_state['actions'],
            nbinsx=30,
            marker_color='#9b59b6',
            opacity=0.7
        ))
        actions_fig.add_vline(x=0, line_dash="dash", line_color="red")
        actions_fig.update_layout(
            title="Actions Distribution",
            xaxis_title="Action (Position Change)",
            yaxis_title="Frequency",
            template='plotly_white',
            margin=dict(l=40, r=40, t=40, b=40)
        )
    else:
        actions_fig = go.Figure()
        actions_fig.update_layout(title="Actions Distribution (No data yet)")
    
    # Positions Table
    positions_html = []
    if dashboard_state['positions']:
        for ticker, data in dashboard_state['positions'].items():
            pnl_color = '#27ae60' if data['pnl'] >= 0 else '#e74c3c'
            positions_html.append(html.Div([
                html.Div([
                    html.Span(ticker, style={'fontWeight': 'bold', 'fontSize': '18px'}),
                    html.Span(f" | {data['shares']} shares", style={'color': '#7f8c8d'})
                ]),
                html.Div([
                    html.Span(f"Value: ${data['value']:,.2f}", style={'marginRight': '15px'}),
                    html.Span(f"P&L: ", style={'marginRight': '5px'}),
                    html.Span(f"${data['pnl']:,.2f}", style={'color': pnl_color, 'fontWeight': 'bold'})
                ], style={'marginTop': '5px'})
            ], style={'borderBottom': '1px solid #ecf0f1', 'padding': '10px 0'}))
    else:
        positions_html = [html.P("No open positions", style={'color': '#7f8c8d', 'fontStyle': 'italic'})]
    
    # Alerts
    alerts_html = []
    if dashboard_state['alerts']:
        for alert in dashboard_state['alerts'][-5:]:  # Show last 5
            alert_color = {'HIGH': '#e74c3c', 'MEDIUM': '#f39c12', 'LOW': '#3498db'}[alert['severity']]
            alerts_html.append(html.Div([
                html.Span("● ", style={'color': alert_color, 'fontSize': '20px'}),
                html.Span(alert['message'], style={'fontWeight': 'bold'}),
                html.P(alert['action'], style={'marginLeft': '20px', 'color': '#7f8c8d', 'fontSize': '14px'})
            ], style={'marginBottom': '15px'}))
    else:
        alerts_html = [html.P("No alerts ✓", style={'color': '#27ae60', 'fontWeight': 'bold'})]
    
    return (
        portfolio_display,
        change_display,
        change_style,
        sharpe_display,
        f"{risk_score}",
        risk_style,
        risk_level,
        regime,
        "Confidence: 85%",
        portfolio_fig,
        actions_fig,
        positions_html,
        alerts_html
    )


def update_simulated_data(n):
    """Simulate data updates (replace with real agent data)"""
    # Simulate portfolio value changes
    last_value = dashboard_state['portfolio_value'][-1]
    change = np.random.randn() * 50
    new_value = max(1000, last_value + change)
    
    dashboard_state['portfolio_value'].append(new_value)
    dashboard_state['timestamps'].append(datetime.now())
    
    # Simulate action
    action = np.random.randn() * 0.3
    dashboard_state['actions'].append(action)
    
    # Simulate reward
    reward = change / 100
    dashboard_state['rewards'].append(reward)
    
    # Update risk score
    dashboard_state['risk_score'] = int(np.random.rand() * 100)
    
    # Update regime
    regimes = ['BULL', 'BEAR', 'SIDEWAYS', 'HIGH_VOL']
    if n % 10 == 0:  # Change regime every 20 seconds
        dashboard_state['current_regime'] = np.random.choice(regimes)
    
    # Simulate positions
    dashboard_state['positions'] = {
        'AAPL': {'shares': 10, 'value': 2700, 'pnl': np.random.randn() * 100},
        'GOOGL': {'shares': 5, 'value': 1800, 'pnl': np.random.randn() * 150},
        'MSFT': {'shares': 8, 'value': 3200, 'pnl': np.random.randn() * 120}
    }
    
    # Simulate alerts
    if n % 15 == 0 and np.random.rand() > 0.7:
        severities = ['HIGH', 'MEDIUM', 'LOW']
        messages = [
            "Drawdown approaching limit",
            "Regime change detected",
            "High volatility detected",
            "Position limit reached"
        ]
        dashboard_state['alerts'].append({
            'severity': np.random.choice(severities),
            'message': np.random.choice(messages),
            'action': 'Monitoring situation'
        })
    
    # Keep last 1000 datapoints
    for key in ['portfolio_value', 'timestamps', 'actions', 'rewards']:
        if len(dashboard_state[key]) > 1000:
            dashboard_state[key] = dashboard_state[key][-1000:]


if __name__ == '__main__':
    print("="*60)
    print("🤖 AlphaAgent Live Dashboard")
    print("="*60)
    print("\nStarting dashboard server...")
    print("Open your browser and go to: http://127.0.0.1:8050/")
    print("\nPress Ctrl+C to stop")
    print("="*60)
    
    app.run_server(debug=True, host='127.0.0.1', port=8050)

