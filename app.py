from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import numpy as np
import metrics


df = pd.read_csv('out.csv')
WINDOW = 63

# fun von gemini
def create_kpi_card(title: str, value: str | float, subtitle: str = None):
    body_elements = [
        html.H6(title, className="card-subtitle text-muted mb-2"),
        html.H3(str(value), className="card-title fw-bold text-primary mb-0"),
    ]
    if subtitle:
        body_elements.append(html.Small(subtitle, className="text-muted mt-1"))

    return dbc.Card(
        dbc.CardBody(body_elements),
        className="shadow-sm border-0 h-100",
        style={"borderRadius": "8px"}
    )

def create_return_chart(data: pd.DataFrame):
    df_cum = np.expm1(data[["asset_ret", "benchmark_ret"]].cumsum())
    df_cum["Date"] = data["Date"]

    fig = px.line(
        df_cum, 
        x="Date", 
        y=["asset_ret", "benchmark_ret"],
        title="Cumulative Total Returns"
    )
    fig.update_layout(yaxis_tickformat=".2%", 
                      xaxis_title="Date", 
                      yaxis_title="Return")
    return fig


def create_asset_sharpe_chart(data: pd.DataFrame, window: int):
    df_plot = metrics.get_rolling_sharp(data, window)

    fig = px.line(
        df_plot,
        x="Date",
        y=["asset_rolling_sharpe", "benchmark_rolling_sharpe"],
        title=f"Rolling Sharpe Ratio (Window: {window})"
        )
    fig.update_layout(
        yaxis_title="Sharpe Ratio",
        xaxis_title="Date",
        yaxis_tickformat=".2f"
    )
    return fig

def create_asset_vol_chart(data: pd.DataFrame, window: int):
    df_plot = metrics.get_rolling_vol(data, window)
    fig = px.line(
        df_plot, 
        x="Date", 
        y=["asset_rolling_vol", "benchmark_rolling_vol"], 
        title=f"Rolling Vol (Window: {window})"
    )
    fig.update_layout(
        yaxis_title="Vol",
        xaxis_title="Date",
        yaxis_tickformat=".2f"
    )
    return fig

max_dd, cgar, calmar, sortino, beta, alpha, r_2, information_ratio = metrics.get_all_metrics(df)


fig_returns = create_return_chart(df)
fig_sharpe = create_asset_sharpe_chart(df, WINDOW)
fig_vol = create_asset_vol_chart(df, WINDOW)

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "20px"},
    children=[
        html.H1("Quant Tearsheet", style={"textAlign": "center"}),
        html.Div(dcc.Graph(id="chart-returns", figure=fig_returns)),
        dbc.Row([
            dbc.Col(
                create_kpi_card("Alpha (Ann.)", f"{alpha * 100:.3f}%", "vs. S&P 500"), 
                    width=12, md=3),
            dbc.Col(
                create_kpi_card("Beta", f"{beta:.3f}", "Marktsensitivität"), 
                    width=12, md=3),
            dbc.Col(
                create_kpi_card("R²", f"{r_2:.3f}", "Varianz der Asset-Renditen"), 
                    width=12, md=3),
            dbc.Col(
                create_kpi_card("Information Ratio", f"{information_ratio:.3f}", "risikoadjustierte Überrendite"), 
                    width=12, md=3)]),
       dbc.Row([
            dbc.Col(
                create_kpi_card("Max Drawdown", f"{max_dd * 100:.2f}%", "Peak-to-Trough Verlust"),
                width=12, md=3
            ),
            dbc.Col(
                create_kpi_card("cgar", f"{cgar * 100:.2f}%" if isinstance(cgar, float) else cgar, "Jährl. Wachstumsrate"),
                width=12, md=3
            ),
            dbc.Col(
                create_kpi_card("Calmar Ratio", f"{calmar:.2f}" if isinstance(calmar, float) else calmar, "cgar / Max DD"),
                width=12, md=3
            ),
            dbc.Col(
                create_kpi_card("Sortino Ratio", f"{sortino:.2f}" if isinstance(sortino, float) else sortino, "Downside Risk-adjusted"),
                width=12, md=3
            ),
        ]),
        html.Div(dcc.Graph(id="chart-sharpe", figure=fig_sharpe)),
        html.Div(dcc.Graph(id="chart-vol", figure=fig_vol)),
        
    ]
)

if __name__ == '__main__':
    app.run(debug=True)