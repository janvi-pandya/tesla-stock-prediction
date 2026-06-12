# streamlit_app.py — TSLA Stock Price Prediction Dashboard
# ─────────────────────────────────────────────────────────────────────────
# Run with:  streamlit run streamlit_app.py
#
# Required files in the same directory:
#   - TSLA.csv
#   - tsla_simplernn_model.weights.h5
#   - tsla_lstm_tuned_model.weights.h5
#   - tsla_scaler.pkl
# ─────────────────────────────────────────────────────────────────────────

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Print working directory and files ────────────────────────────────────
print("Current working directory:", os.getcwd())
print("Files in current directory:")
print(os.listdir())

# ── Tensorflow import with graceful fallback ──────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TSLA Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size:2.2rem; font-weight:700; color:#1f77b4; }
    .metric-card { background:#f0f2f6; border-radius:10px; padding:16px 20px; }
    .stMetric    { background:#ffffff; border-radius:8px; padding:8px; box-shadow:0 1px 4px rgba(0,0,0,.08); }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# ── Architecture Builders (must match training notebook exactly) ──────────
# ═══════════════════════════════════════════════════════════════════════════
#
# IMPORTANT: These functions must reconstruct the EXACT same architecture
# that was used when the weights were saved in Colab. If GridSearchCV
# selected different hyperparameters for the tuned LSTM (units / dropout),
# update BEST_LSTM_UNITS / BEST_LSTM_DROPOUT below to match — otherwise
# load_weights() will raise a shape-mismatch error.

# Default SimpleRNN config (from notebook: build_rnn(units=64, dropout=0.2))
RNN_UNITS   = 64
RNN_DROPOUT = 0.2

# Tuned LSTM config — UPDATE THESE to match grid_search.best_params_ if different
BEST_LSTM_UNITS   = 64    # grid_search.best_params_['model__units']
BEST_LSTM_DROPOUT = 0.2   # grid_search.best_params_['model__dropout']
MODEL_WINDOW = 60


def _build_rnn(window=60, units=RNN_UNITS, dropout=RNN_DROPOUT):
    """Two-layer SimpleRNN — matches build_rnn() in training notebook."""
    model = Sequential([
        SimpleRNN(units, return_sequences=True, input_shape=(window, 1), name='rnn_layer_1'),
        Dropout(dropout, name='dropout_1'),
        SimpleRNN(units // 2, return_sequences=False, name='rnn_layer_2'),
        Dropout(dropout, name='dropout_2'),
        Dense(1, name='output')
    ])
    return model


def _build_lstm(window=60, units=BEST_LSTM_UNITS, dropout=BEST_LSTM_DROPOUT):
    """Two-layer tuned LSTM — matches lstm_tuned in training notebook."""
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(window, 1), name='tuned_lstm_1'),
        Dropout(dropout, name='tuned_dropout_1'),
        LSTM(units // 2, return_sequences=False, name='tuned_lstm_2'),
        Dropout(dropout, name='tuned_dropout_2'),
        Dense(1, name='output')
    ])
    return model


# ═══════════════════════════════════════════════════════════════════════════
# ── Utility Functions ──────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    df.sort_index(inplace=True)
    df["Daily_Return"]  = df["Close"].pct_change() * 100
    df["MA_20"]         = df["Close"].rolling(20).mean()
    df["MA_50"]         = df["Close"].rolling(50).mean()
    df["MA_200"]        = df["Close"].rolling(200).mean()
    df["Volatility_20"] = df["Daily_Return"].rolling(20).std()
    return df


@st.cache_resource
def load_assets(rnn_weights_path, lstm_weights_path, scaler_path, window_size=60):
    """Rebuild model architectures and load weights only (Keras-version-safe)."""
    assets = {}
    load_messages = []
    if TF_AVAILABLE:
        if os.path.exists(rnn_weights_path):
            try:
                rnn = _build_rnn(window=window_size)
                rnn.load_weights(rnn_weights_path)
                assets["SimpleRNN"] = rnn
                load_messages.append(f"Loaded SimpleRNN from {rnn_weights_path}")
            except Exception as e:
                load_messages.append(f"SimpleRNN weight load failed: {e}")

        if os.path.exists(lstm_weights_path):
            try:
                lstm = _build_lstm(window=window_size)
                lstm.load_weights(lstm_weights_path)
                assets["LSTM Tuned"] = lstm
                load_messages.append(f"Loaded LSTM Tuned from {lstm_weights_path}")
            except Exception as e:
                load_messages.append(f"LSTM weight load failed: {e}")

    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            assets["scaler"] = pickle.load(f)
        load_messages.append(f"Loaded scaler from {scaler_path}")

    assets["_messages"] = load_messages
    return assets


def create_sequences(data: np.ndarray, window: int = 60):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i, 0])
        y.append(data[i, 0])
    return np.array(X).reshape(-1, window, 1), np.array(y)


def predict_future(model, last_w_scaled: np.ndarray,
                    n_days: int, scaler, window: int) -> np.ndarray:
    seq = np.empty(window + n_days, dtype=np.float32)
    seq[:window] = last_w_scaled.flatten()[-window:]

    if TF_AVAILABLE:
        @tf.function(reduce_retracing=True)
        @tf.autograph.experimental.do_not_convert
        def predict_one(batch):
            return model(batch, training=False)

        # Warm up the compiled graph before the long iterative loop.
        predict_one(tf.zeros((1, window, 1), dtype=tf.float32))

    for i in range(n_days):
        inp = seq[i:i + window].reshape(1, window, 1)
        if TF_AVAILABLE:
            pred = float(predict_one(tf.convert_to_tensor(inp, dtype=tf.float32))[0, 0].numpy())
        else:
            pred = float(model.predict(inp, verbose=0)[0, 0])
        seq[window + i] = pred

    preds = seq[window:]
    return scaler.inverse_transform(preds.reshape(-1, 1)).flatten()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100)
    return {"RMSE": rmse, "MAE": mae, "R²": r2, "MAPE (%)": mape}


def resolve_local_path(path: str) -> str:
    """Resolve relative paths against the app file first, then the launch cwd."""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    app_relative = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    return app_relative


def get_future_trading_dates(last_date, target_date):
    """Return business dates from the day after the dataset ends through target_date."""
    start = pd.Timestamp(last_date).normalize() + pd.offsets.BDay(1)
    end = pd.Timestamp(target_date).normalize()
    if end < start:
        return pd.DatetimeIndex([])
    return pd.bdate_range(start=start, end=end)


# ═══════════════════════════════════════════════════════════════════════════
# ── Sidebar ────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/e8/Tesla_logo.png", width=90)
    st.title("⚙️ Configuration")

    st.subheader("📂 Data")
    data_path = st.text_input("TSLA CSV Path", value="TSLA.csv")

    # Fixed model/scaler asset paths (output of training notebook)
    RNN_WEIGHTS_PATH  = "tsla_simplernn.weights.h5"
    LSTM_WEIGHTS_PATH = "tsla_lstm_tuned.weights.h5"
    SCALER_PATH       = "tsla_scaler.pkl"

    st.divider()

    st.subheader("🔮 Forecast Settings")
    horizon      = st.slider("Forecast Horizon (days)", 1, 30, 10)
    window_size  = MODEL_WINDOW
    st.caption(f"Lookback window: {MODEL_WINDOW} trading days (matches saved model training)")
    selected_mdl = st.selectbox("Active Model", ["SimpleRNN", "LSTM Tuned", "Both"])

    st.divider()
    st.subheader("📊 Chart Settings")
    date_range = st.date_input("Date Range",
                                value=[], help="Leave empty for full range")
    show_vol = st.checkbox("Show Volatility Overlay", value=True)
    show_ma  = st.checkbox("Show Moving Averages",    value=True)

    st.divider()
    st.caption("📌 TSLA Stock Price Prediction · Deep Learning")

# ═══════════════════════════════════════════════════════════════════════════
# ── Load Data & Models ─────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<p class="main-title">📈 Tesla Stock Price Prediction Dashboard</p>',
            unsafe_allow_html=True)
st.markdown("**SimpleRNN & LSTM Deep Learning · Financial Services Domain**")
st.divider()

# Load data
data_path_resolved = resolve_local_path(data_path)
rnn_weights_resolved = resolve_local_path(RNN_WEIGHTS_PATH)
lstm_weights_resolved = resolve_local_path(LSTM_WEIGHTS_PATH)
scaler_path_resolved = resolve_local_path(SCALER_PATH)

if not os.path.exists(data_path_resolved):
    st.error(f"❌ Data file not found: `{data_path}`  \n"
             "Place `TSLA.csv` in the same directory and re-run.")
    st.stop()

df = load_data(data_path_resolved)

assets = load_assets(rnn_weights_resolved, lstm_weights_resolved, scaler_path_resolved,
                      window_size=window_size)

for msg in assets.get("_messages", []):
    st.sidebar.caption(msg)

scaler_loaded = assets.get("scaler")
if scaler_loaded is None:
    scaler_loaded = MinMaxScaler()
    scaler_loaded.fit(df[["Close"]].values)
    st.warning("⚠️ Saved scaler not found — fitted fresh scaler from data "
               "(predictions may be slightly off if training data range differs).")

# ── Filter by date range ──────────────────────────────────────────────────
df_view = df.copy()
if len(date_range) == 2:
    df_view = df_view.loc[str(date_range[0]):str(date_range[1])]

# ═══════════════════════════════════════════════════════════════════════════
# ── Tab Layout ─────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Exploratory Analysis",
    "🤖 Model Predictions",
    "🔮 Future Forecast",
    "📋 Model Comparison"
])

# ─────────────────────────────────────────────────────────────────────────
# TAB 1 — Exploratory Analysis
# ─────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records",   f"{len(df_view):,}")
    col2.metric("Date From",       str(df_view.index.min().date()))
    col3.metric("Date To",         str(df_view.index.max().date()))
    col4.metric("Max Close Price", f"${df_view['Close'].max():.2f}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Min Close Price",   f"${df_view['Close'].min():.2f}")
    col6.metric("Avg Close Price",   f"${df_view['Close'].mean():.2f}")
    col7.metric("Avg Daily Return",  f"{df_view['Daily_Return'].mean():.3f}%")
    col8.metric("20-Day Avg Volume", f"{df_view['Volume'].rolling(20).mean().iloc[-1]/1e6:.1f}M")

    st.divider()

    # Price + MA chart
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df_view.index, df_view["Close"], color="steelblue",
            linewidth=1.2, alpha=0.8, label="Close")
    if show_ma:
        ax.plot(df_view.index, df_view["MA_20"],  color="orange", linewidth=1.2, label="MA-20")
        ax.plot(df_view.index, df_view["MA_50"],  color="green",  linewidth=1.2, label="MA-50")
        ax.plot(df_view.index, df_view["MA_200"], color="red",    linewidth=1.2, label="MA-200")
    ax.set_title("TSLA Closing Price with Moving Averages", fontweight="bold")
    ax.set_ylabel("Price (USD)"); ax.legend()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

    # Volume + Volatility
    fig2, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axes[0].bar(df_view.index, df_view["Volume"]/1e6, color="coral", alpha=0.6, width=1)
    axes[0].set_title("Trading Volume (Millions)", fontweight="bold")
    axes[0].set_ylabel("Volume (M)")
    if show_vol:
        axes[1].fill_between(df_view.index, df_view["Volatility_20"],
                              alpha=0.4, color="crimson")
        axes[1].plot(df_view.index, df_view["Volatility_20"], color="crimson", linewidth=0.8)
        axes[1].set_title("20-Day Rolling Volatility", fontweight="bold")
        axes[1].set_ylabel("Volatility (%)")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    st.pyplot(fig2); plt.close(fig2)

    # Daily return distribution
    st.subheader("Daily Return Distribution")
    fig3, axes3 = plt.subplots(1, 2, figsize=(13, 4))
    axes3[0].hist(df_view["Daily_Return"].dropna(), bins=60,
                  color="steelblue", edgecolor="white", alpha=0.8)
    axes3[0].axvline(0, color="red", linestyle="--")
    axes3[0].set_title("Histogram of Daily Returns")
    axes3[0].set_xlabel("Return (%)")

    sns.kdeplot(df_view["Daily_Return"].dropna(), ax=axes3[1], fill=True, color="steelblue")
    axes3[1].axvline(df_view["Daily_Return"].mean(), color="orange",
                     linestyle="--", label=f"Mean: {df_view['Daily_Return'].mean():.3f}%")
    axes3[1].set_title("KDE of Daily Returns")
    axes3[1].legend()
    plt.tight_layout()
    st.pyplot(fig3); plt.close(fig3)

    # Raw data table
    with st.expander("🗃️ View Raw Data"):
        st.dataframe(
            df_view[["Open", "High", "Low", "Close", "Volume",
                     "Daily_Return", "MA_20", "MA_50"]].round(2),
            use_container_width=True
        )

# ─────────────────────────────────────────────────────────────────────────
# TAB 2 — Model Predictions (Test Set)
# ─────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("🤖 Model Predictions on Test Set")

    if not TF_AVAILABLE:
        st.error("TensorFlow not installed. Run: `pip install tensorflow`")
    elif not assets.get("SimpleRNN") and not assets.get("LSTM Tuned"):
        st.warning("⚠️ No trained model weight files found. Train the models in "
                   "the notebook, save weights, then re-run the app.")
        st.info(f"Expected files in this directory:\n"
                f"- `{RNN_WEIGHTS_PATH}`\n"
                f"- `{LSTM_WEIGHTS_PATH}`\n"
                f"- `{SCALER_PATH}`")
    else:
        close_scaled = scaler_loaded.transform(df[["Close"]].values)
        X_all, y_all = create_sequences(close_scaled, window=window_size)
        split = int(len(X_all) * 0.80)
        X_test, y_test = X_all[split:], y_all[split:]
        y_test_inv = scaler_loaded.inverse_transform(y_test.reshape(-1, 1)).flatten()

        models_to_show = (
            ["SimpleRNN", "LSTM Tuned"] if selected_mdl == "Both"
            else [selected_mdl]
        )

        all_metrics = {}
        fig_pred, ax_pred = plt.subplots(figsize=(14, 5))
        ax_pred.plot(y_test_inv, color="steelblue", linewidth=1.5, label="Actual", zorder=3)

        colors_map = {"SimpleRNN": "orange", "LSTM Tuned": "crimson"}
        try:
            for mname in models_to_show:
                mdl = assets.get(mname)
                if mdl is None:
                    st.warning(f"Model `{mname}` not found — skipping.")
                    continue
                y_pred_sc  = mdl.predict(X_test, verbose=0)
                y_pred_inv = scaler_loaded.inverse_transform(y_pred_sc).flatten()
                m = compute_metrics(y_test_inv, y_pred_inv)
                all_metrics[mname] = m
                ax_pred.plot(y_pred_inv, linestyle="--", linewidth=1.2,
                              color=colors_map.get(mname, "green"), label=f"Predicted ({mname})")
        except Exception as e:
            st.error(f"Prediction failed — likely a window-size / input-shape "
                     f"mismatch with the saved weights. Error: {e}")

        ax_pred.set_title("Actual vs Predicted Closing Price — Test Set", fontweight="bold")
        ax_pred.set_xlabel("Test Day Index"); ax_pred.set_ylabel("Price (USD)")
        ax_pred.legend(); plt.tight_layout()
        st.pyplot(fig_pred); plt.close(fig_pred)

        # Metrics cards
        if all_metrics:
            st.subheader("📊 Performance Metrics")
            cols = st.columns(len(all_metrics))
            for col, (mname, metrics) in zip(cols, all_metrics.items()):
                with col:
                    st.markdown(f"**{mname}**")
                    for k, v in metrics.items():
                        col.metric(k, f"{v:.4f}")

        # Residual plot
        if all_metrics:
            st.subheader("Residuals (Actual − Predicted)")
            fig_res, ax_res = plt.subplots(figsize=(14, 3))
            for mname in models_to_show:
                mdl = assets.get(mname)
                if mdl:
                    y_pred_sc  = mdl.predict(X_test, verbose=0)
                    y_pred_inv = scaler_loaded.inverse_transform(y_pred_sc).flatten()
                    residuals  = y_test_inv - y_pred_inv
                    ax_res.plot(residuals, linewidth=0.8,
                                 label=mname, color=colors_map.get(mname, "green"))
            ax_res.axhline(0, color="grey", linestyle="--")
            ax_res.set_xlabel("Test Day"); ax_res.set_ylabel("Residual (USD)")
            ax_res.legend(); plt.tight_layout()
            st.pyplot(fig_res); plt.close(fig_res)

# ─────────────────────────────────────────────────────────────────────────
# TAB 3 — Future Forecast
# ─────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("🔮 Future Price Forecast")

    if not TF_AVAILABLE or (not assets.get("SimpleRNN") and not assets.get("LSTM Tuned")):
        st.warning("Train the models first and save weights. See Tab 2 for instructions.")
    else:
        last_date = df.index[-1]
        default_target = max(
            pd.Timestamp("2026-07-01"),
            pd.Timestamp(last_date).normalize() + pd.offsets.BDay(horizon)
        )

        forecast_mode = st.radio(
            "Forecast Type",
            ["Specific future date", "Next N trading days"],
            horizontal=True
        )

        if forecast_mode == "Specific future date":
            target_date = st.date_input(
                "Choose future date",
                value=default_target.date(),
                min_value=(pd.Timestamp(last_date).normalize() + pd.offsets.BDay(1)).date()
            )
            future_dates = get_future_trading_dates(last_date, target_date)
            forecast_steps = len(future_dates)
            forecast_label = f"through {pd.Timestamp(target_date).strftime('%B %d, %Y')}"
        else:
            forecast_steps = horizon
            future_dates = pd.bdate_range(start=last_date, periods=forecast_steps + 1)[1:]
            forecast_label = f"next {forecast_steps} trading days"

        if forecast_steps == 0:
            st.warning("Choose a date after the last date in the dataset.")
            st.stop()

        if forecast_steps > 750:
            st.info(
                f"This is a long-range iterative forecast: {forecast_steps:,} trading days "
                f"from the dataset end date ({last_date.date()}). It may take a little while."
            )

        close_scaled = scaler_loaded.transform(df[["Close"]].values)
        last_w = close_scaled[-window_size:]

        forecast_data = {}
        models_to_show = (
            ["SimpleRNN", "LSTM Tuned"] if selected_mdl == "Both"
            else [selected_mdl]
        )
        try:
            with st.spinner(f"Generating forecast {forecast_label}..."):
                for mname in models_to_show:
                    mdl = assets.get(mname)
                    if mdl:
                        forecast_data[mname] = predict_future(
                            mdl, last_w, forecast_steps, scaler_loaded, window_size
                        )
        except Exception as e:
            st.error(f"Forecast failed — likely a window-size mismatch with saved weights. Error: {e}")

        if forecast_data:
            fc_df = pd.DataFrame(forecast_data, index=future_dates.strftime("%Y-%m-%d"))
            fc_df.index.name = "Date"

            target_row = fc_df.iloc[-1]
            target_trading_date = future_dates[-1]
            st.caption(
                f"Dataset ends on {last_date.date()}. Forecast target trading date: "
                f"{target_trading_date.date()}."
            )

            metric_cols = st.columns(len(target_row))
            for col, (mname, value) in zip(metric_cols, target_row.items()):
                change = value - df["Close"].iloc[-1]
                pct_chg = change / df["Close"].iloc[-1] * 100
                col.metric(
                    f"{mname} on {target_trading_date.strftime('%Y-%m-%d')}",
                    f"${value:.2f}",
                    f"{change:+.2f} ({pct_chg:+.2f}%)"
                )

            if (
                forecast_mode == "Specific future date"
                and pd.Timestamp(target_date).normalize() != pd.Timestamp(target_trading_date).normalize()
            ):
                st.info(
                    "Stocks trade on business days, so weekend/holiday-style dates are shown as "
                    "the nearest prior generated trading day."
                )

            display_n = min(30, len(fc_df))
            st.dataframe(fc_df.tail(display_n).round(2), use_container_width=True)
            if len(fc_df) > display_n:
                with st.expander(f"View all {len(fc_df):,} forecasted trading days"):
                    st.dataframe(fc_df.round(2), use_container_width=True)

            context_n = min(60, len(df))
            context_prices = df["Close"].values[-context_n:]
            context_idx    = list(range(context_n))
            future_idx     = list(range(context_n - 1, context_n + forecast_steps))

            fig_fc, ax_fc = plt.subplots(figsize=(14, 5))
            ax_fc.plot(context_idx, context_prices, color="steelblue",
                       linewidth=2, label="Historical (last 60 days)")

            fc_colors = ["orange", "crimson", "green"]
            for (mname, fcast), col in zip(forecast_data.items(), fc_colors):
                y_vals = [context_prices[-1]] + list(fcast)
                marker = "o" if forecast_steps <= 60 else None
                ax_fc.plot(future_idx, y_vals, linestyle="--", marker=marker,
                           markersize=4, linewidth=1.5, color=col, label=f"{mname} Forecast")

            ax_fc.axvline(context_n - 1, color="grey", linestyle=":", linewidth=1.2,
                          label="Forecast Start")
            ax_fc.set_title(f"TSLA Forecast {forecast_label}", fontweight="bold")
            ax_fc.set_xlabel("Trading Days"); ax_fc.set_ylabel("Price (USD)")
            ax_fc.legend(); plt.tight_layout()
            st.pyplot(fig_fc); plt.close(fig_fc)

            for mname, fcast in forecast_data.items():
                direction = "📈 Upward" if fcast[-1] > context_prices[-1] else "📉 Downward"
                change    = fcast[-1] - context_prices[-1]
                pct_chg   = change / context_prices[-1] * 100
                st.info(f"**{mname}** — {forecast_steps:,}-trading-day outlook: **{direction}**  "
                        f"| Predicted change: **${change:+.2f} ({pct_chg:+.2f}%)**")

# ─────────────────────────────────────────────────────────────────────────
# TAB 4 — Model Comparison
# ─────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("📋 Model Architecture & Performance Comparison")

    arch_data = {
        "Property": ["Architecture", "RNN Units (L1)", "RNN Units (L2)",
                     "Dropout Rate", "Optimizer", "Loss Function",
                     "Lookback Window", "Training Epochs (max)", "Early Stopping"],
        "SimpleRNN":  ["2-Layer SimpleRNN", RNN_UNITS, RNN_UNITS // 2,
                       str(RNN_DROPOUT), "Adam (lr=0.001)",
                       "MSE", "60 days", 80, "patience=8"],
        "LSTM Tuned": ["2-Layer LSTM", BEST_LSTM_UNITS, BEST_LSTM_UNITS // 2,
                       str(BEST_LSTM_DROPOUT), "Adam (Best lr)", "MSE",
                       "60 days", 80, "patience=8"],
    }
    st.table(pd.DataFrame(arch_data).set_index("Property"))

    st.divider()
    st.subheader("Why LSTM for Stock Prediction?")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**SimpleRNN**
- ✅ Faster training
- ✅ Fewer parameters
- ✅ Competitive 1-step accuracy
- ❌ Vanishing gradient for long sequences
- ❌ Forgets distant past events
""")
    with col_b:
        st.markdown("""
**LSTM**
- ✅ Remembers long-term patterns (cell state)
- ✅ Smoother multi-step forecasts
- ✅ Better for volatile / event-driven sequences
- ❌ More parameters → slower training
- ❌ Requires more data to generalise
""")

    st.divider()
    st.subheader("Business Impact of Evaluation Metrics")
    impact = {
        "Metric": ["RMSE", "MAE", "MAPE", "R²"],
        "Business Meaning": [
            "Avg USD error (penalises large misses). RMSE ~$28 = ±$28 prediction band.",
            "Typical USD error per day — directly maps to trade P&L exposure.",
            "Scale-free error %; useful across price eras (low-price 2010 vs high 2020).",
            "Variance explained (~85%). High R² → model captures market trend well.",
        ],
        "Acceptable Threshold": ["< $30", "< $20", "< 10%", "> 0.80"],
    }
    st.table(pd.DataFrame(impact))

    st.caption("Dashboard built with Streamlit · Models trained with TensorFlow/Keras")
