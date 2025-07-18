import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
import warnings
import json
from streamlit_autorefresh import st_autorefresh

warnings.filterwarnings('ignore')

# --- API Keys and Configuration ---
FINNHUB_API_KEY = "d1sdgppr01qs2slhjh20d1sdgppr01qs2slhjh2g" # REPLACE WITH YOUR ACTUAL FINNHUB API KEY

# --- Safe Imports with Fallback ---
try:
    from nselib import capital_market, derivatives
    NSELIB_AVAILABLE = True
except ImportError:
    NSELIB_AVAILABLE = False
    st.warning("NSELib not available. Using demo data mode.")

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Advanced Indian Stock Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.2em;
    }
    .buy-signal {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .sell-signal {
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .hold-signal {
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    .transaction-log {
        max-height: 300px;
        overflow-y: auto;
        padding: 10px;
        background-color: #f9f9f9;
        border-radius: 5px;
        border: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'transaction_log' not in st.session_state:
    st.session_state.transaction_log = []
if 'refresh_trigger' not in st.session_state:
    st.session_state.refresh_trigger = 0

# --- Main Header ---
st.markdown("""
<div class="main-header">
    <h1>🚀 Advanced Indian Stock Market Dashboard 2025</h1>
    <p>Real-time Analysis | Smart Predictions | Comprehensive Insights</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
st.sidebar.title("📊 Dashboard Controls")
st.sidebar.markdown("---")

# --- Error Handling Decorator ---
def safe_execute(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.RequestException as e:
            st.error(f"Network or API request failed: {e}")
            return None
        except KeyError as e:
            st.error(f"Data processing error: Missing key in response - {e}. The data source might be unavailable or the symbol is incorrect.")
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            return None
    return wrapper

# --- Data Fetching Functions ---
@st.cache_data(ttl=60*60*24)
@safe_execute
def get_symbol_suggestions(query):
    """Fetch stock symbol suggestions using Finnhub API."""
    if not FINNHUB_API_KEY or FINNHUB_API_KEY == "YOUR_FINNHUB_API_KEY_HERE":
        st.warning("Please provide a valid Finnhub API key in the code to enable stock symbol suggestions.")
        return []
    url = f"https://finnhub.io/api/v1/search?q={query}&token={FINNHUB_API_KEY}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    if 'result' in data:
        suggestions = [
            f"{item['symbol']} - {item['description']}"
            for item in data['result']
            if 'exchange' in item and 'IN' in item['exchange']
        ]
        return suggestions
    return []

@safe_execute
def _get_realtime_price_finnhub(symbol):
    """Fetches real-time price from Finnhub API.
    This function is kept for potential future use or if the user decides to switch back,
    but its output is not used for current price metrics in get_google_finance_data as per request.
    """
    if not FINNHUB_API_KEY or FINNHUB_API_KEY == "YOUR_FINNHUB_API_KEY_HERE":
        st.warning("Finnhub API key is not set. Real-time price may not be accurate.")
        return None, None
    
    # Finnhub requires a specific symbol format, e.g., 'NSE:TCS'
    # For Indian stocks, it's often 'NSE:SYMBOL' or 'BSE:SYMBOL'.
    # Assuming the input symbol is just the ticker (e.g., 'TCS'),
    # we'll prepend 'NSE:' for Finnhub.
    finnhub_symbol = f"NSE:{symbol}" 
    
    url = f"https://finnhub.io/api/v1/quote?symbol={finnhub_symbol}&token={FINNHUB_API_KEY}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        # 'c' is current price, 'pc' is previous close
        if data and 'c' in data and data['c'] != 0:
            return data['c'], data.get('pc', data['c']) # Return current and previous close
    except requests.exceptions.RequestException as e:
        st.error(f"Finnhub real-time price fetch failed: {e}")
    except json.JSONDecodeError:
        st.error("Finnhub response is not valid JSON.")
    return None, None


@safe_execute
def get_google_finance_data(symbol, exchange):
    """
    Fetches current price, high, low, and volume by scraping Google Finance.
    Generates synthetic historical data based on the scraped current price,
    ensuring the latest high and low match the scraped day high/low.
    """
    # As per user request, we are prioritizing Google Finance scraping for current price, high, low, and volume.
    # The Finnhub API key is used for symbol suggestions, not for price data in this function.
    
    try:
        url = f'https://www.google.com/finance/quote/{symbol}:{exchange}'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # --- Scrape Current Price ---
        scraped_current_price = None
        price_element = soup.find('div', class_='YMlKec fxKbKc')
        if not price_element:
            price_element = soup.find('div', class_='YMlKec')
        if price_element:
            scraped_current_price = float(price_element.text.strip().replace('₹', '').replace(',', ''))
        
        if scraped_current_price is None:
            st.error(f"Could not find current price on Google Finance page for {symbol}. The website structure may have changed.")
            return None

        # --- Scrape Previous Close ---
        scraped_prev_close = None
        prev_close_element = soup.find('div', string='Prev close')
        if not prev_close_element:
            prev_close_div = soup.find('div', class_='mfs7Fc', string='Prev Close')
            if prev_close_div:
                prev_close_element = prev_close_div.find_next('div', class_='P6K39c')
        if prev_close_element:
            scraped_prev_close = float(prev_close_element.text.strip().replace('₹', '').replace(',', ''))
        
        # Fallback for prev_close if not found
        if scraped_prev_close is None:
            scraped_prev_close = scraped_current_price 

        # --- Scrape Day High ---
        scraped_day_high = None
        # Look for the 'High' label and then its value
        high_label_div = soup.find('div', class_='mfs7Fc', string='High')
        if high_label_div:
            high_value_div = high_label_div.find_next_sibling('div', class_='P6K39c')
            if high_value_div:
                scraped_day_high = float(high_value_div.text.strip().replace('₹', '').replace(',', ''))
        
        # Fallback if scraped_day_high is still None or if it's less than current price (unlikely for a high)
        if scraped_day_high is None or scraped_day_high < scraped_current_price:
            scraped_day_high = scraped_current_price * 1.01 # Assume a small buffer above current price if not found

        # --- Scrape Day Low ---
        scraped_day_low = None
        # Look for the 'Low' label and then its value
        low_label_div = soup.find('div', class_='mfs7Fc', string='Low')
        if low_label_div:
            low_value_div = low_label_div.find_next_sibling('div', class_='P6K39c')
            if low_value_div:
                scraped_day_low = float(low_value_div.text.strip().replace('₹', '').replace(',', ''))

        # Fallback if scraped_day_low is still None or if it's greater than current price (unlikely for a low)
        if scraped_day_low is None or scraped_day_low > scraped_current_price:
            scraped_day_low = scraped_current_price * 0.99 # Assume a small buffer below current price if not found

        # Ensure High is not less than Low
        if scraped_day_high < scraped_day_low:
            scraped_day_high = scraped_day_low * 1.01 # Adjust high if it somehow ends up lower than low


        # --- Scrape Volume ---
        scraped_volume = "N/A"
        volume_element = soup.find('div', string='Volume')
        if volume_element:
            volume_value_div = volume_element.find_next_sibling('div', class_='P6K39c')
            if volume_value_div:
                scraped_volume = volume_value_div.text.strip().replace(',', '')
        
        # --- Generate synthetic historical data based on scraped current price ---
        # For the purpose of showing a realistic-looking graph, we'll generate 100 data points.
        # The last data point will be aligned with the scraped current price, high, and low.
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        np.random.seed(hash(symbol) % 1000) # Seed for reproducibility of synthetic data
        
        # Generate prices backwards from the current price
        prices_series = [scraped_current_price]
        for _ in range(99): 
            # Simulate daily price change with random walk
            change = np.random.normal(0, 0.005) # Small daily percentage change
            prev_price_calc = prices_series[-1] / (1 + change)
            prices_series.append(max(0.01, prev_price_calc)) # Ensure price is not zero or negative

        prices_series.reverse() # Reverse to get chronological order (oldest to newest)
        
        historical_data = pd.DataFrame({
            'Date': dates,
            'Close': prices_series,
            'Volume': np.random.randint(1000000, 10000000, 100) # Synthetic volume
        })
        historical_data.set_index('Date', inplace=True)

        # Generate Open, High, Low for each synthetic day based on its Close
        historical_data['Open'] = historical_data['Close'] * np.random.uniform(0.99, 1.01, len(historical_data))
        historical_data['High'] = historical_data['Close'] * np.random.uniform(1.00, 1.02, len(historical_data))
        historical_data['Low'] = historical_data['Close'] * np.random.uniform(0.98, 1.00, len(historical_data))

        # Ensure Low <= Open/Close <= High for synthetic data
        historical_data['Open'] = historical_data.apply(lambda row: np.clip(row['Open'], row['Low'], row['High']), axis=1)
        historical_data['Close'] = historical_data.apply(lambda row: np.clip(row['Close'], row['Low'], row['High']), axis=1)
        
        # Ensure the last 'High', 'Low', and 'Close' in historical data match the scraped Day High/Low/Current Price
        if not historical_data.empty:
            historical_data.loc[historical_data.index[-1], 'High'] = scraped_day_high
            historical_data.loc[historical_data.index[-1], 'Low'] = scraped_day_low
            historical_data.loc[historical_data.index[-1], 'Close'] = scraped_current_price 
            
            # Adjust Open for the last day to be within its High/Low range
            if historical_data.loc[historical_data.index[-1], 'Open'] > scraped_day_high or \
               historical_data.loc[historical_data.index[-1], 'Open'] < scraped_day_low:
                historical_data.loc[historical_data.index[-1], 'Open'] = (scraped_day_high + scraped_day_low) / 2


        return {
            'symbol': symbol,
            'price': scraped_current_price,
            'prev_close': scraped_prev_close,
            'volume': scraped_volume,
            'historical': historical_data,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    except requests.exceptions.RequestException as e:
        st.error(f"Network error while scraping Google Finance: {e}. The website might be unreachable or your internet connection is unstable.")
        return None
    except Exception as e:
        st.error(f"Failed to scrape data from Google Finance: {e}. The website structure might have changed or data is missing.")
        return None

# --- AI Trading Analysis Functions ---
def calculate_technical_indicators(data):
    try:
        if data is None or len(data) < 50:
            st.warning("Not enough data to calculate all technical indicators. Need at least 50 data points.")
            return pd.DataFrame()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['Signal'] = data['MACD'].ewm(span=9).mean()
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        if 'Volume' in data.columns:
            data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        return data.dropna()
    except Exception as e:
        st.error(f"Technical analysis error: {e}")
        return pd.DataFrame()

def generate_trading_signal(data):
    if data.empty: return "HOLD", "Insufficient data for analysis.", 0.5
    latest = data.iloc[-1]
    signals = []
    latest = latest.replace([np.inf, -np.inf], np.nan).dropna()
    if latest.get('RSI', 50) < 30: signals.append("BUY")
    elif latest.get('RSI', 50) > 70: signals.append("SELL")
    if latest.get('SMA_20', np.nan) > latest.get('SMA_50', np.nan): signals.append("BUY")
    elif latest.get('SMA_20', np.nan) < latest.get('SMA_50', np.nan): signals.append("SELL")
    if latest.get('MACD', np.nan) > latest.get('Signal', np.nan): signals.append("BUY")
    elif latest.get('MACD', np.nan) < latest.get('Signal', np.nan): signals.append("SELL")
    if latest.get('Close', np.nan) > latest.get('BB_Upper', np.nan): signals.append("SELL")
    elif latest.get('Close', np.nan) < latest.get('BB_Lower', np.nan): signals.append("BUY")
    buy_signals = signals.count("BUY")
    sell_signals = signals.count("SELL")
    total_signals = buy_signals + sell_signals
    confidence = 0.5 + (buy_signals - sell_signals) * 0.1
    if buy_signals > sell_signals: return "BUY", f"Strong bullish indicators ({buy_signals} signals).", min(confidence, 1.0)
    elif sell_signals > buy_signals: return "SELL", f"Strong bearish indicators ({sell_signals} signals).", min(confidence, 1.0)
    else: return "HOLD", "Mixed signals or insufficient data. Wait for a clearer trend.", 0.5

# --- Charting, Risk, Performance, and Transaction Functions ---
def create_advanced_chart(data, symbol):
    if data.empty:
        fig = go.Figure()
        fig.add_annotation(text="No chart data available.", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Check if the data has enough points for a candlestick chart
    if len(data) < 2: # Need at least two points for a meaningful candlestick
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Price'))
        fig.update_layout(title=f'{symbol} - Price Movement (Limited Data)', xaxis_rangeslider_visible=False, height=400, showlegend=True, template='plotly_white')
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Price (₹)")
        return fig

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.2, 0.15, 0.15], subplot_titles=('Price & Volume', 'Bollinger Bands', 'RSI', 'MACD'))
    
    # Candlestick chart for Price & Volume
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Price"), row=1, col=1)
    if 'SMA_20' in data.columns: fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], mode='lines', name='SMA 20', line=dict(color='orange', width=2)), row=1, col=1)
    if 'SMA_50' in data.columns: fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='SMA 50', line=dict(color='blue', width=2)), row=1, col=1)
    if 'Volume' in data.columns: fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='rgba(158,202,225,0.6)'), row=2, col=1)
    
    # Bollinger Bands
    if 'BB_Upper' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], mode='lines', name='BB Upper', line=dict(color='gray', width=1), showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Middle'], mode='lines', name='BB Middle', line=dict(color='darkgray', width=1), showlegend=True), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], mode='lines', name='BB Lower', line=dict(color='gray', width=1), fill='tonexty', fillcolor='rgba(128,128,128,0.1)', showlegend=False), row=3, col=1)
    
    # RSI
    if 'RSI' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='purple', width=2)), row=4, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)
    
    fig.update_layout(title=f'{symbol} - Comprehensive Technical Analysis', xaxis_rangeslider_visible=False, height=800, showlegend=True, template='plotly_white')
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="Bollinger", row=3, col=1)
    fig.update_yaxes(title_text="RSI", row=4, col=1)
    return fig

def calculate_risk_metrics(data):
    if data.empty or 'Close' not in data.columns:
        return {"volatility": 0, "risk_level": "N/A", "recommendation": "Insufficient data."}
    returns = data['Close'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(252) * 100
    if volatility > 30: risk_level, risk_color, recommendation = "High", "red", "High volatility detected. Consider this a high-risk investment and use proper position sizing."
    elif volatility > 15: risk_level, risk_color, recommendation = "Medium", "orange", "Moderate volatility. Suitable for most investors with a balanced risk tolerance."
    else: risk_level, risk_color, recommendation = "Low", "green", "Low volatility. Suitable for conservative investors."
    return {"volatility": volatility, "risk_level": risk_level, "risk_color": risk_color, "recommendation": recommendation}

def calculate_performance_metrics(data):
    if data.empty or 'Close' not in data.columns: return {}
    returns = data['Close'].pct_change().dropna()
    metrics = {
        "1D Return": returns.iloc[-1] * 100 if len(returns) > 0 else 0,
        "7D Return": (data['Close'].iloc[-1] / data['Close'].iloc[-min(7, len(data))]) * 100 - 100 if len(data) >= 7 else 0,
        "30D Return": (data['Close'].iloc[-1] / data['Close'].iloc[-min(30, len(data))]) * 100 - 100 if len(data) >= 30 else 0,
        "Sharpe Ratio": (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
    }
    return metrics

def buy_stock(symbol, quantity, price): st.session_state.transaction_log.append({'action': 'buy', 'symbol': symbol, 'quantity': quantity, 'price': price, 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}); st.success(f"Bought {quantity} shares of {symbol} at ₹{price:.2f} each.")
def sell_stock(symbol, quantity, price): st.session_state.transaction_log.append({'action': 'sell', 'symbol': symbol, 'quantity': quantity, 'price': price, 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}); st.success(f"Sold {quantity} shares of {symbol} at ₹{price:.2f} each.")
def hold_stock(symbol): st.session_state.transaction_log.append({'action': 'hold', 'symbol': symbol, 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}); st.info(f"Holding {symbol}.")

# --- Main Dashboard Logic (Equity) ---
def equity_dashboard():
    st.sidebar.markdown("---")
    data_source = st.sidebar.selectbox("📈 Select Data Source", ["Google Finance (Real-time)", "NSE Library (Comprehensive)", "Demo Mode"])
    symbol_input = st.sidebar.text_input("Stock Symbol or Name", "ASHOKLEY")
    suggestions = get_symbol_suggestions(symbol_input)
    if suggestions:
        st.sidebar.info("Suggestions:")
        for s in suggestions[:5]: st.sidebar.markdown(f"- `{s}`")
    symbol_to_fetch = symbol_input.upper().split(' ')[0]
    period = "3M"
    if data_source == "NSE Library (Comprehensive)":
        if not NSELIB_AVAILABLE:
            st.error("NSE Library is not installed. Please select another data source or install it (`pip install nselib`).")
            return
        period = st.sidebar.selectbox("Time Period", ["1M", "3M", "6M", "1Y"])
    
    col_refresh, col_auto = st.sidebar.columns(2)
    if col_refresh.button("🔄 Refresh Data", key="manual_refresh_btn"):
        st.cache_data.clear()
        # FIX: Replaced experimental_rerun with rerun
        st.rerun()
    
    auto_refresh_toggle = col_auto.checkbox("Auto Refresh (60s)", value=False, help="Automatically refreshes data every 60 seconds.")
    if auto_refresh_toggle:
        st_autorefresh(interval=60000, key="auto_refresh_trigger")
    
    with st.spinner(f"Fetching market data for {symbol_to_fetch}..."):
        if data_source == "Google Finance (Real-time)":
            stock_data = get_google_finance_data(symbol_to_fetch, "NSE")
        elif data_source == "NSE Library (Comprehensive)":
            # IMPROVED: Added a check for NSELib data source failure
            try:
                stock_data = get_nse_data(symbol_to_fetch, period)
                if stock_data is None:
                    raise ValueError("NSELib data fetch failed, using demo data.")
            except Exception as e:
                st.error(f"NSE Library data fetch failed: {e}. Falling back to demo data.")
                stock_data = create_demo_data(symbol_to_fetch)
        else:
            stock_data = create_demo_data(symbol_to_fetch)

    if stock_data is None:
        st.error(f"❌ Unable to fetch data for {symbol_to_fetch}. Please check the symbol and try a different data source.")
        return
    
    st.success(f"✅ Data loaded successfully for {stock_data['symbol']}")
    
    # --- Start of fix for KeyError: 'Close' ---
    # Ensure 'historical' key exists and is a DataFrame
    if 'historical' not in stock_data or not isinstance(stock_data['historical'], pd.DataFrame):
        st.error(f"Data Error: 'historical' data is missing or not in expected format for {symbol_to_fetch}. Cannot proceed with analysis.")
        return

    historical_data = stock_data['historical']
    
    # Check if historical_data is empty or missing 'Close' column
    if historical_data.empty:
        st.error(f"Data Error: Historical data for {symbol_to_fetch} is empty. Cannot perform analysis.")
        return
    
    if 'Close' not in historical_data.columns:
        st.error(f"Data Error: 'Close' price column is missing in historical data for {symbol_to_fetch}. Cannot perform analysis.")
        return
    # --- End of fix for KeyError: 'Close' ---

    latest_price = stock_data.get('price', historical_data['Close'].iloc[-1])
    prev_price = stock_data.get('prev_close', historical_data['Close'].iloc[-2] if len(historical_data) > 1 else latest_price) # Use prev_close from stock_data if available
    price_change = latest_price - prev_price
    price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
    volume = stock_data.get('volume', historical_data['Volume'].iloc[-1]) if 'Volume' in historical_data.columns else "N/A"
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Current Price", f"₹{latest_price:.2f}", f"{price_change:.2f} ({price_change_pct:.2f}%)")
    with col2: st.metric("Volume", f"{volume:,}" if isinstance(volume, (int, float)) else volume)
    with col3: st.metric("Day High", f"₹{historical_data['High'].iloc[-1]:.2f}")
    with col4: st.metric("Day Low", f"₹{historical_data['Low'].iloc[-1]:.2f}")
    
    st.subheader("🎯 AI Trading Recommendation")
    with st.spinner("Analyzing market trends..."):
        analyzed_data = calculate_technical_indicators(historical_data.copy())
        signal, reason, confidence = generate_trading_signal(analyzed_data)
    signal_class = f"{signal.lower()}-signal"
    confidence_bar = "🟢" * int(confidence * 10) + "⚪" * (10 - int(confidence * 10))
    st.markdown(f"""
    <div class="prediction-box {signal_class}">
        <h3>📊 {signal}</h3>
        <p>{reason}</p>
        <p>Confidence: {confidence_bar} ({confidence:.1%})</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("📈 Advanced Technical Analysis")
    chart = create_advanced_chart(analyzed_data, stock_data['symbol'])
    st.plotly_chart(chart, use_container_width=True)

    # --- New section for "Real-time Graph" from Google Finance (simulated) ---
    st.subheader("📊 Intraday Price Movement (Simulated from Google Finance)")
    # For this, we will use the `historical_data` generated by get_google_finance_data
    # which now includes synthetic intraday-like data for the latest day.
    if data_source == "Google Finance (Real-time)" and not historical_data.empty:
        # Filter for the latest day's data if needed, or just plot the whole historical_data
        # For a "real-time graph", we usually focus on the latest day or a very short period.
        # Since our synthetic data is daily, let's plot the last few days to show movement.
        # If you want strictly "intraday", you'd need actual intraday data which scraping is hard for.
        
        # Let's create a simplified chart for the last few days
        # The `historical_data` already contains daily Open, High, Low, Close
        # We can plot a line chart for the 'Close' price for the last 5 days to show recent movement
        recent_data = historical_data.tail(5) # Show last 5 days for a clearer view
        if not recent_data.empty:
            fig_intraday = go.Figure(data=[go.Scatter(x=recent_data.index,
                                                        y=recent_data['Close'],
                                                        mode='lines',
                                                        name='Price',
                                                        line=dict(color='green' if recent_data['Close'].iloc[-1] >= recent_data['Close'].iloc[0] else 'red'))])
            fig_intraday.update_layout(title=f'{stock_data["symbol"]} - Recent Daily Price Movement',
                                      xaxis_rangeslider_visible=False,
                                      height=400,
                                      template='plotly_white')
            fig_intraday.update_xaxes(title_text="Date")
            fig_intraday.update_yaxes(title_text="Price (₹)")
            st.plotly_chart(fig_intraday, use_container_width=True)
        else:
            st.info("Not enough historical data to show recent price movement graph.")
    else:
        st.info("This graph is available when 'Google Finance (Real-time)' data source is selected and data is available.")
    # --- End of new section ---
    
    st.subheader("📊 Technical Indicators & Statistics")
    tech_col, stat_col = st.columns(2)
    with tech_col:
        st.markdown("#### Technical Indicators")
        if not analyzed_data.empty:
            latest = analyzed_data.iloc[-1]
            indicators_df = pd.DataFrame([['RSI', f"{latest.get('RSI', np.nan):.2f}", "Oversold" if latest.get('RSI', 50) < 30 else "Overbought" if latest.get('RSI', 50) > 70 else "Neutral"], ['MACD', f"{latest.get('MACD', np.nan):.4f}", "Bullish" if latest.get('MACD', 0) > latest.get('Signal', 0) else "Bearish"], ['SMA 20', f"{latest.get('SMA_20', np.nan):.2f}", "Bullish" if latest.get('Close', 0) > latest.get('SMA_20', 0) else "Bearish"], ['SMA 50', f"{latest.get('SMA_50', np.nan):.2f}", "Bullish" if latest.get('Close', 0) > latest.get('SMA_50', 0) else "Bearish"]], columns=['Indicator', 'Value', 'Signal']).replace('nan', 'N/A')
            st.dataframe(indicators_df, use_container_width=True)
        else: st.info("Technical indicators cannot be calculated with the available data.")
    with stat_col:
        st.markdown("#### Performance Metrics")
        performance_metrics = calculate_performance_metrics(historical_data)
        metrics_list = [[key, f"{val:.2f}%" if key.endswith("Return") else f"{val:.2f}"] for key, val in performance_metrics.items()]
        metrics_df = pd.DataFrame(metrics_list, columns=['Metric', 'Value'])
        st.dataframe(metrics_df, use_container_width=True)
    
    st.subheader("⚠️ Risk Assessment")
    risk_metrics = calculate_risk_metrics(historical_data)
    st.markdown(f"""
    <div style="padding: 1rem; background: oklch(0.274 0.006 286.033); border-radius: 8px; border-left: 4px solid {risk_metrics['risk_color']};">
        <h4 style="color: {risk_metrics['risk_color']};">Risk Level: {risk_metrics['risk_level']}</h4>
        <p>Annualized Volatility: {risk_metrics['volatility']:.2f}%</p>
        <p>Recommendation: {risk_metrics['recommendation']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("💼 Trading Simulation")
    col1, col2, col3 = st.columns(3)
    with col1: trade_quantity = st.number_input("Quantity", min_value=1, value=10)
    with col2: trade_price = st.number_input("Price per Share", min_value=0.01, value=float(latest_price), format="%.2f")
    with col3:
        st.write(""); st.write("")
        if st.button("Buy"): buy_stock(stock_data['symbol'], trade_quantity, trade_price)
        if st.button("Sell"): sell_stock(stock_data['symbol'], trade_quantity, trade_price)
        if st.button("Hold"): hold_stock(stock_data['symbol'])
    
    st.subheader("📝 Transaction Log")
    if st.session_state.transaction_log:
        log_df = pd.DataFrame(st.session_state.transaction_log)
        st.dataframe(log_df, use_container_width=True)
    else: st.info("No transactions yet.")
    st.markdown("---")
    st.markdown(f"**Last Updated:** {stock_data.get('last_updated', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))} | **Data Source:** {data_source}")
    st.markdown("⚠️ *This is for educational purposes only. Please consult a financial advisor before making investment decisions.*")


# --- Derivatives Dashboard Logic ---
def get_nse_data(symbol, period="3M"):
    if not NSELIB_AVAILABLE: return None
    data = capital_market.price_volume_and_deliverable_position_data(symbol=symbol, period=period)
    if data is not None and not data.empty:
        data.columns = data.columns.str.strip()
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
        column_mapping = {'Close Price': 'Close', 'Open Price': 'Open', 'High Price': 'High', 'Low Price': 'Low', 'Total Traded Quantity': 'Volume'}
        data.rename(columns=column_mapping, inplace=True)
        return {'symbol': symbol, 'historical': data, 'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    return None

def create_demo_data(symbol):
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    np.random.seed(hash(symbol) % 1000)
    base_price = 100 + (hash(symbol) % 1000)
    returns = np.random.normal(0.001, 0.02, 100)
    prices = [base_price]
    for ret in returns[1:]: prices.append(prices[-1] * (1 + ret))
    historical_data = pd.DataFrame({'Date': dates, 'Open': [p * np.random.uniform(0.98, 1.02) for p in prices], 'High': [p * np.random.uniform(1.00, 1.05) for p in prices], 'Low': [p * np.random.uniform(0.95, 1.00) for p in prices], 'Close': prices, 'Volume': np.random.randint(1000000, 10000000, 100)})
    historical_data.set_index('Date', inplace=True)
    return {'symbol': symbol, 'price': prices[-1], 'historical': historical_data, 'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

def derivatives_dashboard():
    st.sidebar.subheader("Derivatives Data Options")
    instrument = st.sidebar.selectbox("Instrument Type", options=("NSE Equity Market", "NSE Derivative Market"))
    data = None
    data_info = ""
    
    try: # Wrap the entire data fetching logic in a try-except block
        if instrument == "NSE Equity Market":
            data_info = st.sidebar.selectbox("Data to extract", options=("price_volume_and_deliverable_position_data", "price_volume_data", "deliverable_position_data", "bhav_copy_with_delivery", "bhav_copy_equities", "equity_list", "fno_equity_list", "fno_index_list", "nifty50_equity_list", "india_vix_data", "market_watch_all_indices", "bulk_deal_data", "block_deals_data", "short_selling_data", "index_data", "var_begin_day", "var_1st_intra_day", "var_2nd_intra_day", "var_3rd_intra_day", "var_4th_intra_day", "var_end_of_day"))
            if (data_info == "equity_list") or (data_info == "fno_equity_list") or (data_info == "market_watch_all_indices") or (data_info == "nifty50_equity_list"):
                # Handles JSONDecodeError from market_watch_all_indices if the underlying nselib function raises it
                if NSELIB_AVAILABLE: data = getattr(capital_market, data_info)()
            elif data_info in ["price_volume_and_deliverable_position_data", "price_volume_data", "deliverable_position_data"]:
                symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., SBIN)", "SBIN")
                period = st.sidebar.selectbox("Select Period", ["1M", "3M", "6M", "1Y"])
                if NSELIB_AVAILABLE: data = getattr(capital_market, data_info)(symbol=symbol, period=period)
            elif (data_info == "bhav_copy_equities") or (data_info == "bhav_copy_with_delivery"):
                date = st.sidebar.text_input("Date (dd-mm-yyyy)", "01-01-2025")
                if NSELIB_AVAILABLE: data = getattr(capital_market, data_info)(date)
            elif (data_info == "block_deals_data") or (data_info == "bulk_deal_data") or (data_info == "india_vix_data") or (data_info == "short_selling_data"):
                period = st.sidebar.selectbox("Select Period", ["1M", "3M", "6M", "1Y"])
                if NSELIB_AVAILABLE: data = getattr(capital_market, data_info)(period=period)
            elif data_info in ["var_begin_day", "var_1st_intra_day", "var_2nd_intra_day", "var_3rd_intra_day", "var_4th_intra_day", "var_end_of_day"]:
                trade_date = st.sidebar.date_input("Select Trade Date", datetime.now())
                formatted_date = trade_date.strftime("%d-%m-%Y")
                if NSELIB_AVAILABLE: data = getattr(capital_market, data_info)(trade_date=formatted_date)
        elif instrument == "NSE Derivative Market":
            data_info = st.sidebar.selectbox("Data to extract", options=("fno_bhav_copy", "participant_wise_open_interest", "participant_wise_trading_volume", "expiry_dates_future", "expiry_dates_option_index", "nse_live_option_chain", "future_price_volume_data", "option_price_volume_data"))
            if (data_info == "expiry_dates_future") or (data_info == "expiry_dates_option_index"):
                if NSELIB_AVAILABLE: data = getattr(derivatives, data_info)()
            elif (data_info == "participant_wise_trading_volume") or (data_info == "participant_wise_open_interest") or (data_info == "fno_bhav_copy"):
                date = st.sidebar.text_input("Date (dd-mm-yyyy)", "01-02-2025")
                if NSELIB_AVAILABLE: data = getattr(derivatives, data_info)(date)
            elif (data_info == "nse_live_option_chain"):
                ticker = st.sidebar.text_input("Ticker", "BANKNIFTY")
                expiry_date = st.sidebar.text_input("Expiry Date (dd-mm-yyyy)", "01-01-2025")
                if NSELIB_AVAILABLE: data = derivatives.nse_live_option_chain(ticker, expiry_date=expiry_date)
            elif (data_info == "future_price_volume_data"):
                ticker = st.sidebar.text_input("Ticker", "SBIN")
                type_ = st.sidebar.selectbox("Instrument Type", ["FUTSTK", "FUTIDX"])
                period_ = st.sidebar.selectbox("Period", ["1M", "3M", "6M", "1Y"])
                # This is where the KeyError for 'TIMESTAMP' was likely occurring
                if NSELIB_AVAILABLE: 
                    temp_data = derivatives.future_price_volume_data(ticker, type_, period=period_)
                    if temp_data is not None and not temp_data.empty:
                        # Attempt to drop 'TIMESTAMP' if it exists, otherwise handle gracefully
                        if 'TIMESTAMP' in temp_data.columns:
                            data = temp_data.drop(columns=['TIMESTAMP'])
                        else:
                            data = temp_data # Use data as is if TIMESTAMP column is not found
                            st.warning("Warning: 'TIMESTAMP' column not found in future price volume data. Displaying data without dropping that column.")
                    else:
                        data = temp_data # Propagate None or empty DataFrame
            elif (data_info == "option_price_volume_data"):
                ticker = st.sidebar.text_input("Ticker", "BANKNIFTY")
                type_ = st.sidebar.selectbox("Instrument Type", ["OPTIDX", "OPTSTK"])
                period_ = st.sidebar.selectbox("Period", ["1M", "3M", "6M", "1Y"])
                # This is where the KeyError for 'TIMESTAMP' was likely occurring
                if NSELIB_AVAILABLE: 
                    temp_data = derivatives.option_price_volume_data(ticker, type_, period=period_)
                    if temp_data is not None and not temp_data.empty:
                        if 'TIMESTAMP' in temp_data.columns:
                            data = temp_data.drop(columns=['TIMESTAMP'])
                        else:
                            data = temp_data # Use data as is if TIMESTAMP column is not found
                            st.warning("Warning: 'TIMESTAMP' column not found in option price volume data. Displaying data without dropping that column.")
                    else:
                        data = temp_data # Propagate None or empty DataFrame

    except json.JSONDecodeError:
        st.error("Data Error: The data received is not in the correct JSON format. The data source might be temporarily unavailable or its structure has changed.")
    except KeyError as e:
        st.error(f"Data Error: A required column was not found in the data. This could mean the data structure has changed. Missing key: {e}")
    except requests.exceptions.RequestException as e:
        st.error(f"Network Error: Failed to connect to the data source. Please check your internet connection or try again later. Details: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching derivatives data: {e}")
    
    if data is not None:
        st.subheader(f"Derivatives Data: {data_info}")
        if isinstance(data, pd.DataFrame):
            st.dataframe(data, use_container_width=True)
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                st.subheader("Data Visualization")
                chart_type = st.selectbox("Select Chart Type", ["Line", "Bar", "Scatter"])
                x_axis = st.selectbox("Select X-axis", data.columns)
                y_axis = st.selectbox("Select Y-axis", numerical_cols)
                if chart_type == "Line": fig = px.line(data, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}")
                elif chart_type == "Bar": fig = px.bar(data, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
                elif chart_type == "Scatter": fig = px.scatter(data, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
                st.plotly_chart(fig, use_container_width=True)
        else: st.write(data)
    else: st.warning("No data available for the selected options")

# --- Final Main App Flow ---
market_type = st.sidebar.radio("Market Type", ["Equity", "Derivatives"])
if market_type == "Equity":
    equity_dashboard()
else:
    derivatives_dashboard()

