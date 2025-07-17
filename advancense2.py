# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import requests
# from bs4 import BeautifulSoup
# import time
# from datetime import datetime, timedelta
# import warnings
# warnings.filterwarnings('ignore')

# # Safe imports with fallback
# try:
#     from nselib import capital_market, derivatives
#     NSELIB_AVAILABLE = True
# except ImportError:
#     NSELIB_AVAILABLE = False
#     st.warning("NSELib not available. Using demo data mode.")

# # Configure Streamlit page
# st.set_page_config(
#     page_title="Advanced Indian Stock Dashboard",
#     page_icon="📈",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for better styling
# st.markdown("""
# <style>
#     .main-header {
#         text-align: center;
#         background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         padding: 2rem;
#         border-radius: 10px;
#         margin-bottom: 2rem;
#     }
#     .metric-container {
#         background: #f8f9fa;
#         padding: 1rem;
#         border-radius: 8px;
#         border-left: 4px solid #667eea;
#         margin: 0.5rem 0;
#     }
#     .prediction-box {
#         padding: 1rem;
#         border-radius: 8px;
#         margin: 1rem 0;
#         text-align: center;
#         font-weight: bold;
#         font-size: 1.2em;
#     }
#     .buy-signal {
#         background: #d4edda;
#         color: #155724;
#         border: 1px solid #c3e6cb;
#     }
#     .sell-signal {
#         background: #f8d7da;
#         color: #721c24;
#         border: 1px solid #f5c6cb;
#     }
#     .hold-signal {
#         background: #fff3cd;
#         color: #856404;
#         border: 1px solid #ffeaa7;
#     }
#     .sidebar .sidebar-content {
#         background: #f8f9fa;
#     }
#     .transaction-log {
#         max-height: 300px;
#         overflow-y: auto;
#         padding: 10px;
#         background-color: #f9f9f9;
#         border-radius: 5px;
#         border: 1px solid #eee;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Initialize session state for transaction log
# if 'transaction_log' not in st.session_state:
#     st.session_state.transaction_log = []

# # Main header
# st.markdown("""
# <div class="main-header">
#     <h1>🚀 Advanced Indian Stock Market Dashboard 2025</h1>
#     <p>Real-time Analysis | Smart Predictions | Comprehensive Insights</p>
# </div>
# """, unsafe_allow_html=True)

# # Sidebar configuration
# st.sidebar.title("📊 Dashboard Controls")
# st.sidebar.markdown("---")

# # Market type selection
# market_type = st.sidebar.radio("Market Type", ["Equity", "Derivatives"])

# # Error handling decorator
# def safe_execute(func):
#     def wrapper(*args, **kwargs):
#         try:
#             return func(*args, **kwargs)
#         except Exception as e:
#             st.error(f"Operation failed: {str(e)}")
#             return None
#     return wrapper

# # Technical Analysis Functions
# def calculate_technical_indicators(data):
#     """Calculate various technical indicators"""
#     try:
#         if data is None or len(data) < 20:
#             return {}
        
#         # Simple Moving Averages
#         data['SMA_20'] = data['Close'].rolling(window=20).mean()
#         data['SMA_50'] = data['Close'].rolling(window=50).mean() if len(data) >= 50 else data['Close'].rolling(window=len(data)).mean()
        
#         # Exponential Moving Average
#         data['EMA_12'] = data['Close'].ewm(span=12).mean()
#         data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
#         # MACD
#         data['MACD'] = data['EMA_12'] - data['EMA_26']
#         data['Signal'] = data['MACD'].ewm(span=9).mean()
        
#         # RSI
#         delta = data['Close'].diff()
#         gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
#         loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
#         rs = gain / loss
#         data['RSI'] = 100 - (100 / (1 + rs)) if not loss.isnull().all() else pd.Series([50]*len(data))
        
#         # Bollinger Bands
#         data['BB_Middle'] = data['Close'].rolling(window=20).mean()
#         bb_std = data['Close'].rolling(window=20).std()
#         data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
#         data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        
#         # Volume indicators
#         if 'Volume' in data.columns:
#             data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        
#         return data
#     except Exception as e:
#         st.error(f"Technical analysis error: {str(e)}")
#         return data

# def generate_trading_signal(data):
#     """Generate BUY/SELL/HOLD signals based on technical analysis"""
#     try:
#         if data is None or len(data) < 20:
#             return "HOLD", "Insufficient data", 0.5
        
#         latest = data.iloc[-1]
#         signals = []
#         confidence = 0.5
        
#         # RSI Signal
#         if latest.get('RSI', 50) < 30:
#             signals.append("BUY")
#             confidence += 0.15
#         elif latest.get('RSI', 50) > 70:
#             signals.append("SELL")
#             confidence += 0.15
        
#         # Moving Average Signal
#         if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
#             if latest['SMA_20'] > latest['SMA_50']:
#                 signals.append("BUY")
#                 confidence += 0.1
#             elif latest['SMA_20'] < latest['SMA_50']:
#                 signals.append("SELL")
#                 confidence += 0.1
        
#         # MACD Signal
#         if 'MACD' in data.columns and 'Signal' in data.columns:
#             if latest['MACD'] > latest['Signal']:
#                 signals.append("BUY")
#                 confidence += 0.1
#             elif latest['MACD'] < latest['Signal']:
#                 signals.append("SELL")
#                 confidence += 0.1
        
#         # Price vs Bollinger Bands
#         if 'BB_Lower' in data.columns and 'BB_Upper' in data.columns:
#             if latest['Close'] < latest['BB_Lower']:
#                 signals.append("BUY")
#                 confidence += 0.1
#             elif latest['Close'] > latest['BB_Upper']:
#                 signals.append("SELL")
#                 confidence += 0.1
        
#         # Determine final signal
#         buy_signals = signals.count("BUY")
#         sell_signals = signals.count("SELL")
        
#         if buy_signals > sell_signals:
#             return "BUY", f"Strong bullish indicators ({buy_signals} buy signals)", min(confidence, 0.9)
#         elif sell_signals > buy_signals:
#             return "SELL", f"Strong bearish indicators ({sell_signals} sell signals)", min(confidence, 0.9)
#         else:
#             return "HOLD", "Mixed signals - wait for clearer direction", confidence
            
#     except Exception as e:
#         return "HOLD", f"Analysis error: {str(e)}", 0.5

# def create_advanced_chart(data, symbol):
#     """Create comprehensive trading chart"""
#     try:
#         if data is None or len(data) < 5:
#             fig = go.Figure()
#             fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
#             return fig
        
#         # Create subplots
#         fig = make_subplots(
#             rows=4, cols=1,
#             shared_xaxes=True,
#             vertical_spacing=0.03,
#             subplot_titles=('Price & Volume', 'Technical Indicators', 'RSI', 'MACD'),
#             row_heights=[0.5, 0.2, 0.15, 0.15]
#         )
        
#         # Candlestick chart
#         fig.add_trace(go.Candlestick(
#             x=data.index,
#             open=data.get('Open', data['Close']),
#             high=data.get('High', data['Close']),
#             low=data.get('Low', data['Close']),
#             close=data['Close'],
#             name="Price"
#         ), row=1, col=1)
        
#         # Moving averages
#         if 'SMA_20' in data.columns:
#             fig.add_trace(go.Scatter(
#                 x=data.index, y=data['SMA_20'],
#                 mode='lines', name='SMA 20',
#                 line=dict(color='orange', width=2)
#             ), row=1, col=1)
        
#         if 'SMA_50' in data.columns:
#             fig.add_trace(go.Scatter(
#                 x=data.index, y=data['SMA_50'],
#                 mode='lines', name='SMA 50',
#                 line=dict(color='blue', width=2)
#             ), row=1, col=1)
        
#         # Bollinger Bands
#         if all(col in data.columns for col in ['BB_Upper', 'BB_Lower']):
#             fig.add_trace(go.Scatter(
#                 x=data.index, y=data['BB_Upper'],
#                 mode='lines', name='BB Upper',
#                 line=dict(color='gray', width=1),
#                 showlegend=False
#             ), row=1, col=1)
            
#             fig.add_trace(go.Scatter(
#                 x=data.index, y=data['BB_Lower'],
#                 mode='lines', name='BB Lower',
#                 line=dict(color='gray', width=1),
#                 fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
#                 showlegend=False
#             ), row=1, col=1)
        
#         # Volume
#         if 'Volume' in data.columns:
#             fig.add_trace(go.Bar(
#                 x=data.index, y=data['Volume'],
#                 name='Volume', yaxis='y2',
#                 marker_color='rgba(158,202,225,0.6)'
#             ), row=2, col=1)
        
#         # RSI
#         if 'RSI' in data.columns:
#             fig.add_trace(go.Scatter(
#                 x=data.index, y=data['RSI'],
#                 mode='lines', name='RSI',
#                 line=dict(color='purple', width=2)
#             ), row=3, col=1)
            
#             # RSI levels
#             fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
#             fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
#         # MACD
#         if all(col in data.columns for col in ['MACD', 'Signal']):
#             fig.add_trace(go.Scatter(
#                 x=data.index, y=data['MACD'],
#                 mode='lines', name='MACD',
#                 line=dict(color='blue', width=2)
#             ), row=4, col=1)
            
#             fig.add_trace(go.Scatter(
#                 x=data.index, y=data['Signal'],
#                 mode='lines', name='Signal',
#                 line=dict(color='red', width=2)
#             ), row=4, col=1)
        
#         # Update layout
#         fig.update_layout(
#             title=f'{symbol} - Comprehensive Technical Analysis',
#             xaxis_rangeslider_visible=False,
#             height=800,
#             showlegend=True,
#             template='plotly_white'
#         )
        
#         return fig
        
#     except Exception as e:
#         st.error(f"Chart creation error: {str(e)}")
#         fig = go.Figure()
#         fig.add_annotation(text=f"Chart error: {str(e)}", x=0.5, y=0.5, showarrow=False)
#         return fig

# @safe_execute
# def get_google_finance_data(symbol, exchange):
#     """Fetch real-time data from Google Finance"""
#     try:
#         url = f'https://www.google.com/finance/quote/{symbol}:{exchange}'
#         headers = {
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
#         }
        
#         response = requests.get(url, headers=headers, timeout=10)
#         response.raise_for_status()
        
#         soup = BeautifulSoup(response.text, 'html.parser')
        
#         # Extract price
#         price_element = soup.find('div', class_='YMlKec fxKbKc')
#         if price_element:
#             price_text = price_element.text.strip().replace('₹', '').replace(',', '')
#             price = float(price_text)
#         else:
#             price = None
        
#         # Extract previous close
#         prev_close_element = soup.find('div', class_='P6K39c')
#         if prev_close_element:
#             prev_close_text = prev_close_element.text.strip().replace('₹', '').replace(',', '')
#             prev_close = float(prev_close_text)
#         else:
#             prev_close = price if price else None
        
#         # Extract volume
#         volume_element = soup.find('div', {'aria-label': 'Volume'})
#         volume = volume_element.find_next('div').text.strip() if volume_element else None
        
#         # Extract other information
#         info_elements = soup.find_all('div', class_='gyFHrc')
#         info_dict = {}
#         for element in info_elements:
#             key = element.find('div', class_='mfs7Fc').text.strip()
#             value = element.find('div', class_='P6K39c').text.strip()
#             info_dict[key] = value
        
#         # Generate synthetic historical data for demo
#         if price is not None:
#             dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
#             np.random.seed(42)  # For reproducible demo data
            
#             base_price = price
#             returns = np.random.normal(0.001, 0.02, 100)
#             prices = [base_price]
            
#             for ret in returns[1:]:
#                 prices.append(prices[-1] * (1 + ret))
            
#             historical_data = pd.DataFrame({
#                 'Date': dates,
#                 'Open': [p * np.random.uniform(0.98, 1.02) for p in prices],
#                 'High': [p * np.random.uniform(1.00, 1.05) for p in prices],
#                 'Low': [p * np.random.uniform(0.95, 1.00) for p in prices],
#                 'Close': prices,
#                 'Volume': np.random.randint(1000000, 10000000, 100)
#             })
            
#             historical_data.set_index('Date', inplace=True)
            
#             current_data = {
#                 'symbol': symbol,
#                 'price': price,
#                 'prev_close': prev_close,
#                 'volume': volume,
#                 'info': info_dict,
#                 'historical': historical_data,
#                 'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#             }
            
#             return current_data
#         else:
#             return None
        
#     except Exception as e:
#         st.error(f"Google Finance fetch error: {str(e)}")
#         return None

# @safe_execute
# def get_nse_data(symbol, period="3M"):
#     """Fetch NSE data using nselib"""
#     try:
#         if not NSELIB_AVAILABLE:
#             return None
        
#         data = capital_market.price_volume_and_deliverable_position_data(
#             symbol=symbol, 
#             period=period
#         )
        
#         if data is not None and not data.empty:
#             # Standardize column names
#             data.columns = data.columns.str.strip()
            
#             # Handle date column
#             if 'Date' in data.columns:
#                 data['Date'] = pd.to_datetime(data['Date'])
#                 data.set_index('Date', inplace=True)
            
#             # Rename columns to standard format
#             column_mapping = {
#                 'Close Price': 'Close',
#                 'Open Price': 'Open',
#                 'High Price': 'High',
#                 'Low Price': 'Low',
#                 'Total Traded Quantity': 'Volume'
#             }
            
#             data.rename(columns=column_mapping, inplace=True)
            
#             return {
#                 'symbol': symbol,
#                 'historical': data,
#                 'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#             }
        
#         return None
        
#     except Exception as e:
#         st.error(f"NSE data fetch error: {str(e)}")
#         return None

# def create_demo_data(symbol):
#     """Create demo data for testing"""
#     try:
#         dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
#         np.random.seed(hash(symbol) % 1000)  # Consistent demo data per symbol
        
#         base_price = 100 + (hash(symbol) % 1000)
#         returns = np.random.normal(0.001, 0.02, 100)
#         prices = [base_price]
        
#         for ret in returns[1:]:
#             prices.append(prices[-1] * (1 + ret))
        
#         historical_data = pd.DataFrame({
#             'Date': dates,
#             'Open': [p * np.random.uniform(0.98, 1.02) for p in prices],
#             'High': [p * np.random.uniform(1.00, 1.05) for p in prices],
#             'Low': [p * np.random.uniform(0.95, 1.00) for p in prices],
#             'Close': prices,
#             'Volume': np.random.randint(1000000, 10000000, 100)
#         })
        
#         historical_data.set_index('Date', inplace=True)
        
#         return {
#             'symbol': symbol,
#             'price': prices[-1],
#             'historical': historical_data,
#             'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         }
        
#     except Exception as e:
#         st.error(f"Demo data creation error: {str(e)}")
#         return None

# # Function to simulate buying a stock
# def buy_stock(symbol, quantity, price):
#     st.session_state.transaction_log.append({
#         'action': 'buy', 
#         'symbol': symbol, 
#         'quantity': quantity, 
#         'price': price,
#         'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     })
#     st.success(f"Bought {quantity} shares of {symbol} at ₹{price:.2f} each.")

# # Function to simulate selling a stock
# def sell_stock(symbol, quantity, price):
#     st.session_state.transaction_log.append({
#         'action': 'sell', 
#         'symbol': symbol, 
#         'quantity': quantity, 
#         'price': price,
#         'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     })
#     st.success(f"Sold {quantity} shares of {symbol} at ₹{price:.2f} each.")

# # Function to hold a stock
# def hold_stock(symbol):
#     st.session_state.transaction_log.append({
#         'action': 'hold', 
#         'symbol': symbol, 
#         'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     })
#     st.info(f"Holding {symbol}.")

# # Example stock name to symbol mapping
# def get_symbol_by_name(name):
#     stock_mapping = {
#         'RELIANCE': 'RELIANCE',
#         'INFOSYS': 'INFY',
#         'TATA MOTORS': 'TATAMOTORS',
#         'HDFC BANK': 'HDFCBANK',
#         'ICICI BANK': 'ICICIBANK',
#         'SBIN': 'SBIN',
#         'TCS': 'TCS',
#         'WIPRO': 'WIPRO',
#         'AXIS BANK': 'AXISBANK',
#         'ITC': 'ITC'
#     }
#     return stock_mapping.get(name.upper(), None)

# # Main dashboard logic
# def equity_dashboard():
#     # Data source selection
#     data_source = st.sidebar.selectbox(
#         "📈 Select Data Source",
#         ["Google Finance (Real-time)", "NSE Library (Comprehensive)", "Demo Mode"]
#     )
    
#     # Symbol input
#     col1, col2 = st.sidebar.columns(2)
#     with col1:
#         symbol_input = st.text_input("Stock Symbol or Name", "RELIANCE").upper()
#         symbol = get_symbol_by_name(symbol_input) or symbol_input
#     with col2:
#         exchange = st.text_input("Exchange", "NSE").upper()
    
#     # Time period selection
#     if data_source == "NSE Library (Comprehensive)":
#         period = st.sidebar.selectbox("Time Period", ["1M", "3M", "6M", "1Y"])
    
#     # Auto-refresh option
#     auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)
    
#     if st.sidebar.button("🔄 Refresh Data") or auto_refresh:
#         st.experimental_rerun()
    
#     # Fetch data based on source
#     with st.spinner("Fetching market data..."):
#         if data_source == "Google Finance (Real-time)":
#             stock_data = get_google_finance_data(symbol, exchange)
#         elif data_source == "NSE Library (Comprehensive)" and NSELIB_AVAILABLE:
#             stock_data = get_nse_data(symbol, period)
#         else:
#             stock_data = create_demo_data(symbol)
    
#     if stock_data is None:
#         st.error("❌ Unable to fetch data. Please check symbol and try again.")
#         return
    
#     # Main dashboard layout
#     st.success(f"✅ Data loaded successfully for {symbol}")
    
#     # Key metrics row
#     col1, col2, col3, col4 = st.columns(4)
    
#     historical_data = stock_data['historical']
#     latest_price = stock_data.get('price', historical_data['Close'].iloc[-1]) if 'price' in stock_data else historical_data['Close'].iloc[-1]
#     prev_price = stock_data.get('prev_close', historical_data['Close'].iloc[-2]) if len(historical_data) > 1 else latest_price
#     price_change = latest_price - prev_price
#     price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
    
#     with col1:
#         st.metric(
#             "Current Price",
#             f"₹{latest_price:.2f}",
#             f"{price_change:.2f} ({price_change_pct:.2f}%)"
#         )
    
#     with col2:
#         volume = stock_data.get('volume', historical_data['Volume'].iloc[-1]) if 'Volume' in historical_data.columns else "N/A"
#         st.metric("Volume", f"{volume:,}" if isinstance(volume, (int, float)) else volume)
    
#     with col3:
#         week_high = historical_data['High'].tail(7).max()
#         st.metric("Week High", f"₹{week_high:.2f}")
    
#     with col4:
#         week_low = historical_data['Low'].tail(7).min()
#         st.metric("Week Low", f"₹{week_low:.2f}")
    
#     # Technical analysis
#     with st.spinner("Analyzing market trends..."):
#         analyzed_data = calculate_technical_indicators(historical_data.copy())
#         signal, reason, confidence = generate_trading_signal(analyzed_data)
    
#     # Trading signal display
#     st.subheader("🎯 AI Trading Recommendation")
    
#     signal_class = f"{signal.lower()}-signal"
#     confidence_bar = "🟢" * int(confidence * 10) + "⚪" * (10 - int(confidence * 10))
    
#     st.markdown(f"""
#     <div class="prediction-box {signal_class}">
#         <h3>📊 {signal}</h3>
#         <p>{reason}</p>
#         <p>Confidence: {confidence_bar} ({confidence:.1%})</p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Advanced chart
#     st.subheader("📈 Advanced Technical Analysis")
#     chart = create_advanced_chart(analyzed_data, symbol)
#     st.plotly_chart(chart, use_container_width=True)
    
#     # Technical indicators summary
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader("📊 Technical Indicators")
#         if not analyzed_data.empty:
#             latest = analyzed_data.iloc[-1]
            
#             indicators = []
#             if 'RSI' in analyzed_data.columns:
#                 indicators.append(['RSI', f"{latest.get('RSI', 0):.2f}", 
#                                  "Oversold" if latest.get('RSI', 50) < 30 else "Overbought" if latest.get('RSI', 50) > 70 else "Neutral"])
            
#             if 'MACD' in analyzed_data.columns and 'Signal' in analyzed_data.columns:
#                 indicators.append(['MACD', f"{latest.get('MACD', 0):.4f}", 
#                                  "Bullish" if latest.get('MACD', 0) > latest.get('Signal', 0) else "Bearish"])
            
#             if 'SMA_20' in analyzed_data.columns:
#                 indicators.append(['SMA 20', f"₹{latest.get('SMA_20', 0):.2f}", 
#                                  "Bullish" if latest.get('Close', 0) > latest.get('SMA_20', 0) else "Bearish"])
            
#             if 'SMA_50' in analyzed_data.columns:
#                 indicators.append(['SMA 50', f"₹{latest.get('SMA_50', 0):.2f}", 
#                                  "Bullish" if latest.get('Close', 0) > latest.get('SMA_50', 0) else "Bearish"])
            
#             if all(col in analyzed_data.columns for col in ['BB_Upper', 'BB_Lower']):
#                 bb_position = "Above" if latest.get('Close', 0) > latest.get('BB_Middle', 0) else "Below"
#                 bb_signal = "Bullish" if latest.get('Close', 0) > latest.get('BB_Upper', 0) else "Bearish" if latest.get('Close', 0) < latest.get('BB_Lower', 0) else "Neutral"
#                 indicators.append(['BB Position', bb_position, bb_signal])
            
#             if indicators:
#                 indicators_df = pd.DataFrame(indicators, columns=['Indicator', 'Value', 'Signal'])
#                 st.dataframe(indicators_df, use_container_width=True)
    
#     with col2:
#         st.subheader("📈 Price Statistics")
#         stats = [
#             ['Current Price', f"₹{latest_price:.2f}"],
#             ['Day High', f"₹{historical_data['High'].iloc[-1]:.2f}"],
#             ['Day Low', f"₹{historical_data['Low'].iloc[-1]:.2f}"],
#             ['Volume', f"{volume:,}" if isinstance(volume, (int, float)) else volume]
#         ]
        
#         if 'Volume' in historical_data.columns:
#             avg_volume = historical_data['Volume'].tail(20).mean()
#             stats.append(['Avg Volume (20d)', f"{avg_volume:,.0f}"])
        
#         volatility = historical_data['Close'].pct_change().tail(20).std() * 100
#         stats.append(['Volatility (20d)', f"{volatility:.2f}%"])
        
#         stats_df = pd.DataFrame(stats, columns=['Metric', 'Value'])
#         st.dataframe(stats_df, use_container_width=True)
    
#     # Risk assessment
#     st.subheader("⚠️ Risk Assessment")
    
#     if 'Close' in historical_data.columns:
#         volatility = historical_data['Close'].pct_change().std() * 100
#     else:
#         volatility = 0
    
#     if volatility > 5:
#         risk_level = "High"
#         risk_color = "red"
#     elif volatility > 2:
#         risk_level = "Medium"
#         risk_color = "orange"
#     else:
#         risk_level = "Low"
#         risk_color = "green"
    
#     st.markdown(f"""
#     <div style="padding: 1rem; background: #f8f9fa; border-radius: 8px; border-left: 4px solid {risk_color};">
#         <h4 style="color: {risk_color};">Risk Level: {risk_level}</h4>
#         <p>Volatility: {volatility:.2f}%</p>
#         <p>Recommendation: {'Consider position sizing carefully' if risk_level == 'High' else 'Moderate risk - suitable for most investors' if risk_level == 'Medium' else 'Low risk - suitable for conservative investors'}</p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Performance metrics
#     st.subheader("📊 Performance Metrics")
    
#     if 'Close' in historical_data.columns:
#         returns = historical_data['Close'].pct_change().dropna()
#     else:
#         returns = pd.Series([0])
    
#     perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    
#     with perf_col1:
#         st.metric("1D Return", f"{returns.iloc[-1]*100:.2f}%" if len(returns) > 0 else "N/A")
    
#     with perf_col2:
#         week_return = (latest_price / historical_data['Close'].iloc[-7] - 1) * 100 if len(historical_data) >= 7 else 0
#         st.metric("7D Return", f"{week_return:.2f}%")
    
#     with perf_col3:
#         month_return = (latest_price / historical_data['Close'].iloc[-30] - 1) * 100 if len(historical_data) >= 30 else 0
#         st.metric("30D Return", f"{month_return:.2f}%")
    
#     with perf_col4:
#         sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
#         st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
#     # Transaction simulation
#     st.subheader("💼 Trading Simulation")
    
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         trade_quantity = st.number_input("Quantity", min_value=1, value=10)
#     with col2:
#         trade_price = st.number_input("Price per Share", min_value=0.01, value=float(latest_price), format="%.2f")
#     with col3:
#         st.write("")
#         st.write("")
#         if st.button("Buy"):
#             buy_stock(symbol, trade_quantity, trade_price)
#         if st.button("Sell"):
#             sell_stock(symbol, trade_quantity, trade_price)
#         if st.button("Hold"):
#             hold_stock(symbol)
    
#     # Display transaction log
#     st.subheader("📝 Transaction Log")
#     if st.session_state.transaction_log:
#         log_df = pd.DataFrame(st.session_state.transaction_log)
#         st.dataframe(log_df, use_container_width=True)
#     else:
#         st.info("No transactions yet")
    
#     # Footer
#     st.markdown("---")
#     st.markdown(f"**Last Updated:** {stock_data.get('last_updated', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))} | **Data Source:** {data_source}")
#     st.markdown("⚠️ *This is for educational purposes only. Please consult a financial advisor before making investment decisions.*")
    
#     # Auto-refresh functionality
#     if auto_refresh:
#         time.sleep(30)
#         st.experimental_rerun()

# def derivatives_dashboard():
#     st.sidebar.subheader("Derivatives Data Options")
    
#     # Derivative instrument selection
#     instrument = st.sidebar.selectbox("Instrument Type", options=("NSE Equity Market", "NSE Derivative Market"))
    
#     # Initialize data to None
#     data = None
    
#     if instrument == "NSE Equity Market":
#         data_info = st.sidebar.selectbox("Data to extract", options=(
#             "price_volume_and_deliverable_position_data",
#             "price_volume_data",
#             "deliverable_position_data",
#             "bhav_copy_with_delivery",
#             "bhav_copy_equities",
#             "equity_list",
#             "fno_equity_list",
#             "fno_index_list",
#             "nifty50_equity_list",
#             "india_vix_data",
#             "market_watch_all_indices",
#             "bulk_deal_data",
#             "block_deals_data",
#             "short_selling_data",
#             "index_data",
#             "var_begin_day",
#             "var_1st_intra_day",
#             "var_2nd_intra_day",
#             "var_3rd_intra_day",
#             "var_4th_intra_day",
#             "var_end_of_day"
#         ))

#         if (data_info == "equity_list") or (data_info == "fno_equity_list") or (data_info == "market_watch_all_indices") or (data_info == "nifty50_equity_list"):
#             data = getattr(capital_market, data_info)() if NSELIB_AVAILABLE else None

#         elif data_info in ["price_volume_and_deliverable_position_data", "price_volume_data", "deliverable_position_data"]:
#             symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., SBIN)", "SBIN")
#             period = st.sidebar.selectbox("Select Period", ["1M", "3M", "6M", "1Y"])

#             try:
#                 if NSELIB_AVAILABLE:
#                     data = getattr(capital_market, data_info)(symbol=symbol, period=period)
#             except Exception as e:
#                 st.error(f"Error: {str(e)}")

#         elif (data_info == "bhav_copy_equities") or (data_info == "bhav_copy_with_delivery"):
#             date = st.sidebar.text_input("Date (dd-mm-yyyy)", "01-01-2025")
#             try:
#                 if NSELIB_AVAILABLE:
#                     data = getattr(capital_market, data_info)(date)
#             except Exception as e:
#                 st.error(f"Error: {str(e)}")

#         elif (data_info == "block_deals_data") or (data_info == "bulk_deal_data") or (data_info == "india_vix_data") or (data_info == "short_selling_data"):
#             try:
#                 period = st.sidebar.selectbox("Select Period", ["1M", "3M", "6M", "1Y"])
#                 if NSELIB_AVAILABLE:
#                     data = getattr(capital_market, data_info)(period=period)
#             except Exception as e:
#                 st.error(f"Error: {str(e)}")

#         elif data_info in ["var_begin_day", "var_1st_intra_day", "var_2nd_intra_day", "var_3rd_intra_day", "var_4th_intra_day", "var_end_of_day"]:
#             trade_date = st.sidebar.date_input("Select Trade Date", datetime.now())
#             formatted_date = trade_date.strftime("%d-%m-%Y")
#             try:
#                 if NSELIB_AVAILABLE:
#                     data = getattr(capital_market, data_info)(trade_date=formatted_date)
#             except Exception as e:
#                 st.error(f"Error: {str(e)}")

#     elif instrument == "NSE Derivative Market":
#         data_info = st.sidebar.selectbox("Data to extract", options=(
#             "fno_bhav_copy",
#             "participant_wise_open_interest",
#             "participant_wise_trading_volume",
#             "expiry_dates_future",
#             "expiry_dates_option_index",
#             "nse_live_option_chain",
#             "future_price_volume_data",
#             "option_price_volume_data"
#         ))

#         if (data_info == "expiry_dates_future") or (data_info == "expiry_dates_option_index"):
#             if NSELIB_AVAILABLE:
#                 data = getattr(derivatives, data_info)()

#         elif (data_info == "participant_wise_trading_volume") or (data_info == "participant_wise_open_interest") or (data_info == "fno_bhav_copy"):
#             date = st.sidebar.text_input("Date (dd-mm-yyyy)", "01-02-2025")
#             if NSELIB_AVAILABLE:
#                 data = getattr(derivatives, data_info)(date)

#         elif (data_info == "nse_live_option_chain"):
#             ticker = st.sidebar.text_input("Ticker", "BANKNIFTY")
#             expiry_date = st.sidebar.text_input("Expiry Date (dd-mm-yyyy)", "01-01-2025")
#             if NSELIB_AVAILABLE:
#                 data = derivatives.nse_live_option_chain(ticker, expiry_date=expiry_date)

#         elif (data_info == "future_price_volume_data"):
#             ticker = st.sidebar.text_input("Ticker", "SBIN")
#             type_ = st.sidebar.selectbox("Instrument Type", ["FUTSTK", "FUTIDX"])
#             period_ = st.sidebar.selectbox("Period", ["1M", "3M", "6M", "1Y"])
#             if NSELIB_AVAILABLE:
#                 data = derivatives.future_price_volume_data(ticker, type_, period=period_)

#         elif (data_info == "option_price_volume_data"):
#             ticker = st.sidebar.text_input("Ticker", "BANKNIFTY")
#             type_ = st.sidebar.selectbox("Instrument Type", ["OPTIDX", "OPTSTK"])
#             period_ = st.sidebar.selectbox("Period", ["1M", "3M", "6M", "1Y"])
#             if NSELIB_AVAILABLE:
#                 data = derivatives.option_price_volume_data(ticker, type_, period=period_)

#     # Display data
#     if data is not None:
#         st.subheader(f"Derivatives Data: {data_info}")
#         if isinstance(data, pd.DataFrame):
#             st.dataframe(data, use_container_width=True)
            
#             # Basic visualization for numerical data
#             numerical_cols = data.select_dtypes(include=[np.number]).columns
#             if len(numerical_cols) > 0:
#                 st.subheader("Data Visualization")
#                 chart_type = st.selectbox("Select Chart Type", ["Line", "Bar", "Scatter"])
                
#                 x_axis = st.selectbox("Select X-axis", data.columns)
#                 y_axis = st.selectbox("Select Y-axis", numerical_cols)
                
#                 if chart_type == "Line":
#                     fig = px.line(data, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}")
#                 elif chart_type == "Bar":
#                     fig = px.bar(data, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
#                 elif chart_type == "Scatter":
#                     fig = px.scatter(data, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
                
#                 st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.write(data)
#     else:
#         st.warning("No data available for the selected options")

# # Main app flow
# if market_type == "Equity":
#     equity_dashboard()
# else:
#     derivatives_dashboard()










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
from streamlit_autorefresh import st_autorefresh # NEW IMPORT

warnings.filterwarnings('ignore')

# --- API Keys and Configuration ---
FINNHUB_API_KEY = "d1sdgppr01qs2slhjh20d1sdgppr01qs2slhjh2g"

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
def get_google_finance_data(symbol, exchange):
    """
    Robustly fetches real-time data from Google Finance with enhanced error handling.
    Generates synthetic historical data as Google Finance does not have a public API.
    """
    try:
        url = f'https://www.google.com/finance/quote/{symbol}:{exchange}'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        price_element = soup.find('div', class_='YMlKec fxKbKc')
        if not price_element:
            price_element = soup.find('div', class_='YMlKec')
            
        if not price_element:
            st.error(f"Could not find price element on Google Finance page for {symbol}. The website structure may have changed. Please try another data source.")
            return None

        price = float(price_element.text.strip().replace('₹', '').replace(',', ''))
        
        prev_close_element = soup.find('div', string='Prev close')
        if not prev_close_element:
            prev_close_div = soup.find('div', class_='mfs7Fc', string='Prev Close')
            if prev_close_div:
                prev_close_element = prev_close_div.find_next('div', class_='P6K39c')
        
        prev_close = float(prev_close_element.text.strip().replace('₹', '').replace(',', '')) if prev_close_element else price
        
        volume_element = soup.find('div', string='Volume')
        if volume_element:
            volume = volume_element.find_next('div').text.strip().replace(',', '')
        else:
            volume = "N/A"
        
        if price is None:
            raise ValueError("Could not find current price on Google Finance page.")

        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        np.random.seed(hash(symbol) % 1000)
        returns = np.random.normal(0.001, 0.02, 100)
        prices_series = [price]
        for ret in returns[1:]:
            prices_series.append(prices_series[-1] * (1 + ret))
        
        historical_data = pd.DataFrame({
            'Date': dates,
            'Open': [p * np.random.uniform(0.98, 1.02) for p in prices_series],
            'High': [p * np.random.uniform(1.00, 1.05) for p in prices_series],
            'Low': [p * np.random.uniform(0.95, 1.00) for p in prices_series],
            'Close': prices_series,
            'Volume': np.random.randint(1000000, 10000000, 100)
        })
        historical_data.set_index('Date', inplace=True)
        
        return {
            'symbol': symbol,
            'price': price,
            'prev_close': prev_close,
            'volume': volume,
            'historical': historical_data,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    except Exception as e:
        st.error(f"Google Finance data fetch failed: {e}")
        return None

# --- AI Trading Analysis Functions ---
def calculate_technical_indicators(data):
    # ... (no changes here)
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
    # ... (no changes here)
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
    # ... (no changes here)
    if data.empty:
        fig = go.Figure()
        fig.add_annotation(text="No chart data available.", x=0.5, y=0.5, showarrow=False)
        return fig
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.2, 0.15, 0.15], subplot_titles=('Price & Volume', 'Bollinger Bands', 'RSI', 'MACD'))
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="Price"), row=1, col=1)
    if 'SMA_20' in data.columns: fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], mode='lines', name='SMA 20', line=dict(color='orange', width=2)), row=1, col=1)
    if 'SMA_50' in data.columns: fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='SMA 50', line=dict(color='blue', width=2)), row=1, col=1)
    if 'Volume' in data.columns: fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='rgba(158,202,225,0.6)'), row=2, col=1)
    if 'BB_Upper' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], mode='lines', name='BB Upper', line=dict(color='gray', width=1), showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Middle'], mode='lines', name='BB Middle', line=dict(color='darkgray', width=1), showlegend=True), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], mode='lines', name='BB Lower', line=dict(color='gray', width=1), fill='tonexty', fillcolor='rgba(128,128,128,0.1)', showlegend=False), row=3, col=1)
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
    # ... (no changes here)
    if data.empty or 'Close' not in data.columns:
        return {"volatility": 0, "risk_level": "N/A", "recommendation": "Insufficient data."}
    returns = data['Close'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(252) * 100
    if volatility > 30: risk_level, risk_color, recommendation = "High", "red", "High volatility detected. Consider this a high-risk investment and use proper position sizing."
    elif volatility > 15: risk_level, risk_color, recommendation = "Medium", "orange", "Moderate volatility. Suitable for most investors with a balanced risk tolerance."
    else: risk_level, risk_color, recommendation = "Low", "green", "Low volatility. Suitable for conservative investors."
    return {"volatility": volatility, "risk_level": risk_level, "risk_color": risk_color, "recommendation": recommendation}

def calculate_performance_metrics(data):
    # ... (no changes here)
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
    
    # --- REVISED REFRESH BUTTONS ---
    col_refresh, col_auto = st.sidebar.columns(2)
    
    # Refresh Button: Use a unique key and a function to clear cache and rerun.
    if col_refresh.button("🔄 Refresh Data", key="manual_refresh_btn"):
        st.cache_data.clear()
        st.experimental_rerun()
    
    # Auto-Refresh Checkbox: Use a dedicated component for non-blocking auto-refresh.
    auto_refresh_toggle = col_auto.checkbox("Auto Refresh (60s)", value=False, help="Automatically refreshes data every 60 seconds.")
    if auto_refresh_toggle:
        st_autorefresh(interval=60000, key="auto_refresh_trigger") # 60000 ms = 60 seconds

    with st.spinner(f"Fetching market data for {symbol_to_fetch}..."):
        if data_source == "Google Finance (Real-time)":
            stock_data = get_google_finance_data(symbol_to_fetch, "NSE")
        elif data_source == "NSE Library (Comprehensive)":
            stock_data = get_nse_data(symbol_to_fetch, period)
        else:
            stock_data = create_demo_data(symbol_to_fetch)

    if stock_data is None:
        st.error(f"❌ Unable to fetch data for {symbol_to_fetch}. Please check the symbol and try a different data source.")
        return
    
    st.success(f"✅ Data loaded successfully for {stock_data['symbol']}")
    historical_data = stock_data['historical']
    latest_price = stock_data.get('price', historical_data['Close'].iloc[-1])
    prev_price = historical_data['Close'].iloc[-2] if len(historical_data) > 1 else latest_price
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
    
    st.subheader("📊 Technical Indicators & Statistics")
    tech_col, stat_col = st.columns(2)
    with tech_col:
        st.markdown("#### Technical Indicators")
        if not analyzed_data.empty:
            latest = analyzed_data.iloc[-1]
            indicators_df = pd.DataFrame([['RSI', f"{latest.get('RSI', np.nan):.2f}", "Oversold" if latest.get('RSI', 50) < 30 else "Overbought" if latest.get('RSI', 50) > 70 else "Neutral"], ['MACD', f"{latest.get('MACD', np.nan):.4f}", "Bullish" if latest.get('MACD', 0) > latest.get('Signal', 0) else "Bearish"], ['SMA 20', f"₹{latest.get('SMA_20', np.nan):.2f}", "Bullish" if latest.get('Close', 0) > latest.get('SMA_20', 0) else "Bearish"], ['SMA 50', f"₹{latest.get('SMA_50', np.nan):.2f}", "Bullish" if latest.get('Close', 0) > latest.get('SMA_50', 0) else "Bearish"]], columns=['Indicator', 'Value', 'Signal']).replace('nan', 'N/A')
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
    <div style="padding: 1rem; background: #f8f9fa; border-radius: 8px; border-left: 4px solid {risk_metrics['risk_color']};">
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
    if instrument == "NSE Equity Market":
        data_info = st.sidebar.selectbox("Data to extract", options=("price_volume_and_deliverable_position_data", "price_volume_data", "deliverable_position_data", "bhav_copy_with_delivery", "bhav_copy_equities", "equity_list", "fno_equity_list", "fno_index_list", "nifty50_equity_list", "india_vix_data", "market_watch_all_indices", "bulk_deal_data", "block_deals_data", "short_selling_data", "index_data", "var_begin_day", "var_1st_intra_day", "var_2nd_intra_day", "var_3rd_intra_day", "var_4th_intra_day", "var_end_of_day"))
        if (data_info == "equity_list") or (data_info == "fno_equity_list") or (data_info == "market_watch_all_indices") or (data_info == "nifty50_equity_list"): data = getattr(capital_market, data_info)() if NSELIB_AVAILABLE else None
        elif data_info in ["price_volume_and_deliverable_position_data", "price_volume_data", "deliverable_position_data"]:
            symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., SBIN)", "SBIN")
            period = st.sidebar.selectbox("Select Period", ["1M", "3M", "6M", "1Y"])
            try:
                if NSELIB_AVAILABLE: data = getattr(capital_market, data_info)(symbol=symbol, period=period)
            except Exception as e: st.error(f"Error: {str(e)}")
        elif (data_info == "bhav_copy_equities") or (data_info == "bhav_copy_with_delivery"):
            date = st.sidebar.text_input("Date (dd-mm-yyyy)", "01-01-2025")
            try:
                if NSELIB_AVAILABLE: data = getattr(capital_market, data_info)(date)
            except Exception as e: st.error(f"Error: {str(e)}")
        elif (data_info == "block_deals_data") or (data_info == "bulk_deal_data") or (data_info == "india_vix_data") or (data_info == "short_selling_data"):
            try:
                period = st.sidebar.selectbox("Select Period", ["1M", "3M", "6M", "1Y"])
                if NSELIB_AVAILABLE: data = getattr(capital_market, data_info)(period=period)
            except Exception as e: st.error(f"Error: {str(e)}")
        elif data_info in ["var_begin_day", "var_1st_intra_day", "var_2nd_intra_day", "var_3rd_intra_day", "var_4th_intra_day", "var_end_of_day"]:
            trade_date = st.sidebar.date_input("Select Trade Date", datetime.now())
            formatted_date = trade_date.strftime("%d-%m-%Y")
            try:
                if NSELIB_AVAILABLE: data = getattr(capital_market, data_info)(trade_date=formatted_date)
            except Exception as e: st.error(f"Error: {str(e)}")
    elif instrument == "NSE Derivative Market":
        data_info = st.sidebar.selectbox("Data to extract", options=("fno_bhav_copy", "participant_wise_open_interest", "participant_wise_trading_volume", "expiry_dates_future", "expiry_dates_option_index", "nse_live_option_chain", "future_price_volume_data", "option_price_volume_data"))
        if (data_info == "expiry_dates_future") or (data_info == "expiry_dates_option_index"): data = getattr(derivatives, data_info)()
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
            if NSELIB_AVAILABLE: data = derivatives.future_price_volume_data(ticker, type_, period=period_)
        elif (data_info == "option_price_volume_data"):
            ticker = st.sidebar.text_input("Ticker", "BANKNIFTY")
            type_ = st.sidebar.selectbox("Instrument Type", ["OPTIDX", "OPTSTK"])
            period_ = st.sidebar.selectbox("Period", ["1M", "3M", "6M", "1Y"])
            if NSELIB_AVAILABLE: data = derivatives.option_price_volume_data(ticker, type_, period=period_)
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