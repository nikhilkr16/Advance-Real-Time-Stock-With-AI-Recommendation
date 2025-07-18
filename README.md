# Advanced Real-Time Stock Dashboard with AI Recommendations

A comprehensive Streamlit-based stock market dashboard featuring real-time data analysis and AI-powered trading recommendations with enhanced security and accuracy.

## 🚀 Key Features

### ✅ Enhanced Security
- **Secure API Key Management**: Finnhub API key now retrieved from environment variables instead of hardcoded values
- **Input Validation**: Comprehensive stock symbol validation with proper error handling
- **Data Source Verification**: Optional symbol availability checking before processing

### 🤖 Advanced AI Trading Recommendations
- **Mathematical Model**: Enhanced algorithm using weighted scoring of multiple technical indicators
- **80%+ Accuracy Target**: Sophisticated signal processing with confidence calculations
- **Multiple Indicators**: RSI, MACD, SMA, Bollinger Bands, Stochastic Oscillator, Williams %R, and momentum analysis
- **Risk Assessment**: Comprehensive volatility analysis and risk-level recommendations

### 📊 Technical Analysis
- **Real-time Data**: Google Finance scraping, NSELib integration, and Finnhub API support
- **Advanced Charts**: Multi-subplot candlestick charts with technical overlays
- **Performance Metrics**: Returns analysis, Sharpe ratio, and risk metrics
- **Volume Analysis**: On-Balance Volume (OBV) and volume rate of change

### 🛡️ Error Handling
- **Symbol Validation**: Prevents processing of invalid ticker symbols
- **Network Resilience**: Graceful handling of API failures with fallback options
- **Data Integrity**: Comprehensive checks for data quality and completeness

## 🔧 Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up API Key (Optional but Recommended)
Set your Finnhub API key as an environment variable for enhanced functionality:

```bash
# Linux/Mac
export FINNHUB_API_KEY="your_api_key_here"

# Windows
set FINNHUB_API_KEY=your_api_key_here
```

**Note**: The dashboard works without the API key but with limited symbol suggestions.

### 3. Run the Application
```bash
streamlit run AdvanceStockAnalysis.py
```

## 📋 Usage Guide

### Stock Analysis
1. **Select Data Source**: Choose between Google Finance (real-time), NSE Library, or Demo Mode
2. **Enter Symbol**: Input a valid Indian stock symbol (e.g., TCS, INFY, RELIANCE)
3. **Optional Verification**: Enable symbol availability checking for validation
4. **View Analysis**: Get AI-powered recommendations with confidence scores

### Understanding AI Recommendations

The enhanced AI model analyzes multiple factors:

- **RSI (20% weight)**: Relative Strength Index for momentum
- **MACD (25% weight)**: Moving Average Convergence Divergence
- **Moving Averages (20% weight)**: SMA 20 and SMA 50 trend analysis
- **Bollinger Bands (15% weight)**: Volatility and price bounds
- **Stochastic Oscillator (10% weight)**: Momentum indicator
- **Williams %R (10% weight)**: Momentum oscillator

### Signal Interpretation

- **BUY**: Strong bullish consensus (Score ≥ 1.5) or moderate bullish signals (0.5-1.5)
- **SELL**: Strong bearish consensus (Score ≤ -1.5) or moderate bearish signals (-1.5 to -0.5)
- **HOLD**: Mixed signals or insufficient data for clear direction

Confidence levels indicate signal strength:
- **90%+**: Very high confidence
- **70-89%**: High confidence
- **50-69%**: Moderate confidence
- **<50%**: Low confidence

## 🧪 Testing

Run the comprehensive test suite to validate all functionality:

```bash
python test_ai_model.py
```

Tests cover:
- Symbol validation
- Technical indicator calculations
- AI signal generation
- Data integrity

## 📁 File Structure

- `AdvanceStockAnalysis.py`: Main dashboard application
- `requirements.txt`: Python dependencies
- `packages.txt`: System package dependencies
- `test_ai_model.py`: Comprehensive test suite
- `README.md`: This documentation

## 🔒 Security Features

### API Key Protection
- Environment variable usage prevents accidental exposure
- Graceful degradation when API key is unavailable
- Clear user guidance for secure setup

### Input Validation
- Symbol format validation (1-10 characters, alphanumeric)
- Rejection of purely numeric symbols
- Invalid character detection
- Empty input handling

### Error Handling
- Network timeout protection
- JSON parsing error handling
- Data structure validation
- Graceful fallback to demo data

## 📈 AI Model Accuracy

The enhanced AI model targets 80%+ accuracy through:

1. **Weighted Scoring**: Each indicator contributes based on its historical effectiveness
2. **Consensus Analysis**: Multiple indicators must agree for strong signals
3. **Momentum Validation**: Price momentum confirms or questions indicator signals
4. **Confidence Calibration**: Conservative confidence scoring reduces false positives

### Model Performance Features
- **Strict Thresholds**: Higher score requirements for BUY/SELL signals
- **Momentum Support**: ROC validation for additional confirmation
- **Risk-Adjusted Confidence**: Lower confidence for moderate signals
- **Comprehensive Analysis**: 6+ technical indicators with volume analysis

## ⚠️ Disclaimer

This application is for educational purposes only. Always consult with a qualified financial advisor before making investment decisions. The AI recommendations are based on technical analysis and should not be considered as financial advice.

## 🤝 Contributing

To contribute to this project:
1. Ensure all tests pass: `python test_ai_model.py`
2. Follow the existing code style and documentation patterns
3. Add tests for new functionality
4. Update this README for significant changes

## 📞 Support

For issues or questions:
1. Check that all dependencies are correctly installed
2. Verify API key setup if using Finnhub features
3. Run the test suite to identify any configuration issues
4. Review error messages for specific guidance