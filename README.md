# Crypto Trading Bot

A sophisticated cryptocurrency trading bot leveraging artificial intelligence to predict optimal buy and sell opportunities. This system integrates machine learning, market analysis, and automation to deliver actionable trading signals directly to your Telegram account.

---

## Overview

### Core Features

* **Multi-Timeframe Analysis**: Evaluates 15m, 30m, 1h, and 4h charts for comprehensive market insight.
* **AI-Powered Predictions**: Employs advanced machine learning models to identify high-probability trading setups.
* **Risk Management**: Implements position sizing, stop-loss, and take-profit mechanisms to manage exposure effectively.
* **Telegram Integration**: Sends real-time trading signals to your Telegram account.
* **Backtesting**: Simulates past trading performance to evaluate strategy viability.
* **Live Monitoring**: Continuously scans markets for opportunities while you are away.

---

## Requirements

To use the bot, ensure the following are installed or available:

* Python 3.10 or later
* Telegram Bot Token (generated via [@BotFather](https://t.me/BotFather))
* Basic understanding of trading principles

---

## Installation

1. **Clone and Set Up the Environment:**

   ```bash
   git clone <repository-url>
   cd signal_bot
   python -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure the Bot:**

   ```bash
   cp config.example.yaml config.yaml
   nano config.yaml
   ```

---

## Configuration

Edit `config.yaml` to define:

* **Trading Parameters**: Cryptocurrency pairs, timeframes, etc.
* **Model Parameters**: AI configuration and performance settings.
* **Risk Controls**: Stop-loss and take-profit rules.
* **Telegram Settings**: Token and chat ID.
* **Logging Options**: Output and log file preferences.

> **Important:** Never commit `config.yaml` containing real API keys or sensitive credentials.

---

## Usage

### Training the Models

```bash
python cli.py train --config config.yaml --interval 1h
```

### Backtesting

```bash
python cli.py backtest --config config.yaml --interval 1h --export backtest_results.csv
```

### Market Monitoring

```bash
python cli.py monitor --config config.yaml
```

### Launching the Trading Bot

```bash
python bot.py --config config.yaml
```

---

## Useful Commands

### Train Multiple Timeframes

```bash
for interval in 15m 30m 1h 4h; do
    python cli.py train --config config.yaml --interval $interval
done
```

### Advanced Backtesting

```bash
python cli.py backtest \
    --config config.yaml \
    --interval 1h \
    --start 2023-01-01 \
    --end 2024-01-01 \
    --export my_trades.csv
```

### Signal Filtering

```bash
python cli.py monitor --config config.yaml --threshold 0.7
```

---

## Testing

To run all tests:

```bash
python -m pytest tests/
```

To run a specific test:

```bash
python -m pytest tests/test_signals.py::test_compute_stop_take_prices
```

---

## Project Structure

```
signal_bot/
├── bot.py              # Main trading logic
├── cli.py              # Command line interface
├── monitor.py          # Market monitoring functions
├── signal_cli.py       # Signal handling
├── signals.py          # Technical calculations
├── train.py            # AI model training
├── backtest.py         # Backtesting engine
├── utils.py            # Utility functions
├── config.yaml         # User configuration file
├── requirements.txt    # Dependencies
├── models/             # Trained AI models
├── state/              # Persistent bot state
├── logs/               # Log files
└── tests/              # Test suite
```

---

## Advanced Configuration

### Supported AI Models

* **LightGBM** (default): High performance and efficiency.
* **Random Forest**: Ensemble model for robust predictions.
* **Logistic Regression**: Lightweight and interpretable option.

### Risk Management

* Dynamic stop-loss and take-profit levels.
* Position sizing based on AI confidence scores.
* Trade cooldowns to prevent overtrading.

### Technical Indicators

* RSI, MACD, Bollinger Bands, Moving Averages, and more.

---

## Performance Tracking

Comprehensive logging includes:

* Signal accuracy and performance metrics.
* Trade history and profitability reports.
* Real-time profit/loss monitoring.

---

## Contributing

1. Fork the repository.
2. Implement and test your improvements.
3. Submit a pull request for review.

---

## Disclaimer

This software is provided for educational and research purposes only. Cryptocurrency trading carries significant financial risk. Use the bot at your own discretion and never invest more than you can afford to lose. The developers assume no responsibility for financial losses or damages incurred through the use of this tool.
