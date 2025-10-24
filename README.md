# 🚀 Crypto Trading Bot - Make Money While You Sleep! 💸

Yo! This is a sick crypto trading bot that uses AI to predict when to buy and sell crypto. It's basically like having a mini Warren Buffett working for you 24/7, but way cooler and it sends signals to your Telegram.

## 🎯 What This Bad Boy Can Do

- **Multi-Timeframe Analysis**: Checks 15m, 30m, 1h, and 4h charts (like a pro trader)
- **AI Magic**: Uses machine learning models that are smarter than your average finance bro
- **Risk Management**: Won't let you blow up your account (hopefully)
- **Telegram Integration**: Get signals on your phone like a boss
- **Backtesting**: See if your strategy would've made you rich in the past
- **Live Monitoring**: Scans the markets while you're gaming or sleeping

## 📋 What You Need

- Python 3.10+ (if you don't have this, google it lol)
- Binance API keys (get them from binance.com)
- Telegram Bot Token (talk to @BotFather on Telegram)
- Basic brain cells (optional but recommended)

## 🛠️ How to Set This Shit Up

1. **Clone this repo and setup**
```bash
git clone <repository-url>
cd signal_bot
python -m venv venv
source venv/bin/activate  # Windows users: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure the bot**
```bash
# Copy the example config and edit it with your stuff
nano config.yaml
```

## ⚙️ Configuration Stuff

Edit `config.yaml` to setup:
- **Trading Settings**: Which crypto, timeframes, etc.
- **Model Stuff**: AI parameters (if you're a nerd)
- **Risk Management**: Stop-loss, take-profit (so you don't lose everything)
- **Telegram**: Your bot token and chat ID
- **Logging**: Where to save the logs

> 🔥 **IMPORTANT**: Don't commit your `config.yaml` with real API keys unless you want people to steal your money (not a good look).

## 🎯 Let's Make Some Money!

1. **Train the AI models**
```bash
python cli.py train --config config.yaml --interval 1h
```

2. **Test if this thing actually works**
```bash
python cli.py backtest --config config.yaml --interval 1h --export backtest_results.csv
```

3. **Start monitoring the markets**
```bash
python cli.py monitor --once --config config.yaml  # Quick check
python cli.py monitor --config config.yaml         # Keep it running
```

4. **Launch the trading bot**
```bash
python bot.py --config config.yaml
```

## 📊 Cool Commands You Can Use

### Train All Timeframes
```bash
for interval in 15m 30m 1h 4h; do
    python cli.py train --config config.yaml --interval $interval
done
```

### Backtest Like a Pro
```bash
python cli.py backtest \
    --config config.yaml \
    --interval 1h \
    --start 2023-01-01 \
    --end 2024-01-01 \
    --export my_trades.csv
```

### Get High-Quality Signals Only
```bash
python cli.py monitor --config config.yaml --threshold 0.7
```

## 🧪 Testing (for the nerds)

```bash
python -m pytest tests/
```

Run specific tests:
```bash
python -m pytest tests/test_signals.py::test_compute_stop_take_prices
```

## 📁 What's in This Folder?

```
signal_bot/
├── bot.py              # The main bot that does the trading
├── cli.py              # Command line stuff
├── monitor.py          # Watches the markets
├── signal_cli.py       # Signal processing
├── signals.py          # Math stuff for signals
├── train.py            # Trains the AI models
├── backtest.py         # Tests strategies
├── utils.py            # Helper functions
├── config.yaml         # YOUR CONFIG (don't share this!)
├── requirements.txt    # Python packages
├── models/             # AI brain files (.pkl)
├── state/              # Bot memory
├── logs/               # Log files
└── tests/              # Test files (boring)
```

## 🔧 Advanced Stuff (if you're brave)

### AI Models You Can Use
- **LightGBM**: Fast and smart (default)
- **Random Forest**: Like a forest of decision trees
- **Logistic Regression**: Basic but works sometimes

### Risk Management Features
- Stop-loss so you don't lose your shirt
- Multiple take-profit levels 
- Position sizing based on how confident the AI is
- Cooldown periods (so you don't overtrade)

### Technical Indicators
- RSI, MACD (all that good stuff)
- Bollinger Bands
- Moving averages
- Volatility stuff
- And more math things

## 📈 How to Know if You're Winning

The bot logs everything:
- How good the signals are
- All your trades
- AI performance
- Real-time profit/loss

## 🤝 Wanna Contribute?

1. Fork this repo
2. Make your changes
3. Test your stuff
4. Submit a pull request
5. Hope I don't reject it (jk... mostly)

## ⚠️ SERIOUS WARNING

This is not financial advice! Crypto trading is risky as hell and you can lose all your money. Don't invest more than you can afford to lose. I'm not responsible if you blow up your account. Use at your own risk!

---

**Stuck?** Check the [AGENTS.md](AGENTS.md) file for more technical stuff.