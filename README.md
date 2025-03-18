# Helios AI Governance

This Python-based AI governance tool autonomously browses the web, collects blockchain asset data, and determines weight adjustments for governance proposals in the Helios blockchain.

## Features:
- Automatically selects the best AI model based on system resources.
- Scrapes search engines for real-time market, security, and regulatory data.
- Uses Mixtral 8x7B, LLaMA 3 13B, or Mistral 7B depending on available RAM.
- Summarizes asset risks and suggests governance decisions.

## Installation:
```
pip install -r requirements.txt
```

## Usage:
```
python helios_ai.py
```

## Dependencies:
- Python 3.8+
- Transformers library (for AI inference)
- Google Search API or DuckDuckGo API
