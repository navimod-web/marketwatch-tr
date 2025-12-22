"""
BIST Market Data Generator v2.0
===============================
Fetches BIST 50 stocks and index data from Yahoo Finance and produces JSON file.
HTML dashboard loads this JSON.

Usage:
    python bist_data_generator.py

Output:
    bist_data.json - Data file to be loaded by dashboard
"""

import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# BIST 50 STOCKS + ENDEKSLER (44 Sembol)
# =============================================================================
STOCK_UNIVERSE = {
    # === INDICES (Outside sectors - not included in breadth calculation) ===
    'XU030.IS': ('Index', 'Main', 'BIST 30'),
    'XU100.IS': ('Index', 'Main', 'BIST 100'),
    
    # === 1. BANKING & FINANCE (7 stocks) ===
    'AKBNK.IS': ('Banking & Finance', 'Bank', 'Akbank'),
    'GARAN.IS': ('Banking & Finance', 'Bank', 'Garanti BBVA'),
    'ISCTR.IS': ('Banking & Finance', 'Bank', 'Is Bank'),
    'YKBNK.IS': ('Banking & Finance', 'Bank', 'Yapi Kredi'),
    'HALKB.IS': ('Banking & Finance', 'Bank', 'Halkbank'),
    'VAKBN.IS': ('Banking & Finance', 'Bank', 'Vakifbank'),
    'TSKB.IS': ('Banking & Finance', 'Development', 'TSKB'),
    
    # === 2. HOLDING & INVESTMENT (3 stocks) ===
    'KCHOL.IS': ('Holding & Investment', 'Holding', 'Koc Holding'),
    'SAHOL.IS': ('Holding & Investment', 'Holding', 'Sabanci Holding'),
    'DOHOL.IS': ('Holding & Investment', 'Holding', 'Dogan Holding'),
    
    # === 3. CONSTRUCTION & REIT (4 stocks) ===
    'ALARK.IS': ('Construction & REIT', 'Contractor', 'Alarko Holding'),
    'ENKAI.IS': ('Construction & REIT', 'Contractor', 'Enka Insaat'),
    'TKFEN.IS': ('Construction & REIT', 'Contractor', 'Tekfen Holding'),
    'EKGYO.IS': ('Construction & REIT', 'REIT', 'Emlak Konut REIT'),
    
    # === 4. HEAVY INDUSTRY (7 stocks) ===
    'EREGL.IS': ('Heavy Industry', 'Steel', 'Eregli Steel'),
    'KRDMD.IS': ('Heavy Industry', 'Steel', 'Kardemir'),
    'FROTO.IS': ('Heavy Industry', 'Automotive', 'Ford Otosan'),
    'TOASO.IS': ('Heavy Industry', 'Automotive', 'Tofas'),
    'ARCLK.IS': ('Heavy Industry', 'Appliances', 'Arcelik'),
    'VESTL.IS': ('Heavy Industry', 'Electronics', 'Vestel'),
    'ASELS.IS': ('Heavy Industry', 'Defense', 'Aselsan'),
    
    # === 5. ENERGY, CHEMICALS & MINING (9 stocks) ===
    'TUPRS.IS': ('Energy & Chemicals', 'Refinery', 'Tupras'),
    'PETKM.IS': ('Energy & Chemicals', 'Petrochemical', 'Petkim'),
    'SASA.IS': ('Energy & Chemicals', 'Chemical', 'Sasa Polyester'),
    'HEKTS.IS': ('Energy & Chemicals', 'Agro Chemical', 'Hektas'),
    'GUBRF.IS': ('Energy & Chemicals', 'Fertilizer', 'Gubre Fabrikalari'),
    'SISE.IS': ('Energy & Chemicals', 'Glass', 'Sisecam'),
    'OYAKC.IS': ('Energy & Chemicals', 'Cement', 'Oyak Cement'),
    'TRALT.IS': ('Energy & Chemicals', 'Gold', 'Turk Altin'),
    'TRMET.IS': ('Energy & Chemicals', 'Gold', 'Turk Metal'),
    
    # === 6. RETAIL & FOOD (6 stocks) ===
    'BIMAS.IS': ('Retail & Food', 'Retail', 'BIM'),
    'MGROS.IS': ('Retail & Food', 'Retail', 'Migros'),
    'SOKM.IS': ('Retail & Food', 'Retail', 'Sok Marketler'),
    'CCOLA.IS': ('Retail & Food', 'Beverage', 'Coca-Cola Icecek'),
    'AEFES.IS': ('Retail & Food', 'Beverage', 'Anadolu Efes'),
    'ULKER.IS': ('Retail & Food', 'Food', 'Ulker'),
    
    # === 7. TRANSPORTATION (3 stocks) ===
    'THYAO.IS': ('Transportation', 'Aviation', 'Turkish Airlines'),
    'PGSUS.IS': ('Transportation', 'Aviation', 'Pegasus'),
    'TAVHL.IS': ('Transportation', 'Airport', 'TAV Airports'),
    
    # === 8. TECHNOLOGY & TELECOM (3 stocks) ===
    'TCELL.IS': ('Technology & Telecom', 'GSM', 'Turkcell'),
    'TTKOM.IS': ('Technology & Telecom', 'Telecom', 'Turk Telekom'),
    'KONTR.IS': ('Technology & Telecom', 'Technology', 'Kontrolmatik'),
}

# =============================================================================
# MARKET RATIOS (for regime detection - XU100 as benchmark)
# =============================================================================
MARKET_RATIOS = {
    "XU100_LEVEL": {"num": "XU100.IS", "denom": None, "cat": "Market", "name": "BIST 100 Level"},
    "BANK_XU100": {"num": "GARAN.IS", "denom": "XU100.IS", "cat": "Sector", "name": "Banking / BIST 100"},
    "INDUSTRY_XU100": {"num": "EREGL.IS", "denom": "XU100.IS", "cat": "Sector", "name": "Industry / BIST 100"},
    "TRANSPORT_XU100": {"num": "THYAO.IS", "denom": "XU100.IS", "cat": "Sector", "name": "Transport / BIST 100"},
}

PERIODS = {
    '1W': 5, '2W': 10, '1M': 21, '3M': 63, '6M': 126, '12M': 252
}

# Sector list (Index excluded - for breadth calculation)
SECTORS = [
    'Banking & Finance',
    'Holding & Investment',
    'Construction & REIT',
    'Heavy Industry',
    'Energy & Chemicals',
    'Retail & Food',
    'Transportation',
    'Technology & Telecom'
]

# =============================================================================
# CALCULATIONS
# =============================================================================
def fetch_prices(symbols, days=400):
    """Fetch prices from Yahoo Finance - with retry mechanism"""
    import time
    
    print(f"📡 Fetching {len(symbols)} symbols from Yahoo Finance...")
    end = datetime.now()
    start = end - timedelta(days=days)
    
    all_prices = pd.DataFrame()
    symbols_list = list(symbols)
    
    batch_size = 10
    max_retries = 5
    
    for i in range(0, len(symbols_list), batch_size):
        batch = symbols_list[i:i+batch_size]
        print(f"   Batch {i//batch_size + 1}/{(len(symbols_list)-1)//batch_size + 1}: {', '.join([s.replace('.IS','') for s in batch[:3]])}...")
        
        for attempt in range(max_retries):
            try:
                data = yf.download(
                    batch, 
                    start=start, 
                    end=end, 
                    progress=False, 
                    auto_adjust=True,
                    threads=False
                )
                
                if len(batch) == 1:
                    prices = data['Close'] if 'Close' in data.columns else data
                    if isinstance(prices, pd.Series):
                        prices = prices.to_frame(batch[0])
                else:
                    prices = data['Close'] if 'Close' in data.columns else data
                
                if prices is not None and len(prices) > 0:
                    all_prices = pd.concat([all_prices, prices], axis=1)
                break
                
            except Exception as e:
                print(f"   ⚠️ Attempt {attempt+1}/{max_retries}: {str(e)[:40]}...")
                if attempt < max_retries - 1:
                    wait = (attempt + 1) * 3
                    print(f"   ⏳ Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"   ❌ Skipping: {', '.join(batch)}")
        
        time.sleep(1)
    
    all_prices = all_prices.loc[:, ~all_prices.columns.duplicated()]
    
    print(f"✅ Fetched {len(all_prices)} days, {len(all_prices.columns)} symbols")
    return all_prices

def calc_metrics(prices, period_days):
    """Calculate metrics for a period"""
    
    clean_prices = prices.dropna()
    
    min_required = max(3, period_days // 2)
    if len(clean_prices) < min_required:
        return None
        
    actual_days = min(period_days + 1, len(clean_prices))
    p = clean_prices.tail(actual_days)
    
    if len(p) < 3:
        return None
    
    ret = (p.iloc[-1] / p.iloc[0] - 1) * 100
    
    x = np.arange(len(p))
    try:
        slope, _, r, _, _ = stats.linregress(x, np.log(p.values + 0.0001))
    except Exception:
        return None
    trend = (np.exp(slope * 252) - 1) * 100
    quality = r ** 2
    
    mid = len(p) // 2
    if mid < 2:
        accel = 0
    else:
        try:
            slope1, _, _, _, _ = stats.linregress(np.arange(mid), np.log(p.iloc[:mid].values + 0.0001))
            slope2, _, _, _, _ = stats.linregress(np.arange(len(p)-mid), np.log(p.iloc[mid:].values + 0.0001))
            accel = (slope2 - slope1) * 252 * 100
        except Exception:
            accel = 0
    
    rets = p.pct_change().dropna()
    if len(rets) < 2:
        return None
    
    ann_ret = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    neg_rets = rets[rets < 0]
    down_vol = neg_rets.std() * np.sqrt(252) if len(neg_rets) > 0 else ann_vol
    sortino = ann_ret / down_vol if down_vol > 0 else 0
    sortino = max(-10, min(10, sortino))
    
    cum = (1 + rets).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    min_dd = min(dd, -0.005)
    stability = ann_ret / abs(min_dd)
    stability = max(-10, min(10, stability))
    
    return {
        'RETURN': round(ret, 2),
        'TREND': round(trend, 2),
        'ACCELERATION': round(accel, 2),
        'QUALITY': round(quality, 3),
        'SHARPE': round(sharpe, 2),
        'SORTINO': round(sortino, 2),
        'STABILITY': round(stability, 2)
    }

def calc_regime(ratios, stocks, prices):
    """Calculate market regime based on volatility and breadth"""
    
    signals = {}
    risk_score = 0
    
    # === SECTOR BREADTH (Index excluded - only 8 sectors) ===
    breadth_positive = 0
    breadth_total = 0
    sector_returns = {}
    
    for s in stocks:
        sector = s.get('Category', 'Other')
        # Exclude indices from breadth calculation
        if sector == 'Index':
            continue
            
        ret_1w = s.get('1W', {}).get('RETURN') or 0
        
        if sector not in sector_returns:
            sector_returns[sector] = []
        sector_returns[sector].append(ret_1w)
    
    for sector, returns in sector_returns.items():
        if returns:
            breadth_total += 1
            avg_return = sum(returns) / len(returns)
            if avg_return > 0:
                breadth_positive += 1
    
    breadth_ratio = breadth_positive / breadth_total if breadth_total > 0 else 0.5
    breadth_score = 1 if breadth_positive >= 5 else (-1 if breadth_positive <= 2 else 0)
    signals['BREADTH'] = {
        'value': f'{breadth_positive}/{breadth_total} POSITIVE',
        'score': breadth_score,
        'positive': breadth_positive,
        'total': breadth_total
    }
    
    # === VOLATILITY-BASED RISK SCORE ===
    # Calculate 20-day annualized volatility of XU100
    volatility = None
    volatility_signal = 'N/A'
    
    if 'XU100.IS' in prices and len(prices['XU100.IS']) >= 20:
        xu100_prices = prices['XU100.IS']
        # Get last 20 days
        recent_prices = xu100_prices.tail(21)  # 21 prices = 20 returns
        if len(recent_prices) >= 2:
            # Calculate daily returns
            daily_returns = recent_prices.pct_change().dropna()
            if len(daily_returns) >= 10:
                # Calculate annualized volatility
                daily_std = daily_returns.std()
                volatility = daily_std * (252 ** 0.5) * 100  # Annualized %
                
                # Determine risk score based on volatility
                # BIST thresholds: <25% low, 25-40% normal, >40% high
                if volatility < 25:
                    risk_score = 1
                    volatility_signal = f'LOW {volatility:.1f}%'
                elif volatility > 40:
                    risk_score = -1
                    volatility_signal = f'HIGH {volatility:.1f}%'
                else:
                    risk_score = 0
                    volatility_signal = f'NORMAL {volatility:.1f}%'
    
    signals['VOLATILITY'] = {
        'value': volatility_signal,
        'score': risk_score,
        'level': volatility if volatility else 0,
        'change': 0
    }
    
    # === SIMPLE REGIME ALGORITHM (Risk + Breadth) ===
    risk_cat = 'negative' if risk_score <= -1 else ('positive' if risk_score >= 1 else 'neutral')
    breadth_cat = 'weak' if breadth_positive <= 2 else ('strong' if breadth_positive >= 5 else 'mixed')
    
    regime_matrix = {
        # High volatility scenarios
        ('negative', 'weak'):   ('RISK-OFF', 'High volatility + weak breadth - defensive mode'),
        ('negative', 'mixed'):  ('CAUTION', 'High volatility + mixed breadth - be careful'),
        ('negative', 'strong'): ('CAUTION', 'High volatility but broad participation'),
        
        # Normal volatility scenarios
        ('neutral', 'weak'):    ('NEUTRAL', 'Normal volatility but narrow participation'),
        ('neutral', 'mixed'):   ('NEUTRAL', 'Normal volatility + mixed breadth'),
        ('neutral', 'strong'):  ('RISK-ON', 'Normal volatility + broad participation'),
        
        # Low volatility scenarios
        ('positive', 'weak'):   ('CAUTION', 'Low volatility but narrow participation'),
        ('positive', 'mixed'):  ('RISK-ON', 'Low volatility + moderate participation'),
        ('positive', 'strong'): ('RISK-ON', 'Low volatility + broad participation - strong bull'),
    }
    
    overall, regime_note = regime_matrix.get(
        (risk_cat, breadth_cat), 
        ('NEUTRAL', 'Regime undetermined')
    )
    
    return {
        'overall': overall,
        'riskScore': risk_score,
        'volatility': volatility,
        'signals': signals,
        'breadth': {'positive': breadth_positive, 'total': breadth_total},
        'note': regime_note
    }

def normalize_metric(value, min_val, max_val):
    """Normalize metric to 0-100 range"""
    if value is None:
        return 50
    if max_val == min_val:
        return 50
    normalized = (value - min_val) / (max_val - min_val) * 100
    return max(0, min(100, normalized))

def calc_overbought_penalty(prices, lookback=14):
    """Calculate overbought penalty based on 14-day Bollinger Band"""
    if prices is None or len(prices) < lookback:
        return 0
    
    recent = prices.tail(lookback)
    if len(recent) < lookback:
        return 0
    
    current_price = recent.iloc[-1]
    mean = recent.mean()
    std = recent.std()
    
    if std == 0 or std is None:
        return 0
    
    z_score = (current_price - mean) / std
    
    if z_score > 3:
        return 10
    elif z_score > 2:
        return 5
    else:
        return 0

def calc_max_drawdown_penalty(prices, period_days):
    """Calculate max drawdown penalty for specific period"""
    if prices is None or len(prices) < period_days:
        return 0
    
    recent = prices.tail(period_days).dropna()
    if len(recent) < 5:
        return 0
    
    cum_max = recent.cummax()
    drawdown = (recent - cum_max) / cum_max
    max_dd = drawdown.min()
    
    return abs(max_dd) * 100

def calc_quant_score(item, sector_1m_return=None, sector_2w_return=None, overbought_penalty=0, benchmark_6m_return=None, dd_1m=0, dd_3m=0, dd_6m=0):
    """Calculate Quant Score - with XU100 benchmark"""
    
    ret_1m = item.get('1M', {}).get('RETURN', 0) or 0
    if ret_1m < 1:
        return None
    
    PERIODS_LIST = ['1W', '2W', '1M', '3M', '6M']
    PERIOD_WEIGHT = 0.20
    
    METRIC_WEIGHTS = {
        'RETURN': 0.40,
        'SORTINO': 0.20,
        'STABILITY': 0.20,
        'TREND': 0.10,
        'ACCELERATION': 0.10
    }
    
    BOUNDS = {
        'RETURN': (-30, 30),
        'ACCELERATION': (-50, 50),
        'SORTINO': (-3, 3),
        'STABILITY': (-3, 3),
        'TREND': (-100, 100)
    }
    
    period_scores = {}
    negative_count = 0
    
    for period in PERIODS_LIST:
        if period not in item:
            continue
            
        metrics = item[period]
        metric_score = 0
        
        period_return = metrics.get('RETURN', 0)
        if period_return is not None and period_return < 0:
            negative_count += 1
        
        for metric, metric_weight in METRIC_WEIGHTS.items():
            value = metrics.get(metric)
            min_val, max_val = BOUNDS[metric]
            if value is not None:
                value = max(min_val, min(max_val, value))
            normalized = normalize_metric(value, min_val, max_val)
            metric_score += normalized * metric_weight
        
        period_scores[period] = metric_score
    
    if not period_scores:
        return None
    
    total_weight = 0
    weighted_sum = 0
    
    for period, score in period_scores.items():
        weighted_sum += score * PERIOD_WEIGHT
        total_weight += PERIOD_WEIGHT
    
    base_score = weighted_sum / total_weight if total_weight > 0 else 0
    
    penalty = negative_count * 10
    final_score = base_score - penalty
    final_score -= overbought_penalty
    
    ret_3m = item.get('3M', {}).get('RETURN', 0) or 0
    ret_6m = item.get('6M', {}).get('RETURN', 0) or 0
    if ret_1m < 2:
        final_score -= 10
    if ret_3m < 5:
        final_score -= 10
    
    # XU100 BENCHMARK FILTER
    if benchmark_6m_return is not None and ret_6m < benchmark_6m_return:
        return None
    
    if dd_1m > 5:
        final_score -= 10
    if dd_3m > 10:
        final_score -= 10
    if dd_6m > 15:
        final_score -= 10
    
    r2_1m = item.get('1M', {}).get('QUALITY', 1) or 1
    r2_3m = item.get('3M', {}).get('QUALITY', 1) or 1
    r2_6m = item.get('6M', {}).get('QUALITY', 1) or 1
    if r2_1m < 0.2:
        final_score -= 20
    if r2_3m < 0.3:
        final_score -= 20
    if r2_6m < 0.4:
        final_score -= 20
    
    if sector_1m_return is not None and sector_2w_return is not None:
        if sector_1m_return > 0 and sector_2w_return > 0:
            final_score += 5
    
    final_score = max(0, min(100, final_score))
    
    return round(final_score, 1)

def calc_sector_returns(stocks):
    """Calculate 1M and 2W average return for each sector"""
    sector_returns = {}
    
    for stock in stocks:
        category = stock.get('Category', 'Other')
        if category == 'Endeks':
            continue
        
        if category not in sector_returns:
            sector_returns[category] = {'1M': [], '2W': []}
        
        if '1M' in stock and stock['1M'].get('RETURN') is not None:
            sector_returns[category]['1M'].append(stock['1M']['RETURN'])
        
        if '2W' in stock and stock['2W'].get('RETURN') is not None:
            sector_returns[category]['2W'].append(stock['2W']['RETURN'])
    
    sector_avg = {}
    for category, data in sector_returns.items():
        avg_1m = sum(data['1M']) / len(data['1M']) if data['1M'] else 0
        avg_2w = sum(data['2W']) / len(data['2W']) if data['2W'] else 0
        sector_avg[category] = {'1M': avg_1m, '2W': avg_2w}
    
    return sector_avg

def generate_data():
    """Main data generation function"""
    
    stock_symbols = list(STOCK_UNIVERSE.keys())
    
    print(f"📡 Total symbols: {len(stock_symbols)} (42 stocks + 2 indices)")
    
    prices = fetch_prices(stock_symbols)
    
    # Stock Metrikleri (Endeksler dahil)
    print("📊 Calculating metrics...")
    stocks = []
    for symbol, (cat, subcat, name) in STOCK_UNIVERSE.items():
        if symbol not in prices.columns:
            continue
        
        stock = {
            'Symbol': symbol,
            'Name': name,
            'Category': cat,
            'SubCategory': subcat
        }
        
        for period, days in PERIODS.items():
            metrics = calc_metrics(prices[symbol], days)
            if metrics:
                stock[period] = metrics
        
        stocks.append(stock)
    
    print(f"✅ Calculated metrics for {len(stocks)} symbols")
    
    # Ratio Hesapla
    print("📈 Calculating market ratios...")
    ratios = []
    ratio_dict = {}
    
    for ratio_id, config in MARKET_RATIOS.items():
        num = config['num']
        denom = config['denom']
        
        if num not in prices.columns:
            continue
        if denom and denom not in prices.columns:
            continue
        
        ratio_data = {
            'id': ratio_id,
            'name': config['name'],
            'category': config['cat'],
            'values': {},
            'changes': {}
        }
        
        for period, days in PERIODS.items():
            if denom:
                ratio_prices = prices[num] / prices[denom]
            else:
                ratio_prices = prices[num]
            
            if len(ratio_prices) >= days:
                current = ratio_prices.iloc[-1]
                past = ratio_prices.iloc[-days]
                change = ((current / past) - 1) * 100
                
                ratio_data['values'][period] = round(float(current), 4)
                ratio_data['changes'][period] = round(float(change), 2)
        
        ratios.append(ratio_data)
        
        if ratio_data['values']:
            ratio_dict[ratio_id] = {
                '1M': ratio_data['values'].get('1M', 0),
                '1M_chg': ratio_data['changes'].get('1M', 0),
                '1W_chg': ratio_data['changes'].get('1W', 0)
            }
    
    print(f"✅ Calculated {len(ratios)} ratios")
    
    # Calculate regime
    regime = calc_regime(ratio_dict, stocks, prices)
    vol_str = f"{regime.get('volatility', 0):.1f}%" if regime.get('volatility') else "N/A"
    print(f"✅ Regime: {regime['overall']} (Volatility: {vol_str}, Breadth: {regime['breadth']['positive']}/{regime['breadth']['total']})")
    print(f"   Note: {regime.get('note', '')}")
    
    # === QUANT RANKINGS ===
    print("🏆 Calculating Quant Rankings...")
    
    # Get XU100 6M return (benchmark)
    xu100_6m_return = None
    xu100_stock = next((s for s in stocks if s['Symbol'] == 'XU100.IS'), None)
    if xu100_stock and '6M' in xu100_stock:
        xu100_6m_return = xu100_stock['6M'].get('RETURN', 0)
        print(f"📊 XU100 6M Return (benchmark): {xu100_6m_return:.2f}%")
    
    # Stock Quant Scores (Indices excluded)
    sector_avg = calc_sector_returns(stocks)
    for stock in stocks:
        # Don't give quant score to indices
        if stock['Category'] == 'Index':
            stock['QuantScore'] = None
            continue
            
        symbol = stock['Symbol']
        category = stock.get('Category', 'Other')
        sect_data = sector_avg.get(category, {'1M': 0, '2W': 0})
        ob_penalty = 0
        dd_1m, dd_3m, dd_6m = 0, 0, 0
        if symbol in prices.columns:
            ob_penalty = calc_overbought_penalty(prices[symbol])
            dd_1m = calc_max_drawdown_penalty(prices[symbol], 22)
            dd_3m = calc_max_drawdown_penalty(prices[symbol], 66)
            dd_6m = calc_max_drawdown_penalty(prices[symbol], 126)
        stock['QuantScore'] = calc_quant_score(
            stock, 
            sector_1m_return=sect_data['1M'],
            sector_2w_return=sect_data['2W'],
            overbought_penalty=ob_penalty,
            benchmark_6m_return=xu100_6m_return,
            dd_1m=dd_1m,
            dd_3m=dd_3m,
            dd_6m=dd_6m
        )
    
    # Top 10 Stocks (Indices excluded)
    stocks_with_score = [s for s in stocks if s.get('QuantScore') is not None and s['Category'] != 'Index']
    top10_stocks = sorted(stocks_with_score, key=lambda x: x['QuantScore'], reverse=True)[:10]
    
    # === SECTOR QUANT SCORES ===
    print("📊 Calculating Sector Rankings...")
    
    # Collect sector-level metrics
    sector_metrics = {}
    for stock in stocks:
        sector = stock.get('Category', 'Other')
        if sector == 'Index':
            continue
        
        if sector not in sector_metrics:
            sector_metrics[sector] = {
                'stocks': [],
                'RETURN_2W': [], 'RETURN_1M': [], 'RETURN_3M': [], 'RETURN_6M': [],
                'TREND_1M': [], 'TREND_3M': [], 'TREND_6M': [],
                'SORTINO_1M': [], 'SORTINO_3M': [], 'SORTINO_6M': [],
                'STABILITY_1M': [], 'STABILITY_3M': [], 'STABILITY_6M': [],
                'ACCELERATION_1M': [], 'ACCELERATION_3M': [], 'ACCELERATION_6M': [],
                'QUALITY_1M': [], 'QUALITY_3M': [], 'QUALITY_6M': []
            }
        
        sector_metrics[sector]['stocks'].append(stock['Symbol'])
        
        # 2W return
        if '2W' in stock and stock['2W'].get('RETURN') is not None:
            sector_metrics[sector]['RETURN_2W'].append(stock['2W']['RETURN'])
        
        # Collect metrics for each period
        for period in ['1M', '3M', '6M']:
            if period in stock:
                for metric in ['RETURN', 'TREND', 'SORTINO', 'STABILITY', 'ACCELERATION', 'QUALITY']:
                    val = stock[period].get(metric)
                    if val is not None:
                        sector_metrics[sector][f'{metric}_{period}'].append(val)
    
    # Calculate sector averages and QuantScore
    def avg(arr):
        return sum(arr) / len(arr) if arr else 0
    
    sector_rankings = []
    for sector, data in sector_metrics.items():
        stock_count = len(data['stocks'])
        
        # Sector average metrics
        avg_2w_return = avg(data['RETURN_2W'])
        avg_1m_return = avg(data['RETURN_1M'])
        avg_3m_return = avg(data['RETURN_3M'])
        avg_6m_return = avg(data['RETURN_6M'])
        
        # === SECTOR FILTERS ===
        disqualified = False
        disqualify_reason = None
        
        # 2W Return < 0 → disqualified
        if avg_2w_return < 0:
            disqualified = True
            disqualify_reason = f"2W Ret {avg_2w_return:.1f}%"
        
        if disqualified:
            sector_rankings.append({
                'Sector': sector,
                'QuantScore': None,
                'StockCount': stock_count,
                'Disqualified': True,
                'Reason': disqualify_reason
            })
            continue
        
        # Calculate Sector QuantScore
        # Period Weights: 1M, 3M, 6M equal weight
        # Metric Weights: Return=40%, Sortino=20%, Stability=20%, Trend=10%, Acceleration=10%
        
        score = 0
        periods_data = [
            ('1M', avg(data['RETURN_1M']), avg(data['SORTINO_1M']), avg(data['STABILITY_1M']), avg(data['TREND_1M']), avg(data['ACCELERATION_1M']), avg(data['QUALITY_1M'])),
            ('3M', avg(data['RETURN_3M']), avg(data['SORTINO_3M']), avg(data['STABILITY_3M']), avg(data['TREND_3M']), avg(data['ACCELERATION_3M']), avg(data['QUALITY_3M'])),
            ('6M', avg(data['RETURN_6M']), avg(data['SORTINO_6M']), avg(data['STABILITY_6M']), avg(data['TREND_6M']), avg(data['ACCELERATION_6M']), avg(data['QUALITY_6M']))
        ]
        
        period_weight = 1/3  # Equal weight for each period
        
        for period, ret, sortino, stability, trend, accel, quality in periods_data:
            # Normalize to 0-100 range
            ret_norm = max(0, min(100, (ret + 30) / 60 * 100))
            sortino_norm = max(0, min(100, (sortino + 3) / 6 * 100))
            stability_norm = max(0, min(100, (stability + 3) / 6 * 100))
            trend_norm = max(0, min(100, (trend + 50) / 100 * 100))
            accel_norm = max(0, min(100, (accel + 50) / 100 * 100))
            
            period_score = (ret_norm * 0.40 + sortino_norm * 0.20 + stability_norm * 0.20 + 
                          trend_norm * 0.10 + accel_norm * 0.10)
            
            # Negative return penalty
            if ret < 0:
                period_score -= 10
            
            score += period_score * period_weight
        
        # Low return penalties
        if avg_1m_return < 2:
            score -= 10
        if avg_3m_return < 5:
            score -= 10
        
        # R² (Quality) penalties
        if avg(data['QUALITY_1M']) < 0.2:
            score -= 20
        if avg(data['QUALITY_3M']) < 0.3:
            score -= 20
        if avg(data['QUALITY_6M']) < 0.4:
            score -= 20
        
        score = max(0, min(100, score))
        
        sector_rankings.append({
            'Sector': sector,
            'QuantScore': round(score, 1),
            'StockCount': stock_count,
            'Disqualified': False,
            'Reason': None
        })
    
    # Sort: qualified by score, then disqualified alphabetically
    qualified = sorted([s for s in sector_rankings if not s['Disqualified']], key=lambda x: x['QuantScore'], reverse=True)
    disqualified_sectors = sorted([s for s in sector_rankings if s['Disqualified']], key=lambda x: x['Sector'])
    all_sectors = qualified + disqualified_sectors
    
    quant_rankings = {
        'top10_stocks': [{'Symbol': s['Symbol'], 'Name': s['Name'], 'Category': s['Category'], 'QuantScore': s['QuantScore']} for s in top10_stocks],
        'all_sectors': all_sectors
    }
    
    qualified_sectors = [s['Sector'] for s in all_sectors if not s['Disqualified']]
    disqualified_sectors_list = [s['Sector'] for s in all_sectors if s['Disqualified']]
    print(f"✅ Top 10 Stocks: {[s['Symbol'].replace('.IS','') for s in top10_stocks]}")
    print(f"✅ Sectors: {len(qualified_sectors)} qualified, {len(disqualified_sectors_list)} disqualified")
    if disqualified_sectors_list:
        print(f"   Disqualified: {disqualified_sectors_list}")
    
    return {
        'generated_at': datetime.now().isoformat(),
        'update_time': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'stock_count': len(stocks),
        'ratio_count': len(ratios),
        'regime': regime,
        'ratios': ratios,
        'stocks': stocks,
        'quant_rankings': quant_rankings
    }

def main():
    print("=" * 60)
    print("📊 BIST Market Data Generator v2.2")
    print("   42 Stocks + 2 Indices | 8 Sectors")
    print("=" * 60)
    
    try:
        data = generate_data()
        
        def clean_nan(obj):
            if isinstance(obj, dict):
                return {k: clean_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nan(item) for item in obj]
            elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return None
            else:
                return obj
        
        data = clean_nan(data)
        
        with open('bist_data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Data saved to: bist_data.json")
        print(f"   {data['stock_count']} Symbols, {data['ratio_count']} Ratios")
        print(f"   Regime: {data['regime']['overall']}")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
