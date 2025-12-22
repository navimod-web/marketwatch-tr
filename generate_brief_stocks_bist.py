"""
Daily Brief Generator - BIST Stocks v1.0
========================================
Reads bist_data.json and generates brief using OpenAI API.

Usage:
    export OPENAI_API_KEY=your_key
    python generate_brief_stocks_bist.py

Input:
    bist_data.json

Output:
    brief_stocks_bist.json
"""

import json
import os
from datetime import datetime
from openai import OpenAI

DATA_FILE = 'bist_data.json'
OUTPUT_FILE = 'brief_stocks_bist.json'
MODEL = 'gpt-5-mini'

SYSTEM_PROMPT = """You are a senior BIST (Borsa Istanbul) Equity Analyst. Write a daily stock market brief for portfolio managers.

═══════════════════════════════════════════════════════════════
IMPORTANT: BIST 50 STOCKS ONLY
═══════════════════════════════════════════════════════════════

This brief focuses ONLY on BIST 50 stocks.
- Use stock symbols like AKBNK, GARAN, THYAO, EREGL, TUPRS, etc.
- Analyze by these 8 sectors:
  * Banking & Finance: AKBNK, GARAN, ISCTR, YKBNK, HALKB, VAKBN, TSKB
  * Holding & Investment: KCHOL, SAHOL, DOHOL
  * Construction & REIT: ALARK, ENKAI, TKFEN, EKGYO
  * Heavy Industry: EREGL, KRDMD, FROTO, TOASO, ARCLK, VESTL, ASELS
  * Energy & Chemicals: TUPRS, PETKM, SASA, HEKTS, GUBRF, SISE, OYAKC, TRALT, TRMET
  * Retail & Food: BIMAS, MGROS, SOKM, CCOLA, AEFES, ULKER
  * Transportation: THYAO, PGSUS, TAVHL
  * Technology & Telecom: TCELL, TTKOM, KONTR

═══════════════════════════════════════════════════════════════
REGIME LOGIC (VOLATILITY + BREADTH BASED)
═══════════════════════════════════════════════════════════════

The data provides a pre-calculated regime based on:
- VOLATILITY: 20-day annualized XU100 volatility
  * <25% = Low (Score +1)
  * 25-40% = Normal (Score 0)  
  * >40% = High (Score -1)
- BREADTH: Sectors with positive 1W returns
  * ≥5/8 = Strong
  * 3-4/8 = Mixed
  * ≤2/8 = Weak

4 VALID REGIMES:
- RISK-ON: Low/Normal volatility + Strong/Mixed breadth
- RISK-OFF: High volatility + Weak/Mixed breadth
- CAUTION: Conflicting signals (low vol + weak breadth, OR high vol + strong breadth)
- NEUTRAL: Normal volatility + Mixed breadth

═══════════════════════════════════════════════════════════════
TREND LABELS - USE THESE 4 STRICTLY
═══════════════════════════════════════════════════════════════

- "ABOVE TREND" = Both 1W and 1M are POSITIVE
- "BELOW TREND" = Both 1W and 1M are NEGATIVE
- "RECOVERY" = 1M is NEGATIVE but 1W is POSITIVE (Short-term bounce)
- "CORRECTION" = 1M is POSITIVE but 1W is NEGATIVE (Short-term pullback)

═══════════════════════════════════════════════════════════════
WRITING STYLE - CRITICAL
═══════════════════════════════════════════════════════════════

1. NO markdown (no **, no ---, no bullets)
2. Write like a Bloomberg terminal note - SHORT and PUNCHY
3. Each answer: 2-3 sentences MAX
4. Use this format for data: "SYMBOL (1W: +X% | 1M: +X%)"
5. Use ONLY these 4 trend labels: ABOVE TREND, BELOW TREND, RECOVERY, CORRECTION

GOOD STYLE:
"Normal volatility environment with mixed breadth. Vol: 32%. Breadth: 5/8 sectors positive. Banking ABOVE TREND: AKBNK (1W: +3.2% | 1M: +8.5%). Energy in CORRECTION: TUPRS (1W: -2.1% | 1M: +12.3%)."

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT - FOLLOW EXACTLY
═══════════════════════════════════════════════════════════════

## MARKET OVERVIEW

**1. What is the Current Market Regime?**
[State regime with VOLATILITY and BREADTH context]

**2. Which sectors are leading?**
[Top sectors with best performers and correct labels]

**3. Which sectors are lagging?**
[Bottom sectors with correct labels]

## STOCK PERFORMANCE

**4. Top Performing Stocks?**
[Top 5 stocks with correct labels]

**5. Worst Performing Stocks?**
[Bottom 5 stocks with correct labels]

**6. Momentum Stocks?**
[Stocks with strongest 1W moves]

## SECTOR OUTLOOK

**7. Banking Sector Outlook?**
[Key banking stocks with labels]

**8. Industrial Sector Outlook?**
[Key industrial stocks with labels]

**9. Energy & Chemicals Outlook?**
[Key energy stocks with labels]

## PORTFOLIO STRATEGY

**10. Portfolio Recommendation?**
[Clear stance based on regime]

## NEXT WEEK OUTLOOK

**11. Stocks to Watch Next Week?**
[3-5 stocks with potential]

**12. Sectors to Watch Next Week?**
[2-3 sectors to monitor]

**13. Key Levels & Triggers?**
[XU100 levels and catalysts]

## EXECUTIVE SUMMARY

[2-3 sentence summary of market regime, key opportunities, and risks]
"""

def load_data():
    """Load bist_data.json file"""
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_trend_status(ret_1w, ret_1m):
    """Determine trend status based on 1W and 1M returns"""
    if ret_1w > 0 and ret_1m > 0:
        return "ABOVE TREND"
    elif ret_1w < 0 and ret_1m < 0:
        return "BELOW TREND"
    elif ret_1w > 0 and ret_1m < 0:
        return "RECOVERY"
    elif ret_1w < 0 and ret_1m > 0:
        return "CORRECTION"
    else:
        return "NEUTRAL"

def get_delta_direction(ret_1w):
    """Get delta direction symbol"""
    if ret_1w > 1:
        return "↑ Strong"
    elif ret_1w > 0:
        return "↗ Positive"
    elif ret_1w < -1:
        return "↓ Weak"
    elif ret_1w < 0:
        return "↘ Negative"
    else:
        return "→ Flat"

def format_for_prompt(data):
    """Format data for LLM prompt"""
    lines = []
    
    # === REGIME ===
    regime = data.get('regime', {})
    lines.append("=" * 60)
    lines.append("MARKET REGIME")
    lines.append("=" * 60)
    lines.append(f"Overall: {regime.get('overall', 'N/A')}")
    lines.append(f"Note: {regime.get('note', 'N/A')}")
    
    # Volatility info
    volatility = regime.get('volatility')
    if volatility:
        lines.append(f"Volatility: {volatility:.1f}% (20-day annualized XU100)")
    lines.append(f"Volatility Score: {regime.get('riskScore', 0)} (+1=Low, 0=Normal, -1=High)")
    
    breadth = regime.get('breadth', {})
    lines.append(f"Breadth: {breadth.get('positive', 0)}/{breadth.get('total', 0)} sectors positive")
    
    # Signals
    lines.append("\nSIGNALS:")
    for name, sig in regime.get('signals', {}).items():
        lines.append(f"  {name}: {sig.get('value', 'N/A')} (Score: {sig.get('score', 0)})")
    
    # === STOCKS ===
    stocks = data.get('stocks', [])
    stocks_full = []
    
    for s in stocks:
        if s.get('Category') == 'Index':
            continue
        
        sym = s['Symbol'].replace('.IS', '')
        name = s.get('Name', sym)
        cat = s.get('Category', 'Other')
        
        w1 = s.get('1W', {}) or {}
        m1 = s.get('1M', {}) or {}
        
        ret_1w = w1.get('RETURN') or 0
        ret_1m = m1.get('RETURN') or 0
        trend_1m = m1.get('TREND') or 0
        
        status = get_trend_status(ret_1w, ret_1m)
        delta = get_delta_direction(ret_1w)
        
        stocks_full.append({
            'sym': sym,
            'name': name,
            'cat': cat,
            'ret_1w': ret_1w,
            'ret_1m': ret_1m,
            'trend': trend_1m,
            'status': status,
            'delta': delta
        })
    
    # Sort by 1W return
    stocks_full.sort(key=lambda x: x['ret_1w'], reverse=True)
    
    lines.append(f"\n{'='*60}")
    lines.append(f"STOCK PERFORMANCE ({len(stocks_full)} stocks)")
    lines.append(f"{'='*60}")
    
    lines.append("\n📈 TOP 10 PERFORMERS:")
    for i, e in enumerate(stocks_full[:10], 1):
        lines.append(f"  {i}. {e['sym']:6} ({e['cat']:20})")
        lines.append(f"     1W: {e['ret_1w']:+6.2f}% | 1M: {e['ret_1m']:+6.2f}% | {e['status']}")
    
    lines.append("\n📉 BOTTOM 10 PERFORMERS:")
    for i, e in enumerate(stocks_full[-10:], 1):
        lines.append(f"  {i}. {e['sym']:6} ({e['cat']:20})")
        lines.append(f"     1W: {e['ret_1w']:+6.2f}% | 1M: {e['ret_1m']:+6.2f}% | {e['status']}")
    
    # === KEY STOCKS Detail ===
    lines.append(f"\n{'='*60}")
    lines.append("KEY STOCKS DETAILED VIEW")
    lines.append(f"{'='*60}")
    
    key_stocks = ['AKBNK', 'GARAN', 'ISCTR', 'YKBNK',  # Banking
                  'THYAO', 'PGSUS',  # Transportation
                  'EREGL', 'FROTO', 'TOASO', 'ASELS',  # Industry
                  'TUPRS', 'SASA', 'SISE',  # Energy/Chemicals
                  'BIMAS', 'MGROS',  # Retail
                  'TCELL', 'KCHOL', 'SAHOL']  # Tech & Holdings
    
    for sym in key_stocks:
        for s in data.get('stocks', []):
            if s['Symbol'].replace('.IS', '') == sym:
                w1 = s.get('1W', {}) or {}
                m1 = s.get('1M', {}) or {}
                ret_1w = w1.get('RETURN') or 0
                ret_1m = m1.get('RETURN') or 0
                trend_1m = m1.get('TREND') or 0
                status = get_trend_status(ret_1w, ret_1m)
                delta = get_delta_direction(ret_1w)
                
                lines.append(f"\n{sym} ({s['Name']}):")
                lines.append(f"  Returns: 1W: {ret_1w:+.2f}% | 1M: {ret_1m:+.2f}%")
                lines.append(f"  Trend (annualized): {trend_1m:+.2f}%")
                lines.append(f"  Direction: {delta}")
                lines.append(f"  Status: {status}")
                break
    
    # === SECTOR SUMMARY ===
    lines.append(f"\n{'='*60}")
    lines.append("SECTOR SUMMARY (1W Based)")
    lines.append(f"{'='*60}")
    
    sector_perf = {}
    for e in stocks_full:
        cat = e['cat']
        if cat not in sector_perf:
            sector_perf[cat] = {'ret_1w': [], 'ret_1m': []}
        sector_perf[cat]['ret_1w'].append(e['ret_1w'])
        sector_perf[cat]['ret_1m'].append(e['ret_1m'])
    
    sector_avg = []
    breadth_positive = 0
    for cat, data_dict in sector_perf.items():
        avg_1w = sum(data_dict['ret_1w']) / len(data_dict['ret_1w']) if data_dict['ret_1w'] else 0
        avg_1m = sum(data_dict['ret_1m']) / len(data_dict['ret_1m']) if data_dict['ret_1m'] else 0
        sector_avg.append((cat, avg_1w, avg_1m, len(data_dict['ret_1w'])))
        if avg_1w > 0:
            breadth_positive += 1
    
    sector_avg.sort(key=lambda x: x[1], reverse=True)
    
    lines.append(f"\n  📊 SECTOR BREADTH: {breadth_positive}/{len(sector_avg)} sectors positive (1W)")
    lines.append("")
    
    for cat, avg_1w, avg_1m, count in sector_avg:
        status = "✅" if avg_1w > 0 else "❌"
        lines.append(f"  {status} {cat:20}: 1W: {avg_1w:+.2f}% | 1M: {avg_1m:+.2f}% ({count} stocks)")
    
    # === QUANT RANKINGS ===
    rankings = data.get('quant_rankings', {})
    
    lines.append(f"\n{'='*60}")
    lines.append("QUANT RANKINGS")
    lines.append(f"{'='*60}")
    
    # Top 10 Stocks
    top10 = rankings.get('top10_stocks', [])
    if top10:
        lines.append("\n🏆 TOP 10 QUANT SCORE:")
        for i, s in enumerate(top10, 1):
            sym = s.get('Symbol', '').replace('.IS', '')
            score = s.get('QuantScore', 0)
            cat = s.get('Category', '')
            lines.append(f"  {i}. {sym} ({cat}): {score:.1f}")
    
    # All Sectors
    all_sectors = rankings.get('all_sectors', [])
    if all_sectors:
        lines.append("\n📊 SECTOR QUANT RANKINGS:")
        for s in all_sectors:
            if s.get('Disqualified'):
                lines.append(f"  ❌ {s['Sector']}: DISQUALIFIED ({s.get('Reason', 'N/A')})")
            else:
                lines.append(f"  ✅ {s['Sector']}: Score {s.get('QuantScore', 0):.1f} ({s.get('StockCount', 0)} stocks)")
    
    return "\n".join(lines)

def generate_brief(data):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set!")
    
    client = OpenAI(api_key=api_key)
    
    today = datetime.now().strftime('%B %d, %Y')
    prompt_data = format_for_prompt(data)
    
    print("🤖 Calling OpenAI API...")
    print(f"   Model: {MODEL}")
    print(f"   Data size: {len(prompt_data)} chars")
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"""TODAY'S DATE: {today}

{prompt_data}

IMPORTANT REMINDERS:
1. Every stock mention MUST have dual timeframe: (1W: X% | 1M: Y%)
2. Every stock MUST have trend status (ABOVE/BELOW TREND, RECOVERY, CORRECTION)
3. Focus on sector leaders and laggards
4. Compare 1W vs 1M to identify confirmations and divergences

Generate the BIST Stock Daily Brief now."""}
        ],
        max_completion_tokens=4000
    )
    
    content = None
    if response.choices and len(response.choices) > 0:
        choice = response.choices[0]
        if hasattr(choice.message, 'content') and choice.message.content:
            content = choice.message.content
    
    if not content:
        print("❌ No content found in response!")
        content = "Brief generation failed - no content returned from API"
    
    return {
        'date': today,
        'generated_at': datetime.now().isoformat(),
        'model': MODEL,
        'content': content
    }

def main():
    print("=" * 60)
    print("🤖 BIST Stock Brief Generator v1.0")
    print("   + Volatility-based Regime")
    print("   + Sector Quant Rankings")
    print("   + Time-Frame Labels (1W vs 1M)")
    print("=" * 60)
    
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: {DATA_FILE} not found!")
        print("   Run bist_data_generator.py first")
        return
    
    try:
        data = load_data()
        stock_count = len(data.get('stocks', []))
        print(f"✅ Loaded: {stock_count} Stocks")
        print(f"   Regime: {data['regime']['overall']}")
        
        vol = data['regime'].get('volatility')
        if vol:
            print(f"   Volatility: {vol:.1f}%")
        
        if stock_count == 0:
            print("❌ No stock data found!")
            return
        
        brief = generate_brief(data)
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(brief, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Brief saved to: {OUTPUT_FILE}")
        print("\n" + "=" * 60)
        print("📋 GENERATED BRIEF:")
        print("=" * 60)
        print(brief['content'][:2000] + "..." if len(brief['content']) > 2000 else brief['content'])
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        fallback = {
            'date': datetime.now().strftime('%B %d, %Y'),
            'generated_at': datetime.now().isoformat(),
            'error': str(e),
            'content': '## Brief Unavailable\n\nCould not generate brief. Check API key and try again.'
        }
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(fallback, f, indent=2)

if __name__ == "__main__":
    main()
