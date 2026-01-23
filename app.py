import streamlit as st
from datetime import date, timedelta
import yfinance as yf
from plotly import graph_objs as go
import pandas as pd
import numpy as np
from openai import OpenAI
import google.generativeai as genai
import json
import logging
import threading
import pandas_market_calendars as mcal  # 新增: 用於處理交易日

# 設定 logging 以便 debug
logging.basicConfig(level=logging.INFO)

# --- 1. 頁面設定 ---
st.set_page_config(page_title="Alpha-Sim: 2026 AI 戰情室 (Pro)", layout="wide", page_icon="🛡️")

# 自定義 CSS
st.markdown("""
    <style>
    .stButton>button {width: 100%; border-radius: 8px; height: 3.5em; font-weight: bold; font-size: 1.2em;}
    .report-box {background-color: #f0f2f6; border-left: 6px solid #ff4b4b; padding: 20px; border-radius: 10px; margin-bottom: 20px;}
    .metric-container {background-color: #ffffff; padding: 10px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);}
    div[data-testid="stMetricValue"] {font-size: 1.4rem;}
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Alpha-Sim: 雙引擎量化決策系統 (2026 進化版)")
st.caption("核心：Perplexity (情報) + Gemini (決策) + Monte Carlo (風控) + Fundamental (價值過濾)")

# --- 2. 核心邏輯函數 ---
@st.cache_data(ttl=3600)
def get_market_data(ticker):
    """抓取歷史股價數據 (含 MultiIndex 修正)"""
    try:
        data = yf.download(ticker, period="2y", interval="1d", auto_adjust=True, progress=False)
        
        # 更健壯的 MultiIndex 處理
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)  # 假設 level 1 是 ticker，重置為單層
        elif data.empty:
            return None
        
        data = data.ffill()
        return data
    except Exception as e:
        logging.error(f"get_market_data error: {e}")
        return None

@st.cache_data(ttl=3600)
def get_fundamentals(ticker):
    """抓取基本面數據：避開 PE 陷阱"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # 提取關鍵數據，若無則回傳 N/A
        fundamentals = {
            "pe_ratio": info.get('trailingPE', None),
            "forward_pe": info.get('forwardPE', None),
            "peg_ratio": info.get('pegRatio', None),  # 重點：本益成長比
            "price_to_book": info.get('priceToBook', None),
            "free_cashflow": info.get('freeCashflow', None),  # 重點：真錢
            "debt_to_equity": info.get('debtToEquity', None),
            "profit_margins": info.get('profitMargins', None),
            "roic": info.get('returnOnEquity', None),  # 使用 ROE 作為 ROIC 近似
            "fcf_yield": (info.get('freeCashflow', 0) / info.get('marketCap', 1)) if info.get('marketCap', 1) != 0 else None,
            "revenue_growth": info.get('revenueGrowth', None),
            "industry": info.get('industry', 'Unknown')
        }
        return fundamentals
    except Exception as e:
        logging.error(f"get_fundamentals error: {e}")
        return {}

def get_perplexity_intel(ticker, api_key):
    """引擎 A: 搜尋 2026 最新全球情報"""
    if not api_key:
        return "API Key 缺失。"
    
    client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
    today_str = date.today().strftime("%Y-%m-%d")
    
    prompt = f"""
    Search for the latest financial news, institutional reports, and supply chain rumors for {ticker} as of today ({today_str}).
    Focus on:
    1. Revenue/Earnings outlook for 2026.
    2. Key themes (e.g., AI Server, iPhone 17, Tesla).
    3. Institutional sentiment.
    Keep it concise, factual, and forward-looking.
    """
    
    try:
        response = client.chat.completions.create(
            model="sonar-pro",
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Perplexity Error: {e}")
        return f"Perplexity Error: {str(e)}"

def get_gemini_decision(ticker, api_key, intel_text, fundamentals):
    """引擎 B: 2026 旗艦決策 (加入基本面過濾邏輯)"""
    if not api_key:
        return None
    
    genai.configure(api_key=api_key)
    
    # 將基本面數據轉換為文字描述
    fund_str = json.dumps(fundamentals, indent=2)
    
    prompt = f"""
You are a strictly Quantitative Portfolio Manager (Professor level) in year 2026.
You **MUST NEVER** make buy/sell/hold decisions primarily based on trailing PE ratio.
Trailing PE is often misleading in 2026 due to one-time gains, accounting tricks, or cycle peaks.

CRITICAL RULES - Follow these strictly in order:
1. PEG Ratio is the #1 valuation metric. If PEG > 2.0 → likely overvalued. If PEG < 1.0 and growing → attractive.
2. Free Cash Flow is the #2 truth detector. If FCF negative or sharply declining → be extremely cautious or SELL, even if PE looks low.
3. For cyclical industries (memory, shipping, steel, panel, traditional manufacturing): Low PE is usually a SELL signal at cycle peak, NOT a buy.
4. Only use forward PE as a secondary check, never as primary driver.
5. Debt/Equity > 100% + negative FCF = high risk.
6. Profit margin compression + revenue slowdown = red flag.

Target: {ticker}

[Input: Latest Market Intel - 2026 outlook]
{intel_text}

[Input: Core Fundamentals - The Real Numbers]
{fund_str}

Task:
1. Analyze ONLY using the rules above + intel.
2. Output Expected Annual Return (arithmetic mean) in range [-0.40 ~ +0.60]
3. Output Volatility Multiplier [0.8 ~ 2.5]

Output STRICT JSON only:
{{
  "verdict": "BUY" | "HOLD" | "SELL",
  "reasoning": "必須詳細解釋為何用 PEG/FCF/週期特性判斷，不能只提 PE...",
  "expected_return": 0.12,
  "volatility_multiplier": 1.3
}}
    """
    
    # --- 2026 旗艦模型清單 (更新為潛在未來穩定版，保留 fallback) ---
    models = [
        "gemini-3-flash-preview",
        "gemini-2.5-flash-preview",
        "gemini-flash-latest",
        "gemini-2.0-flash"
    ]
    
    for model_name in models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            # 安全解析 JSON
            try:
                result = json.loads(response.text)
            except json.JSONDecodeError:
                continue  # 如果解析失敗，試下一個模型
            result['model_used'] = model_name
            return result
        except Exception as e:
            logging.error(f"Gemini model {model_name} error: {e}")
            continue
    
    # Fallback 值
    return {
        "verdict": "HOLD",
        "reasoning": "AI 模型回應失敗，使用預設中立立場。",
        "expected_return": 0.05,
        "volatility_multiplier": 1.0,
        "model_used": "Fallback"
    }

def run_monte_carlo(data, days, sims, expected_return, vol_mult, seed=42):
    """執行 2026 量化風險模擬 (GBM)，添加 seed 以重現"""
    np.random.seed(seed)  # 可重現性
    log_returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
    hist_vol = log_returns.std() * np.sqrt(250)  # 台灣股市約 250 交易日
    
    adj_vol = hist_vol * max(0.8, min(2.5, vol_mult))  # 調整範圍
    daily_vol = adj_vol / np.sqrt(250)
    daily_drift = (expected_return / 250) - (0.5 * daily_vol ** 2)
    
    last_price = data['Close'].iloc[-1]
    
    Z = np.random.normal(0, 1, (days, sims))
    daily_returns = np.exp(daily_drift + daily_vol * Z)
    
    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = last_price
    
    for t in range(1, days):
        price_paths[t] = price_paths[t-1] * daily_returns[t]
    
    return price_paths

def get_trading_days(start_date, days_to_predict, exchange='XTAI'):  # 新增: 獲取交易日
    """獲取未來交易日 (台灣股市 TSE)"""
    cal = mcal.get_calendar(exchange)
    end_date = start_date + timedelta(days=days_to_predict * 2)  # 多抓一些以防
    schedule = cal.schedule(start_date=start_date, end_date=end_date)
    return schedule.index[:days_to_predict]  # 取前 N 個交易日

# --- 3. Session State 管理器 (提高安全，不用 cookie) ---
if 'pplx_key' not in st.session_state:
    st.session_state['pplx_key'] = ''
if 'google_key' not in st.session_state:
    st.session_state['google_key'] = ''

# --- 4. 側邊欄：設定 ---
with st.sidebar:
    st.header("🔑 金鑰管理")
    pplx_api_key = st.text_input("Perplexity API Key", type="password", value=st.session_state['pplx_key'])
    google_api_key = st.text_input("Google AI Studio Key", type="password", value=st.session_state['google_key'])
    
    col_btn1, col_btn2 = st.columns(2)
    if col_btn1.button("記住我"):
        st.session_state['pplx_key'] = pplx_api_key
        st.session_state['google_key'] = google_api_key
        st.success("✅ 已儲存於 session (伺服器端)")
    if col_btn2.button("清除"):
        st.session_state['pplx_key'] = ''
        st.session_state['google_key'] = ''
        st.rerun()
    
    st.divider()
    st.header("🎯 標的設定")
    selected_stock = st.text_input("股票代碼", "2330.TW")
    simulations = st.slider("模擬路徑數", 500, 3000, 1000)
    days_to_predict = st.slider("預測天數", 30, 365, 90)

# --- 5. 主程式執行 ---
data = get_market_data(selected_stock)
if data is None:
    st.error(f"❌ 無法獲取 {selected_stock} 數據。")
else:
    last_price = data['Close'].iloc[-1]
    
    # 抓取基本面
    fundamentals = get_fundamentals(selected_stock)
    
    # --- 顯示基本面看板 ---
    st.subheader(f"📊 {selected_stock} 綜合戰情板 | 參考價：{last_price:.2f}")
    
    with st.expander("🏥 基本面體質診斷 (Fundamental Check)", expanded=True):
        cols = st.columns(6)  # 多一欄給 FCF Yield
        
        pe = fundamentals.get('pe_ratio')
        peg = fundamentals.get('peg_ratio')
        fcf = fundamentals.get('free_cashflow')
        pb = fundamentals.get('price_to_book')
        margin = fundamentals.get('profit_margins')
        fcf_yield = fundamentals.get('fcf_yield')
        
        # 添加 None 檢查
        cols[0].metric("PE (本益比)", f"{pe:.1f}x" if pe is not None else "N/A", help="高於 20 通常視為成長股")
        
        # PEG 顏色邏輯：越低越好 (0.5~1.0 最佳)
        peg_display = f"{peg:.2f}" if peg is not None else "N/A"
        if peg is not None:
            peg_delta = "低估 (Buy)" if peg < 1.0 else ("偏貴" if peg > 2.0 else "合理")
            peg_color = "normal" if peg < 1.0 else ("inverse" if peg > 2.0 else "off")
        else:
            peg_delta = None
            peg_color = "off"
        cols[1].metric("PEG (本益成長比)", peg_display, peg_delta, delta_color=peg_color, help="考量成長性。若 PE 低但 PEG 高，小心陷阱！")
        
        # 現金流顯示
        fcf_display = f"{fcf/1e9:.1f}B" if fcf is not None else "N/A"
        cols[2].metric("自由現金流", fcf_display, help="公司的真實造血能力")
        
        cols[3].metric("PB (股價淨值比)", f"{pb:.1f}x" if pb is not None else "N/A")
        cols[4].metric("淨利率", f"{margin*100:.1f}%" if margin is not None else "N/A")
        
        # 新增 FCF Yield
        fcf_yield_display = f"{fcf_yield*100:.1f}%" if fcf_yield is not None else "N/A"
        cols[5].metric("FCF Yield", fcf_yield_display, help="自由現金流收益率，高於6% 通常吸引人")
    
    if st.button("🚀 啟動 2026 雙引擎量化決策", type="primary"):
        if not pplx_api_key or not google_api_key:
            st.error("❌ 請提供 API Keys。")
        else:
            ai_decision = None
            paths = None
            
            with st.status("正在連線 2026 數據中心...", expanded=True) as status:
                st.write("🌐 Perplexity: 正在掃描全球情報...")
                intel = get_perplexity_intel(selected_stock, pplx_api_key)
                
                st.write("🧠 Gemini (Professor Mode): 正在分析基本面與市場情緒...")
                # 將基本面數據傳入 AI
                ai_decision = get_gemini_decision(selected_stock, google_api_key, intel, fundamentals)
                
                # 後處理防呆：如果 BUY 但 PEG > 2.5 或 FCF < 0，強制 HOLD
                peg = fundamentals.get('peg_ratio')
                fcf = fundamentals.get('free_cashflow')
                if ai_decision and ai_decision['verdict'] == "BUY":
                    if peg is not None and peg > 2.5:
                        ai_decision['verdict'] = "HOLD"
                        ai_decision['reasoning'] += "\n[後處理修正] PEG 過高，強制降級為 HOLD。"
                    if fcf is not None and fcf < 0:
                        ai_decision['verdict'] = "HOLD"
                        ai_decision['reasoning'] += "\n[後處理修正] 自由現金流為負，強制降級為 HOLD。"
                
                if ai_decision:
                    ai_return = ai_decision.get("expected_return", 0.05)
                    ai_vol = ai_decision.get("volatility_multiplier", 1.0)
                    verdict = ai_decision.get("verdict", "HOLD")
                    reasoning = ai_decision.get("reasoning", "No data")
                    model_used = ai_decision.get("model_used", "Unknown")
                    
                    st.write(f"📊 Monte Carlo: 使用 {model_used} 執行模擬...")
                    # 背景執行模擬以優化性能
                    def run_sim():
                        global paths
                        paths = run_monte_carlo(data, days_to_predict, simulations, ai_return, ai_vol)
                    
                    thread = threading.Thread(target=run_sim)
                    thread.start()
                    thread.join()  # 等待完成
                    
                    status.update(label="分析完成！", state="complete", expanded=False)
                else:
                    status.update(label="AI 決策引擎回應失敗", state="error")
                    st.warning("使用 fallback 值繼續顯示。")
            
            # --- 結果展現區 ---
            if ai_decision and paths is not None:
                col_chart, col_report = st.columns([2, 1])
                
                with col_chart:
                    st.subheader("🔮 預測路徑模擬")
                    # 使用交易日
                    start_date = data.index[-1] + timedelta(days=1)
                    future_dates = get_trading_days(start_date, days_to_predict)
                    if len(future_dates) < days_to_predict:
                        st.warning("交易日不足，調整為可用日期。")
                    
                    p5, p50, p95 = np.percentile(paths, [5, 50, 95], axis=1)
                    
                    # 歷史數據長度檢查
                    hist_len = min(90, len(data))
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data.index[-hist_len:], y=data['Close'][-hist_len:], name='歷史', line=dict(color='gray')))
                    fig.add_trace(go.Scatter(x=future_dates, y=p95, mode='lines', line=dict(width=0), showlegend=False))
                    fig.add_trace(go.Scatter(x=future_dates, y=p5, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 100, 255, 0.1)', name='90% 區間'))
                    fig.add_trace(go.Scatter(x=future_dates, y=p50, mode='lines', name='AI 預測中位數', line=dict(color='#0064ff', width=3)))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("悲觀 (P5)", f"{p5[-1]:.2f}", delta=f"{(p5[-1]/last_price-1)*100:.1f}%", delta_color="inverse")
                    m2.metric("中位 (P50)", f"{p50[-1]:.2f}", delta=f"{(p50[-1]/last_price-1)*100:.1f}%")
                    m3.metric("樂觀 (P95)", f"{p95[-1]:.2f}", delta=f"{(p95[-1]/last_price-1)*100:.1f}%")
                with col_report:
                    st.subheader("📝 教授決策報告")
                    
                    # 計算盈虧比，避免負值或零
                    upside = max(p95[-1] - last_price, 0.01)
                    downside = max(last_price - p5[-1], 0.01)
                    r_r_ratio = upside / downside
                    
                    # 調整 buy_percent 邏輯，更一致
                    if verdict == "BUY":
                        buy_percent = 80 if r_r_ratio > 1.5 else 60 if r_r_ratio > 1.0 else 50
                    elif verdict == "SELL":
                        buy_percent = 0
                    else:
                        buy_percent = 40
                    
                    st.progress(buy_percent / 100)
                    st.markdown(f"**建議倉位: {buy_percent}% (R/R: {r_r_ratio:.1f})**")
                    
                    # 顯示 AI 邏輯
                    st.info(f"**分析邏輯 ({model_used}):**\n\n{reasoning}")
                    st.caption("AI 已考量 PEG 與現金流數據進行修正。")
                    
                    with st.expander("情報原文"):
                        st.write(intel)

