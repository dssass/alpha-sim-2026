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
import re
import requests

# 設定 logging
logging.basicConfig(level=logging.INFO)

# --- 1. 頁面設定 ---
st.set_page_config(page_title="Alpha-Sim: 2026 AI 戰情室 (Ultimate)", layout="wide", page_icon="🛡️")

# 自定義 CSS
st.markdown("""
    <style>
    .stButton>button {width: 100%; border-radius: 8px; height: 3.5em; font-weight: bold; font-size: 1.2em;}
    div[data-testid="stMetricValue"] {font-size: 1.4rem;}
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Alpha-Sim: 2026 旗艦戰情室")

# --- 2. 核心邏輯函數 ---

def clean_json_string(json_str):
    """🛠️ 關鍵修復：移除 Markdown 標記，防止 JSON 解析失敗"""
    try:
        cleaned = re.sub(r"```json\s*", "", json_str, flags=re.IGNORECASE)
        cleaned = re.sub(r"```", "", cleaned)
        return cleaned.strip()
    except:
        return json_str

@st.cache_data(ttl=3600)
def get_market_data(ticker):
    """抓取歷史股價數據 (加強防呆版)"""
    try:
        # 1. 嘗試使用 Ticker.history (較穩定)
        stock = yf.Ticker(ticker)
        data = stock.history(period="2y")
        
        # 2. 如果空的，嘗試 yf.download
        if data is None or data.empty:
            data = yf.download(ticker, period="2y", interval="1d", progress=False)
            
        # 🚨 關鍵修復：如果還是空的，直接回傳 None
        if data is None or data.empty or len(data) == 0:
            return None
        
        # 3. 資料清洗
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
            
        if isinstance(data.columns, pd.MultiIndex):
            try:
                data.columns = data.columns.get_level_values(0)
            except:
                pass
            
        if 'Close' not in data.columns:
            if 'Adj Close' in data.columns:
                data['Close'] = data['Adj Close']
            else:
                return None 
                
        if len(data) == 0:
            return None
            
        return data.ffill()
    except Exception as e:
        logging.error(f"Market Data Error: {e}")
        return None

@st.cache_data(ttl=21600)
def get_fundamentals_finmind(ticker, token):
    """基本面數據抓取"""
    # 若無 Token 或非台股直接用 Yahoo
    if not token or ".TW" not in ticker:
        return get_fundamentals_yfinance(ticker)

    fundamentals = {}
    stock_id = ticker.replace(".TW", "")
    
    try:
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            "dataset": "TaiwanStockPER",
            "data_id": stock_id,
            "start_date": (date.today() - timedelta(days=60)).strftime("%Y-%m-%d"),
            "token": token
        }
        res = requests.get(url, params=params)
        if res.status_code == 200:
            df = pd.DataFrame(res.json().get('data', []))
            if not df.empty:
                latest = df.iloc[-1]
                fundamentals['pe_ratio'] = latest.get('PE')
                fundamentals['price_to_book'] = latest.get('PBR')
    except:
        pass
    
    # 融合 Yahoo 數據
    yf_fund = get_fundamentals_yfinance(ticker)
    for k, v in yf_fund.items():
        if k not in fundamentals or fundamentals[k] is None:
            fundamentals[k] = v
            
    return fundamentals

@st.cache_data(ttl=3600)
def get_fundamentals_yfinance(ticker):
    """Yahoo Finance 基本面"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "pe_ratio": info.get('trailingPE'),
            "forward_pe": info.get('forwardPE'),
            "peg_ratio": info.get('pegRatio'),
            "price_to_book": info.get('priceToBook'),
            "free_cashflow": info.get('freeCashflow'),
            "profit_margins": info.get('profitMargins'),
            "revenue_growth": info.get('revenueGrowth'),
            "fcf_yield": (info.get('freeCashflow', 0) / info.get('marketCap', 1)) if info.get('marketCap') else None
        }
    except:
        return {}

def get_perplexity_intel(ticker, api_key):
    """Perplexity 情報"""
    if not api_key: return "無 API Key，無法獲取情報。"
    
    client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
    today_str = date.today().strftime("%Y-%m-%d")
    
    prompt = f"""
    Search for the latest financial news and analyst reports for {ticker} as of {today_str}.
    Focus on: 2026 Revenue outlook, AI/Supply chain impact, and Institutional sentiment.
    Keep it concise.
    """
    try:
        response = client.chat.completions.create(
            model="sonar-pro",
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Perplexity Error: {str(e)}"

def get_gemini_decision(ticker, api_key, intel_text, fundamentals):
    """Gemini 決策"""
    if not api_key: return None
    
    genai.configure(api_key=api_key)
    fund_str = json.dumps(fundamentals, indent=2)
    
    prompt = f"""
    Act as a Hedge Fund Manager in 2026. Target: {ticker}
    
    [Intel]: {intel_text}
    [Fundamentals]: {fund_str}
    
    Rules:
    1. PEG > 2.0 is risky. PEG < 1.0 is good.
    2. Negative Free Cash Flow is a red flag.
    
    Output JSON ONLY:
    {{
        "verdict": "BUY/SELL/HOLD",
        "reasoning": "Concise analysis...",
        "expected_return": 0.15, 
        "volatility_multiplier": 1.1
    }}
    (expected_return: range -0.3 to 0.5, vol_multiplier: 0.8 to 2.0)
    """
    
    models = ["gemini-3-flash-preview", "gemini-2.5-flash-preview", "gemini-flash-latest", "gemini-2.0-flash"]
    
    for model_name in models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            
            cleaned_text = clean_json_string(response.text)
            result = json.loads(cleaned_text)
            result['model_used'] = model_name
            return result
        except:
            continue
            
    return {"verdict": "HOLD", "reasoning": "AI Failed", "expected_return": 0.05, "volatility_multiplier": 1.0, "model_used": "Fallback"}

def run_bootstrap_monte_carlo(data, days, sims, expected_return, vol_mult=1.0, block_size=10):
    """Bootstrap 模擬"""
    log_returns = np.log(data['Close'] / data['Close'].shift(1)).dropna().values
    if len(log_returns) < block_size: return None
    
    n_returns = len(log_returns)
    num_blocks = days // block_size + 1
    last_price = data['Close'].iloc[-1]
    paths = np.full((days, sims), last_price)
    
    daily_drift = (expected_return / 252) * vol_mult
    
    for sim in range(sims):
        path = [last_price]
        for _ in range(num_blocks):
            start_idx = np.random.randint(0, n_returns - block_size)
            block = log_returns[start_idx : start_idx + block_size]
            block_adj = block + daily_drift 
            path.extend(path[-1] * np.exp(np.cumsum(block_adj)))
        
        paths[:, sim] = path[1:days+1]
        
    return paths

# --- 3. Session State ---
if 'pplx_key' not in st.session_state: st.session_state['pplx_key'] = ''
if 'google_key' not in st.session_state: st.session_state['google_key'] = ''
if 'finmind_token' not in st.session_state: st.session_state['finmind_token'] = ''

# --- 4. 側邊欄 ---
with st.sidebar:
    st.header("🔑 金鑰管理")
    pplx_api_key = st.text_input("Perplexity API Key", type="password", value=st.session_state['pplx_key'])
    google_api_key = st.text_input("Google AI Studio Key", type="password", value=st.session_state['google_key'])
    finmind_token = st.text_input("FinMind Token (選填)", type="password", value=st.session_state['finmind_token'])
    
    if st.button("💾 暫存金鑰"):
        st.session_state['pplx_key'] = pplx_api_key
        st.session_state['google_key'] = google_api_key
        st.session_state['finmind_token'] = finmind_token
        st.success("已暫存")

    st.divider()
    st.header("🎯 標的設定")
    selected_stock = st.text_input("股票代碼", "2330.TW")
    simulations = st.slider("模擬次數", 500, 3000, 1000)
    days_to_predict = st.slider("預測天數", 30, 365, 90)

# --- 5. 主程式 ---
data = get_market_data(selected_stock)

# 🚨 防呆：檢查數據是否存在
if data is None or data.empty:
    st.error(f"❌ 找不到 {selected_stock} 的資料。")
    st.warning("解決方案：\n1. 請確認代碼 (如 2330.TW)。\n2. 可能是 Yahoo Finance 暫時阻擋雲端 IP，請稍後再試。\n3. 嘗試輸入美股 (如 NVDA) 測試。")
else:
    # 這裡現在安全了
    last_price = data['Close'].iloc[-1]
    
    # 抓基本面
    fundamentals = get_fundamentals_finmind(selected_stock, st.session_state['finmind_token'])
    
    # --- 基本面看板 ---
    st.subheader(f"📊 {selected_stock} 戰情室 | 股價: {last_price:.2f}")
    
    with st.expander("🏥 基本面體檢 (Fundamental Check)", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        
        pe = fundamentals.get('pe_ratio')
        peg = fundamentals.get('peg_ratio')
        fcf = fundamentals.get('free_cashflow')
        pb = fundamentals.get('price_to_book')
        margin = fundamentals.get('profit_margins')
        
        c1.metric("PE (本益比)", f"{pe:.1f}x" if pe else "N/A")
        
        # 🛠️ 這裡修復了語法錯誤
        peg_val = f"{peg:.2f}" if peg else "N/A"
        if peg:
            peg_color = "normal" if peg < 1.0 else ("inverse" if peg > 2.0 else "off")
        else:
            peg_color = "off"
            
        c2.metric("PEG (成長比)", peg_val, delta_color=peg_color)
        
        fcf_val = f"{fcf/1e9:.1f}B" if fcf else "N/A"
        c3.metric("自由現金流", fcf_val)
        c4.metric("PB (淨值比)", f"{pb:.1f}x" if pb else "N/A")
        c5.metric("淨利率", f"{margin*100:.1f}%" if margin else "N/A")

    # --- 啟動按鈕邏輯 ---
    if st.button("🚀 啟動雙引擎分析", type="primary"):
        if not pplx_api_key or not google_api_key:
            st.error("❌ 請輸入 API Keys")
        else:
            with st.status("正在進行 AI 運算...", expanded=True) as status:
                
                # Step 1: 搜集情報
                st.write("🌍 Perplexity: 搜集市場情報...")
                intel = get_perplexity_intel(selected_stock, pplx_api_key)
                
                # Step 2: AI 決策
                st.write("🧠 Gemini: 進行多空決策...")
                decision = get_gemini_decision(selected_stock, google_api_key, intel, fundamentals)
                
                # 取出參數
                ai_return = decision.get("expected_return", 0.05)
                ai_vol = decision.get("volatility_multiplier", 1.0)
                verdict = decision.get("verdict", "HOLD")
                model_used = decision.get("model_used", "Unknown")
                
                st.success(f"決策模型鎖定: {model_used}")
                
                # Step 3: 模擬
                st.write(f"📊 Bootstrap Monte Carlo: 執行 {simulations} 次模擬...")
                paths = run_bootstrap_monte_carlo(data, days_to_predict, simulations, ai_return, ai_vol)
                
                status.update(label="分析完成！", state="complete", expanded=False)

            # --- 結果呈現區 ---
            if paths is not None:
                col_chart, col_report = st.columns([2, 1])
                
                with col_chart:
                    st.subheader("🔮 價格路徑模擬")
                    future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=days_to_predict)
                    
                    p5 = np.percentile(paths, 5, axis=1)
                    p50 = np.percentile(paths, 50, axis=1)
                    p95 = np.percentile(paths, 95, axis=1)
                    
                    fig = go.Figure()
                    # 顯示近 60 天歷史
                    hist_len = min(60, len(data))
                    fig.add_trace(go.Scatter(x=data.index[-hist_len:], y=data['Close'][-hist_len:], name='歷史', line=dict(color='gray')))
                    
                    # 預測區間
                    fig.add_trace(go.Scatter(x=future_dates, y=p95, mode='lines', line=dict(width=0), showlegend=False))
                    fig.add_trace(go.Scatter(x=future_dates, y=p5, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 100, 255, 0.1)', name='90% 區間'))
                    fig.add_trace(go.Scatter(x=future_dates, y=p50, mode='lines', name='預測中位數', line=dict(color='#0064ff', width=2)))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                with col_report:
                    st.subheader("📝 CIO 決策報告")
                    color_map = {"BUY": "#28a745", "SELL": "#dc3545", "HOLD": "#ffc107", "NEUTRAL": "gray"}
                    v_color = color_map.get(verdict, "gray")
                    
                    st.markdown(f"""
                    <div style="border: 2px solid {v_color}; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px; background-color: {v_color}10;">
                        <h2 style="color: {v_color}; margin:0;">{verdict}</h2>
                        <small>Model: {model_used}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"**AI 分析摘要：**\n\n{decision.get('reasoning')}")
                    st.markdown("---")
                    st.markdown(f"- 預期年化報酬: **{ai_return*100:+.1f}%**")
                    st.markdown(f"- 波動加權係數: **{ai_vol}x**")
                    
                    with st.expander("查看原始情報"):
                        st.write(intel)
    else:
        st.info("👈 準備就緒，請點擊按鈕啟動分析。")
