
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

# --- 視覺優化 CSS ---
st.markdown("""
    <style>
    /* 全局按鈕樣式 */
    .stButton>button {
        width: 100%; 
        border-radius: 8px; 
        height: 3.5em; 
        font-weight: bold; 
        font-size: 1.2em;
    }
    
    /* 數值字體放大 */
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem;
    }

    /* 教授報告專用卡片樣式 */
    .report-card {
        background-color: #1e293b; 
        border: 1px solid #334155;
        border-radius: 10px; 
        padding: 20px; 
        margin-bottom: 20px;
        color: #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* 分析邏輯藍色區塊 */
    .analysis-box {
        background-color: #172554;
        border-left: 4px solid #3b82f6;
        padding: 15px;
        margin-top: 15px;
        border-radius: 6px;
        font-size: 0.95em;
        color: #bfdbfe;
        line-height: 1.6;
    }
    
    /* 買賣建議標籤 */
    .verdict-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: bold;
        color: white;
        font-size: 0.9em;
        margin-bottom: 10px;
    }
    .verdict-buy { background-color: #10b981; }
    .verdict-sell { background-color: #ef4444; }
    .verdict-hold { background-color: #f59e0b; }
    
    /* 進度條容器 */
    .progress-container {
        background-color: #334155; 
        border-radius: 5px; 
        height: 10px; 
        width: 100%;
        margin-top: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Alpha-Sim: 2026 旗艦戰情室")

# --- 2. 核心邏輯函數 ---
def clean_json_string(json_str):
    """移除 Markdown 標記，防止 JSON 解析失敗"""
    try:
        cleaned = re.sub(r"```json\s*", "", json_str, flags=re.IGNORECASE)
        cleaned = re.sub(r"```", "", cleaned)
        return cleaned.strip()
    except:
        return json_str

@st.cache_data(ttl=3600)
def get_market_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="2y")
        if data is None or data.empty:
            data = yf.download(ticker, period="2y", interval="1d", progress=False)
        if data is None or data.empty:
            return None
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if 'Close' not in data.columns:
            if 'Adj Close' in data.columns:
                data['Close'] = data['Adj Close']
            else:
                return None
        return data.ffill()
    except Exception as e:
        st.error(f"抓取資料失敗: {e}")
        return None

@st.cache_data(ttl=21600)
def get_fundamentals_finmind(ticker, token):
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
    yf_fund = get_fundamentals_yfinance(ticker)
    for k, v in yf_fund.items():
        if k not in fundamentals or fundamentals[k] is None:
            fundamentals[k] = v
    return fundamentals

@st.cache_data(ttl=3600)
def get_fundamentals_yfinance(ticker):
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
    if not api_key:
        return "未提供 Perplexity API Key"
    client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
    today_str = date.today().strftime("%Y-%m-%d")
    prompt = f"Search for latest financial news and 2026 outlook for {ticker} as of {today_str}. Keep concise."
    try:
        response = client.chat.completions.create(
            model="sonar-pro",
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Perplexity Error: {str(e)}"

def get_gemini_decision(ticker, api_key, intel_text, fundamentals):
    if not api_key:
        return None
    genai.configure(api_key=api_key)
    fund_str = json.dumps(fundamentals, indent=2, ensure_ascii=False)
    prompt = f"""
    Act as a Hedge Fund Manager in 2026. Target: {ticker}
    [Intel]: {intel_text}
    [Fundamentals]: {fund_str}
    Rules: PEG > 2.0 is risky. Negative FCF is bad.
    Output JSON ONLY:
    {{
        "verdict": "BUY/SELL/HOLD",
        "reasoning": "Concise analysis in Traditional Chinese...",
        "expected_return": 0.15, 
        "volatility_multiplier": 1.1
    }}
    """
    # 2026 年常用模型（依實際情況調整，最前面的優先）
    models = ["gemini-2.0-flash-exp", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-flash"]
    for model_name in models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            result = json.loads(clean_json_string(response.text))
            result['model_used'] = model_name
            return result
        except Exception as e:
            continue
    return {
        "verdict": "HOLD",
        "reasoning": "AI 模型呼叫失敗，預設持倉",
        "expected_return": 0.05,
        "volatility_multiplier": 1.0,
        "model_used": "Fallback"
    }

def run_bootstrap_monte_carlo(data, days, sims, expected_return, vol_mult=1.0, block_size=10):
    log_returns = np.log(data['Close'] / data['Close'].shift(1)).dropna().values
    if len(log_returns) < block_size:
        return None
    n_returns = len(log_returns)
    num_blocks = days // block_size + 1
    last_price = data['Close'].iloc[-1]
    paths = np.full((days, sims), last_price)
    daily_drift = (expected_return / 252) * vol_mult
    for sim in range(sims):
        path = [last_price]
        for _ in range(num_blocks):
            start_idx = np.random.randint(0, n_returns - block_size)
            block = log_returns[start_idx:start_idx + block_size]
            block_adj = block + daily_drift
            path.extend(path[-1] * np.exp(np.cumsum(block_adj)))
        paths[:, sim] = path[1:days+1]
    return paths

def calculate_risk_metrics(paths, last_price, confidence=0.95):
    final_prices = paths[-1, :]
    returns = (final_prices - last_price) / last_price
    sorted_returns = np.sort(returns)
    var_idx = int((1 - confidence) * len(sorted_returns))
    var = -sorted_returns[var_idx]
    cvar = -np.mean(sorted_returns[:var_idx + 1])
    return var, cvar

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

if data is None or data.empty:
    st.error(f"❌ 找不到 {selected_stock} 的資料，請確認代碼或網路連線")
else:
    last_price = data['Close'].iloc[-1]
    fundamentals = get_fundamentals_finmind(selected_stock, st.session_state['finmind_token'])
    
    st.subheader(f"📊 {selected_stock} 戰情室 | 最新股價: {last_price:.2f} TWD")
    
    with st.expander("🏥 基本面體檢", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        pe = fundamentals.get('pe_ratio')
        peg = fundamentals.get('peg_ratio')
        fcf = fundamentals.get('free_cashflow')
        pb = fundamentals.get('price_to_book')
        margin = fundamentals.get('profit_margins')
        
        c1.metric("PE (本益比)", f"{pe:.1f}x" if pe else "N/A")
        peg_val = f"{peg:.2f}" if peg else "N/A"
        peg_color = "normal" if peg and peg < 1.0 else ("inverse" if peg and peg > 2.0 else "off")
        c2.metric("PEG (成長比)", peg_val, delta_color=peg_color)
        c3.metric("自由現金流", f"{fcf/1e9:.1f}B" if fcf else "N/A")
        c4.metric("PB (淨值比)", f"{pb:.1f}x" if pb else "N/A")
        c5.metric("淨利率", f"{margin*100:.1f}%" if margin else "N/A")

    if st.button("🚀 啟動雙引擎分析", type="primary"):
        if not pplx_api_key or not google_api_key:
            st.error("❌ 請先輸入 Perplexity 與 Google AI Studio 的 API Key")
        else:
            with st.status("AI 雙引擎運算中...", expanded=True) as status:
                st.write("🌍 Perplexity 搜集中...")
                intel = get_perplexity_intel(selected_stock, pplx_api_key)
                
                st.write("🧠 Gemini 決策中...")
                decision = get_gemini_decision(selected_stock, google_api_key, intel, fundamentals)
                
                ai_return = decision.get("expected_return", 0.05)
                ai_vol = decision.get("volatility_multiplier", 1.0)
                verdict = decision.get("verdict", "HOLD")
                model_used = decision.get("model_used", "Unknown")
                
                st.write("📊 Bootstrap Monte Carlo 模擬中...")
                paths = run_bootstrap_monte_carlo(data, days_to_predict, simulations, ai_return, ai_vol)
                
                status.update(label="分析完成！", state="complete", expanded=False)

            # === 結果顯示區（已修正縮進與變數順序）===
            if paths is not None:
                # 計算百分位數（確保兩個欄位都能讀到）
                p5 = np.percentile(paths, 5, axis=1)
                p50 = np.percentile(paths, 50, axis=1)
                p95 = np.percentile(paths, 95, axis=1)
                
                col_chart, col_report = st.columns([2, 1])
                
                # 左側：價格路徑圖
                with col_chart:
                    st.subheader("🔮 未來價格路徑模擬")
                    future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=days_to_predict)
                    
                    fig = go.Figure()
                    hist_len = min(60, len(data))
                    fig.add_trace(go.Scatter(x=data.index[-hist_len:], y=data['Close'][-hist_len:], name='歷史股價', line=dict(color='gray')))
                    fig.add_trace(go.Scatter(x=future_dates, y=p95, mode='lines', line=dict(width=0), showlegend=False))
                    fig.add_trace(go.Scatter(x=future_dates, y=p5, mode='lines', line=dict(width=0), fill='tonexty', 
                                          fillcolor='rgba(0, 100, 255, 0.2)', name='90% 信賴區間'))
                    fig.add_trace(go.Scatter(x=future_dates, y=p50, mode='lines', name='AI 中位數路徑', line=dict(color='#3b82f6', width=3)))
                    fig.update_layout(height=420, margin=dict(l=0, r=0, t=30, b=0), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("悲觀 (P5)", f"{p5[-1]:.2f}", f"{(p5[-1]/last_price-1)*100:+.1f}%", delta_color="inverse")
                    m2.metric("中位 (P50)", f"{p50[-1]:.2f}", f"{(p50[-1]/last_price-1)*100:+.1f}%")
                    m3.metric("樂觀 (P95)", f"{p95[-1]:.2f}", f"{(p95[-1]/last_price-1)*100:+.1f}%")
                    
                    st.subheader("風險指標")
                    var_95, cvar_95 = calculate_risk_metrics(paths, last_price)
                    r1, r2 = st.columns(2)
                    r1.metric("VaR 95%", f"-{var_95*100:.1f}%", delta_color="inverse")
                    r2.metric("CVaR 95%", f"-{cvar_95*100:.1f}%", delta_color="inverse")

                # 右側：教授報告卡片（已確保變數存在，HTML 正確渲染）
                with col_report:
                    st.subheader("📝 教授決策報告")
                    
                    verdict = decision.get("verdict", "HOLD").upper()
                    reasoning = decision.get("reasoning", "AI 未提供詳細分析")
                    model_used = decision.get("model_used", "Unknown")
                    
                    upside = max(p95[-1] - last_price, 0.01)
                    downside = max(last_price - p5[-1], 0.01)
                    rr_ratio = upside / downside
                    
                    if "BUY" in verdict:
                        percent = 80 if rr_ratio > 1.5 else 60
                        badge_class = "verdict-buy"
                        badge_color = "#10b981"
                    elif "SELL" in verdict:
                        percent = 0
                        badge_class = "verdict-sell"
                        badge_color = "#ef4444"
                    else:
                        percent = 40
                        badge_class = "verdict-hold"
                        badge_color = "#f59e0b"
                    
                    safe_reasoning = reasoning.replace("{", "(").replace("}", ")").replace("\n", "<br>")
                    
                    html_code = f"""
                    <div class="report-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span class="verdict-badge {badge_class}">{verdict}</span>
                            <span style="font-size: 0.8em; color: #94a3b8; font-family: monospace;">{model_used}</span>
                        </div>
                        
                        <div style="margin-top: 15px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                <span style="font-weight: bold; font-size: 1.1em;">建議倉位: {percent}%</span>
                                <span style="font-size: 0.9em; color: #94a3b8;">R/R: {rr_ratio:.1f}x</span>
                            </div>
                            <div class="progress-container">
                                <div style="background-color: #3b82f6; width: {percent}%; height: 100%; border-radius: 5px;"></div>
                            </div>
                        </div>
                        
                        <div class="analysis-box">
                            <strong style="color: #60a5fa;">分析邏輯:</strong><br>
                            {safe_reasoning}
                        </div>
                        
                        <div style="margin-top: 15px; font-size: 0.8em; color: #94a3b8; border-top: 1px solid #334155; padding-top: 10px;">
                            AI 已考量 PEG、現金流、市場情報進行決策<br>
                            預期年化報酬: <span style="color: #e2e8f0;">{ai_return*100:.1f}%</span> | 
                            波動倍數: <span style="color: #e2e8f0;">{ai_vol:.1f}x</span>
                        </div>
                    </div>
                    """
                    st.markdown(html_code, unsafe_allow_html=True)
                    
                    with st.expander("📄 原始情報來源 (Perplexity)"):
                        st.write(intel)
    else:
        st.info("👈 輸入完金鑰與標的後，點擊「啟動雙引擎分析」開始")
