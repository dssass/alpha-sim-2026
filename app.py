import streamlit as st
from datetime import date, timedelta
import yfinance as yf
from plotly import graph_objs as go
import pandas as pd
import numpy as np
from openai import OpenAI
import google.generativeai as genai
import json
import re
import os # <-- 新增: 用於檢查檔案是否存在

# --- 0. 金鑰儲存功能 (新功能) ---
KEYS_FILE = "api_keys.json"

def save_api_keys(pplx_key, google_key):
    """將 API Keys 儲存到本地 JSON 檔案"""
    keys = {"perplexity_api_key": pplx_key, "google_api_key": google_key}
    try:
        with open(KEYS_FILE, "w") as f:
            json.dump(keys, f)
        return True
    except Exception as e:
        st.error(f"儲存金鑰失敗: {e}")
        return False

def load_api_keys():
    """從本地 JSON 檔案載入 API Keys"""
    if os.path.exists(KEYS_FILE):
        try:
            with open(KEYS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # 如果檔案是空的或格式錯誤，回傳空字典
            return {}
    return {}


# --- 1. 頁面設定 ---
st.set_page_config(page_title="Alpha-Sim: 2026 AI 戰情室", layout="wide", page_icon="🛡️")

# 自定義 CSS
st.markdown("""
    <style>
    .stButton>button {width: 100%; border-radius: 8px; height: 3.5em; font-weight: bold; font-size: 1.2em;}
    .report-box {background-color: #f0f2f6; border-left: 6px solid #ff4b4b; padding: 20px; border-radius: 10px; margin-bottom: 20px;}
    .metric-container {background-color: #ffffff; padding: 10px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);}
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Alpha-Sim: 雙引擎量化決策系統 (2026 版)")
st.caption("架構：Perplexity (情報搜集) ➡ Gemini (參數決策) ➡ Monte Carlo (風險模擬)")

# --- 2. 側邊欄：設定 ---
with st.sidebar:
    st.header("🔑 啟動金鑰")
    
    # --- 金鑰管理 UI (已更新) ---
    saved_keys = load_api_keys()
    
    pplx_api_key = st.text_input(
        "Perplexity API Key", 
        type="password",
        value=saved_keys.get("perplexity_api_key", "") # 自動填入已儲存的 key
    )
    google_api_key = st.text_input(
        "Google AI Studio Key", 
        type="password",
        value=saved_keys.get("google_api_key", "") # 自動填入已儲存的 key
    )
    
    if st.button("儲存金鑰 (Save Keys)"):
        if pplx_api_key and google_api_key:
            if save_api_keys(pplx_api_key, google_api_key):
                st.success("金鑰已成功儲存！下次開啟會自動載入。")
        else:
            st.warning("請先輸入兩個 API Keys 再儲存。")
    
    st.divider()
    
    st.header("🎯 標的設定")
    selected_stock = st.text_input("股票代碼 (Ticker)", "4938.TW") 
    
    st.subheader("模擬參數")
    simulations = st.slider("模擬路徑數 (N)", 500, 3000, 1000)
    days_to_predict = st.slider("預測天數 (T)", 30, 365, 90)
    
    st.info("💡 **系統狀態**：\n已載入 2026 年模型清單 (Gemini 3.0/2.5)。\nAI 將自動判斷市場情緒並設定數學參數。")

# --- 3. 核心函數 ---

@st.cache_data(ttl=3600)
def get_market_data(ticker):
    """抓取歷史股價數據"""
    try:
        data = yf.download(ticker, period="2y", interval="1d", auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if data.empty: return None
        data = data.ffill()
        return data
    except Exception as e:
        return None

def get_perplexity_intel(ticker, api_key):
    """引擎 A: 搜尋最新情報 (使用 Sonar-Pro)"""
    if not api_key: return "Perplexity API Key 未提供。"
    
    client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
    today_str = date.today().strftime("%Y-%m-%d")
    
    prompt = f"""
    Search for the latest financial news, institutional reports, and supply chain rumors for {ticker} as of today ({today_str}).
    Focus on:
    1. Revenue/Earnings outlook for 2026.
    2. Key themes (e.g., AI Server, iPhone 17, Tesla).
    3. Institutional sentiment (Foreign investors buying/selling?).
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
        return f"Perplexity Error: {str(e)}"

def get_gemini_decision(ticker, api_key, intel_text):
    """引擎 B: 決策與參數設定 (Agentic) - 2026 旗艦版"""
    if not api_key: return {
        "verdict": "NEUTRAL",
        "reasoning": "Google AI Studio Key 未提供。",
        "expected_return": 0.05,
        "volatility_multiplier": 1.0,
        "model_used": "None"
    }
    
    genai.configure(api_key=api_key)
    
    prompt = f"""
    You are a Quantitative Portfolio Manager in year 2026. 
    Target Stock: {ticker}
    Latest Intel:
    {intel_text}
    
    Task:
    1. Analyze the sentiment based on 2026 market conditions.
    2. Determine the 'Expected Annual Return' (drift) for a Monte Carlo simulation.
       - Range: -0.30 (Crash) to +0.50 (To the moon).
       - 0.0 means neutral/sideways.
    3. Determine the 'Volatility Multiplier'.
       - 1.0 = Normal historical volatility.
       - >1.0 = High uncertainty/Risk.
       - <1.0 = Stable.
    
    Output strictly in JSON format:
    {{
        "verdict": "BUY" or "HOLD" or "SELL",
        "reasoning": "Short summary of why...",
        "expected_return": 0.15,
        "volatility_multiplier": 1.1
    }}
    """
    
    models = [
        "gemini-3-flash-preview",
        "gemini-2.5-flash-preview",
        "gemini-flash-latest",
        "gemini-2.0-flash"
    ]
    
    last_error = ""
    
    for model_name in models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt, 
                generation_config={"response_mime_type": "application/json"}
            )
            result = json.loads(response.text)
            result['model_used'] = model_name 
            return result
        except Exception as e:
            last_error = str(e)
            print(f"⚠️ {model_name} 失敗，切換下一順位... ({str(e)[:50]})")
            continue
            
    return {
        "verdict": "NEUTRAL",
        "reasoning": f"AI Analysis failed across all models. Last error: {last_error}",
        "expected_return": 0.05,
        "volatility_multiplier": 1.0,
        "model_used": "None"
    }

def run_monte_carlo(data, days, sims, expected_return, vol_mult):
    """執行數學模擬 (GBM 模型)"""
    log_returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
    hist_vol = log_returns.std() * (252 ** 0.5)
    adj_vol = hist_vol * vol_mult
    daily_vol = adj_vol / (252 ** 0.5)
    daily_drift = (expected_return / 252) - (0.5 * daily_vol ** 2)
    last_price = data['Close'].iloc[-1]
    Z = np.random.normal(0, 1, (days, sims))
    daily_returns = np.exp(daily_drift + daily_vol * Z)
    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = last_price
    
    for t in range(1, days):
        price_paths[t] = price_paths[t-1] * daily_returns[t]
        
    return price_paths

# --- 4. 主程式邏輯 ---

data = get_market_data(selected_stock)

if data is None:
    st.error(f"❌ 無法獲取 {selected_stock} 的數據。請確認代碼是否正確 (台股請加 .TW，如 4938.TW)")
else:
    last_price = data['Close'].iloc[-1]
    
    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("當前股價", f"{last_price:.2f}")
    col_info2.metric("資料日期", str(data.index[-1].date()))
    col_info3.markdown(f"**分析標的**: {selected_stock}")
    
    st.divider()

    if st.button("🚀 啟動雙引擎戰情室 (Start Analysis)", type="primary"):
        # 使用側邊欄輸入框中的最新值
        if not pplx_api_key or not google_api_key:
            st.error("❌ 請先在側邊欄輸入 API Keys 才能啟動大師級分析。")
        else:
            with st.status("正在進行深度運算...", expanded=True) as status:
                
                st.write("🌍 Perplexity (Sonar-Pro): 正在掃描全球財經新聞...")
                intel = get_perplexity_intel(selected_stock, pplx_api_key)
                
                st.write("🧠 Gemini (Auto-Switch): 正在解讀情報並設定模型參數...")
                ai_decision = get_gemini_decision(selected_stock, google_api_key, intel)
                
                ai_return = ai_decision.get("expected_return", 0.05)
                ai_vol = ai_decision.get("volatility_multiplier", 1.0)
                verdict = ai_decision.get("verdict", "HOLD")
                reasoning = ai_decision.get("reasoning", "No data")
                model_used = ai_decision.get("model_used", "Unknown")
                
                st.write(f"⚙️ 參數設定完成 (Model: {model_used}): 預期年化報酬 {ai_return*100:.1f}%, 波動加權 {ai_vol}x")
                
                st.write(f"📊 Monte Carlo: 正在執行 {simulations} 次平行宇宙模擬...")
                paths = run_monte_carlo(data, days_to_predict, simulations, ai_return, ai_vol)
                
                status.update(label="分析完成！", state="complete", expanded=False)

            # --- 5. 結果呈現 ---
            col_chart, col_report = st.columns([2, 1])
            
            with col_chart:
                st.subheader("🔮 未來價格路徑模擬")
                future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=days_to_predict)
                p5 = np.percentile(paths, 5, axis=1)
                p50 = np.percentile(paths, 50, axis=1)
                p95 = np.percentile(paths, 95, axis=1)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index[-60:], y=data['Close'][-60:], name='歷史 (近60日)', line=dict(color='gray', width=1)))
                fig.add_trace(go.Scatter(x=future_dates, y=p95, mode='lines', line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=future_dates, y=p5, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 100, 255, 0.1)', name='90% 機率區間'))
                fig.add_trace(go.Scatter(x=future_dates, y=p50, mode='lines', name='AI 預測中位數', line=dict(color='#0064ff', width=2)))
                st.plotly_chart(fig, use_container_width=True)
                
                final_p5 = p5[-1]
                final_p95 = p95[-1]
                r_r_ratio = (final_p95 - last_price) / (last_price - final_p5) if last_price > final_p5 else 0
                
                m1, m2, m3 = st.columns(3)
                m1.metric("悲觀情境 (P5)", f"{final_p5:.2f}", delta=f"{(final_p5/last_price-1)*100:.1f}%", delta_color="inverse")
                m2.metric("樂觀情境 (P95)", f"{final_p95:.2f}", delta=f"{(final_p95/last_price-1)*100:.1f}%")
                m3.metric("盈虧比 (R/R)", f"{r_r_ratio:.2f}")

            with col_report:
                st.subheader("📝 CIO 決策報告")
                
                color_map = {"BUY": "#28a745", "SELL": "#dc3545", "HOLD": "#ffc107", "NEUTRAL": "gray"}
                v_color = color_map.get(verdict, "gray")
                
                st.markdown(f"""
                <div style="border: 2px solid {v_color}; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 20px; background-color: {v_color}10;">
                    <h2 style="color: {v_color}; margin:0;">{verdict}</h2>
                    <small>Strategy Engine: {model_used}</small>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"**AI 觀點摘要：**\n\n{reasoning}")
                st.markdown("---")
                st.markdown("**模型參數 (AI 設定)：**")
                st.markdown(f"- 預期趨勢 (Drift): **{ai_return*100:+.1f}%**")
                st.markdown(f"- 波動係數 (Vol): **{ai_vol}x**")
                
                with st.expander("查看 Perplexity 原始情報"):
                    st.write(intel)
    else:
        st.info("👈 請在側邊欄輸入 API Keys，點擊「儲存」後，再啟動分析！")
