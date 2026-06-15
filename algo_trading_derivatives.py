import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
import random

st.set_page_config(
    page_title="Algo Trading with Derivatives Lab",
    page_icon="⚡",
    layout="wide"
)

def pct(x,d=2): return f"{round(x,d)}%"
def cr(x): return f"₹{x:,.2f}"

# ── BSM & Greeks ──────────────────────────────────────────
def bsm(S,K,T,r,sigma,opt="call"):
    if T<=0 or sigma<=0:
        return max(S-K,0) if opt=="call" else max(K-S,0)
    d1=(np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    if opt=="call":
        return S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)

def greeks(S,K,T,r,sigma):
    if T<=0 or sigma<=0:
        return {"dc":1.0,"dp":-1.0,"gamma":0,"vega":0,"tc":0,"tp":0}
    d1=(np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    dc=norm.cdf(d1); dp=dc-1
    gamma=norm.pdf(d1)/(S*sigma*np.sqrt(T))
    vega=S*norm.pdf(d1)*np.sqrt(T)/100
    tc=(-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T))-r*K*np.exp(-r*T)*norm.cdf(d2))/365
    tp=(-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T))+r*K*np.exp(-r*T)*norm.cdf(-d2))/365
    return {"dc":dc,"dp":dp,"gamma":gamma,"vega":vega,"tc":tc,"tp":tp}

@st.cache_data
def gen_path(n=252, start=22000, seed=42, trend=0.0002, vol=0.012):
    np.random.seed(seed)
    rets=np.random.normal(trend,vol,n)
    px=[start]
    for r in rets: px.append(px[-1]*(1+r))
    dates=pd.date_range(end=pd.Timestamp.today(),periods=n+1,freq='B')
    return pd.DataFrame({"Date":dates,"Close":px})

st.title("⚡ Algo Trading with Derivatives Lab")
st.markdown("""
How systematic strategies are built around **futures, options, and volatility** —
delta hedging, volatility arbitrage, basis trading, theta harvesting, and
automated Greeks-based risk management.

Covered: ✅ Delta Hedging ✅ Vol Arbitrage ✅ Futures Basis ✅ Options Theta Harvesting
✅ Calendar Spread Algos ✅ VIX-Based Switching ✅ Greeks Risk Engine ✅ Backtesting
""")

menu = st.sidebar.radio("Choose Module", [
    "Introduction — Why Derivatives + Algos",
    "Delta Hedging — Automated Rebalancing",
    "Gamma Scalping",
    "Volatility Arbitrage (IV vs Realised Vol)",
    "Futures-Cash Basis Trading",
    "Calendar Spread Algo",
    "Theta Harvesting (Short Premium Systematic)",
    "VIX-Based Strategy Switching",
    "Options Market Making",
    "Greeks Risk Dashboard",
    "Iron Condor Automation",
    "Risk Management for Derivatives Algos",
    "Backtesting a Vol-Selling Strategy",
    "Step-by-Step Solver",
    "Quiz Engine",
    "Formula Cheat Sheet",
    "Case Study — Systematic Short Straddle on Nifty",
])

# =========================================================
if menu == "Introduction — Why Derivatives + Algos":
    st.header("⚡ Why Combine Algo Trading with Derivatives?")
    st.markdown("""
## The Special Case of Derivatives Algos

Equity algo strategies (MA crossover, RSI) bet on DIRECTION. Derivatives algos
can bet on **direction, volatility, time decay, or relative pricing** —
often with a **market-neutral** profile.

## Why Derivatives Need Algorithms

| Reason | Explanation |
|---|---|
| **Greeks change constantly** | Delta, Gamma, Theta, Vega all move with spot, time, and IV — manual tracking is impossible at scale |
| **Continuous rebalancing** | Delta-hedged positions need rebalancing many times a day |
| **Speed of pricing** | Option prices must be recalculated as inputs change — milliseconds matter |
| **Multi-leg complexity** | Spreads, condors, straddles involve 2-4 legs that must execute together |
| **Volatility is the asset** | Many derivatives strategies trade VOLATILITY itself, not price direction |

## Categories of Derivatives Algo Strategies

| Category | Core Idea | Market View |
|---|---|---|
| **Delta Hedging** | Continuously neutralise directional exposure | Market-neutral |
| **Gamma Scalping** | Profit from re-hedging as price moves | Long volatility |
| **Volatility Arbitrage** | Trade gap between Implied Vol and Realised Vol | Vol mean-reversion |
| **Theta Harvesting** | Systematically sell options, collect time decay | Short volatility |
| **Futures Basis Trading** | Exploit futures vs spot mispricing | Market-neutral |
| **Calendar Spreads** | Exploit term-structure of volatility | Vol term structure |
| **Options Market Making** | Continuously quote option bid/ask | Spread capture |

## The Foundation: Greeks Drive Everything

$$\\Delta P \\approx \\Delta \\times \\Delta S + \\frac{1}{2}\\Gamma \\times (\\Delta S)^2 + \\Theta \\times \\Delta t + \\nu \\times \\Delta \\sigma$$

Every derivatives algo is, at its core, a system for managing this equation
in real time.
""")
    col1,col2 = st.columns(2)
    col1.success("""
**Where this is used in India:**
- Proprietary trading desks (Nifty/BankNifty options)
- Market makers on NSE option chains
- Quant funds running vol-selling strategies
- Retail platforms: Sensibull, Streak, Tradetron (semi-automated)
""")
    col2.warning("""
**Prerequisites:**
- Strong grasp of Black-Scholes & Greeks
- Real-time data feeds (tick-by-tick)
- Low-latency execution (for hedging)
- Robust risk limits — vol-selling can blow up fast
""")

# =========================================================
elif menu == "Delta Hedging — Automated Rebalancing":
    st.header("🎯 Delta Hedging — Automated Rebalancing")
    st.markdown("""
## The Core Idea

A short option position has DELTA exposure. To stay market-neutral, an algo
continuously buys/sells the underlying (futures) to offset delta.

$$\\text{Hedge Position} = -\\Delta_{\\text{option}} \\times \\text{Option Qty}$$

**Re-hedge trigger:** when |Δ_current - Δ_hedged| > threshold (e.g., 0.05)
""")

    col1,col2 = st.columns(2)
    with col1:
        S0=st.number_input("Nifty Spot",value=22000.0)
        K=st.number_input("Strike (Short Call)",value=22000.0)
        T_days=st.number_input("Days to Expiry",value=15)
        sigma=st.number_input("IV %",value=18.0)/100
        r=0.07
        n_contracts=st.number_input("Lots Short",value=4)
        threshold=st.slider("Re-hedge Delta Threshold",0.01,0.20,0.05,0.01)

    T=T_days/365
    g0=greeks(S0,K,T,r,sigma)
    delta0=g0["dc"]
    hedge_units0=delta0*n_contracts*25

    with col2:
        st.metric("Option Delta (Call)",round(delta0,4))
        st.metric("Total Option Delta Exposure",round(delta0*n_contracts*25,1))
        st.metric("Futures to BUY (hedge)",f"{hedge_units0:.1f} units ≈ {hedge_units0/25:.2f} lots")

    st.subheader("Simulating Spot Moves & Re-Hedging")
    spot_moves=st.slider("Number of price steps to simulate",5,30,15)
    np.random.seed(1)
    spots=[S0]
    for _ in range(spot_moves):
        spots.append(spots[-1]*(1+np.random.normal(0,0.004)))

    hedge_log=[]
    current_hedge=hedge_units0
    total_rehedges=0
    cum_pnl=0
    for i in range(1,len(spots)):
        T_rem=max(T-(i/spot_moves)*T,0.001)
        g=greeks(spots[i],K,T_rem,r,sigma)
        new_delta_exposure=g["dc"]*n_contracts*25
        diff=new_delta_exposure-current_hedge
        # PnL from futures hedge held
        futures_pnl=(spots[i]-spots[i-1])*current_hedge
        cum_pnl+=futures_pnl
        rehedge="No"
        if abs(diff)/25 > threshold*n_contracts:
            current_hedge=new_delta_exposure
            total_rehedges+=1
            rehedge="YES"
        hedge_log.append({
            "Step":i,"Spot":round(spots[i],2),"Option Delta":round(g["dc"],4),
            "Target Hedge (units)":round(new_delta_exposure,1),
            "Current Hedge (units)":round(current_hedge,1),
            "Re-hedge?":rehedge,"Futures P&L (₹)":round(futures_pnl,1)
        })

    df_hedge=pd.DataFrame(hedge_log)
    st.dataframe(df_hedge,use_container_width=True)

    col1,col2,col3=st.columns(3)
    col1.metric("Total Re-hedges",total_rehedges)
    col2.metric("Cumulative Hedge P&L",cr(cum_pnl))
    col3.metric("Final Spot",f"{spots[-1]:,.2f}")

    fig=go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(spots))),y=spots,name='Spot',line=dict(color='#174EA6',width=2)))
    fig.update_layout(title="Simulated Spot Path",height=300)
    st.plotly_chart(fig,use_container_width=True)

    st.info("""
**Trade-off:** Tighter threshold = more accurate hedge but MORE transaction
costs from frequent re-hedging. Wider threshold = lower costs but more
"hedge slippage" — the algo must balance hedging frequency vs cost.
""")

# =========================================================
elif menu == "Gamma Scalping":
    st.header("🌀 Gamma Scalping")
    st.markdown("""
## The Concept

A LONG options position (e.g., long straddle) has POSITIVE GAMMA — delta
changes favourably as spot moves. By delta-hedging repeatedly, the algo
"scalps" profits from price oscillation, while the option position itself
decays (theta cost).

**Profitable if:** Realised volatility > Implied volatility (the cost of theta)

$$\\text{Gamma P&L (per rehedge)} \\approx \\frac{1}{2}\\Gamma \\times (\\Delta S)^2$$
$$\\text{Theta Cost} = \\Theta \\times \\Delta t$$

**Net edge** = Sum of Gamma P&L − Theta decay over the holding period
""")

    col1,col2 = st.columns(2)
    with col1:
        S0=st.number_input("Spot",value=22000.0,key="gs_s")
        K=st.number_input("Strike (ATM)",value=22000.0,key="gs_k")
        T_days=st.number_input("Days held",value=5,key="gs_t")
        iv=st.number_input("Implied Vol %",value=15.0,key="gs_iv")/100
        realised_vol=st.number_input("Realised Vol % (actual)",value=20.0,key="gs_rv")/100
        n_lots=st.number_input("Lots (Long Straddle)",value=2,key="gs_n")

    T=T_days/365
    r=0.07
    g=greeks(S0,K,T,r,iv)
    gamma_total=g["gamma"]*n_lots*25*2  # call+put both have gamma
    theta_total=(g["tc"]+g["tp"])*n_lots*25  # both negative (cost)

    # Expected daily move based on realised vol
    daily_move=S0*realised_vol/np.sqrt(252)
    expected_gamma_pnl_per_day=0.5*gamma_total*daily_move**2

    with col2:
        st.metric("Position Gamma (total)",round(gamma_total,4))
        st.metric("Position Theta (₹/day, cost)",cr(theta_total))
        st.metric("Expected daily move (1σ)",f"{daily_move:.1f} pts")
        st.metric("Expected Gamma P&L/day",cr(expected_gamma_pnl_per_day))

    net_daily=expected_gamma_pnl_per_day+theta_total
    if net_daily>0:
        st.success(f"✅ Net edge = ₹{net_daily:.1f}/day. Realised vol ({pct(realised_vol*100)}) > Implied vol ({pct(iv*100)}) — gamma scalping profitable!")
    else:
        st.error(f"❌ Net edge = ₹{net_daily:.1f}/day. Theta decay exceeds gamma P&L — long straddle losing money.")

    # Breakeven realised vol
    breakeven_move=np.sqrt(2*abs(theta_total)/gamma_total) if gamma_total>0 else 0
    breakeven_vol=breakeven_move/S0*np.sqrt(252)*100
    st.info(f"""
**Breakeven Realised Volatility:** {pct(breakeven_vol)}

If realised volatility EXCEEDS this level, the long straddle + delta hedging
(gamma scalping) is profitable on average. Below this, theta decay wins.
""")

    st.subheader("Vol Edge vs P&L")
    rv_range=np.arange(8,30,1)
    pnls=[]
    for rv in rv_range:
        dm=S0*(rv/100)/np.sqrt(252)
        gp=0.5*gamma_total*dm**2
        pnls.append(gp+theta_total)
    fig=go.Figure(go.Scatter(x=rv_range,y=pnls,mode='lines+markers',line=dict(color='#157A42',width=2)))
    fig.add_hline(y=0,line_color='red',line_dash='dash')
    fig.add_vline(x=iv*100,line_dash='dot',line_color='blue',annotation_text=f'IV={pct(iv*100)}')
    fig.update_layout(title="Daily Gamma Scalping P&L vs Realised Volatility",
                      xaxis_title="Realised Vol (%)",yaxis_title="Net Daily P&L (₹)",height=350)
    st.plotly_chart(fig,use_container_width=True)

# =========================================================
elif menu == "Volatility Arbitrage (IV vs Realised Vol)":
    st.header("📊 Volatility Arbitrage — IV vs Realised Vol")
    st.markdown("""
## The Core Trade

$$\\text{Vol Edge} = IV - RV_{\\text{forecast}}$$

- **IV > RV forecast** → SELL options (collect rich premium)
- **IV < RV forecast** → BUY options (cheap insurance)

Algo systems continuously forecast realised volatility (GARCH, historical
windows) and compare to the market's implied volatility, generating
systematic vol-selling or vol-buying signals.
""")

    col1,col2 = st.columns(2)
    with col1:
        seed_va=st.number_input("Market Seed",value=10,min_value=1)
        atm_iv_base=st.number_input("Base ATM IV %",value=16.0)
        lookback=st.slider("Realised Vol Lookback (days)",5,30,10)

    df=gen_path(n=252,seed=seed_va,trend=0.0001,vol=0.012)
    df['Returns']=df['Close'].pct_change()
    df['Realised_Vol']=df['Returns'].rolling(lookback).std()*np.sqrt(252)*100

    # Simulate IV as somewhat correlated but with persistent premium
    np.random.seed(seed_va+1)
    iv_noise=np.random.normal(0,1.5,len(df))
    df['Implied_Vol']=atm_iv_base+df['Realised_Vol'].rolling(5).mean().fillna(atm_iv_base)*0.3+iv_noise
    df['Implied_Vol']=df['Implied_Vol'].clip(lower=8)
    df['Vol_Edge']=df['Implied_Vol']-df['Realised_Vol']

    with col2:
        avg_edge=df['Vol_Edge'].mean()
        pct_iv_premium=(df['Vol_Edge']>0).mean()*100
        st.metric("Average IV - RV (Vol Risk Premium)",pct(avg_edge))
        st.metric("% of days IV > RV","")
        st.metric("% of days IV > RV (rerun)",pct(pct_iv_premium))

    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'],y=df['Implied_Vol'],name='Implied Vol',line=dict(color='#C03B3B',width=2)))
    fig.add_trace(go.Scatter(x=df['Date'],y=df['Realised_Vol'],name='Realised Vol',line=dict(color='#174EA6',width=2)))
    fig.update_layout(title="Implied Volatility vs Realised Volatility",yaxis_title="Volatility (%)",height=350)
    st.plotly_chart(fig,use_container_width=True)

    fig2=go.Figure(go.Scatter(x=df['Date'],y=df['Vol_Edge'],mode='lines',
                              line=dict(color='#F5A623',width=2),fill='tozeroy'))
    fig2.add_hline(y=0,line_color='black')
    fig2.update_layout(title="Vol Risk Premium (IV - RV) — Positive = Sell Vol Signal",height=300)
    st.plotly_chart(fig2,use_container_width=True)

    st.success("""
**The Volatility Risk Premium (VRP):** Historically, IV tends to be HIGHER
than subsequently realised vol most of the time — this is why systematic
option SELLING strategies have a structural edge on average. BUT: the
days when RV >> IV (vol spikes / crashes) can wipe out months of premium.
This is the classic "picking up pennies in front of a steamroller" risk.
""")

# =========================================================
elif menu == "Futures-Cash Basis Trading":
    st.header("📐 Futures-Cash Basis Trading (Cash-and-Carry Arbitrage)")
    st.markdown("""
## The Concept

$$\\text{Fair Futures Price} = S \\times e^{(r-d)T}$$
$$\\text{Basis} = F_{\\text{market}} - F_{\\text{fair}}$$

**If Basis > 0 (Futures overpriced):** SELL futures, BUY spot (cash-and-carry)
**If Basis < 0 (Futures underpriced):** BUY futures, SELL spot (reverse cash-and-carry)

This is a CLASSIC algo strategy — pure arbitrage, executed at high speed
before the mispricing disappears.
""")

    col1,col2 = st.columns(2)
    with col1:
        S=st.number_input("Nifty Spot",value=22000.0,key="basis_s")
        r=st.number_input("Risk-free Rate %",value=7.0,key="basis_r")/100
        div_yield=st.number_input("Dividend Yield %",value=1.2,key="basis_d")/100
        T_days=st.number_input("Days to Futures Expiry",value=20,key="basis_t")
        F_market=st.number_input("Current Futures Price",value=22080.0,key="basis_f")

    T=T_days/365
    F_fair=S*np.exp((r-div_yield)*T)
    basis=F_market-F_fair
    annualised_basis=(basis/S)*(365/T_days)*100

    with col2:
        st.metric("Fair Futures Price",f"{F_fair:,.2f}")
        st.metric("Market Futures Price",f"{F_market:,.2f}")
        st.metric("Basis (Market - Fair)",f"{basis:+,.2f}")
        st.metric("Annualised Basis",pct(annualised_basis))

    if basis>5:
        st.success(f"""
✅ **ARBITRAGE SIGNAL: SELL Futures + BUY Spot (Cash-and-Carry)**

Futures overpriced by {basis:.2f} points. Algo action:
1. SELL Nifty Futures @ {F_market:,.2f}
2. BUY Nifty basket (cash) @ {S:,.2f}
3. Hold until expiry — futures converges to spot
4. Locked profit ≈ {basis:.2f} points × 25 = ₹{basis*25:,.0f} per lot
""")
    elif basis<-5:
        st.error(f"""
✅ **ARBITRAGE SIGNAL: BUY Futures + SELL Spot (Reverse Cash-and-Carry)**

Futures underpriced by {abs(basis):.2f} points. Algo action:
1. BUY Nifty Futures @ {F_market:,.2f}
2. SHORT Nifty basket (cash, requires SLB - Securities Lending) @ {S:,.2f}
3. Locked profit ≈ {abs(basis):.2f} points × 25 = ₹{abs(basis)*25:,.0f} per lot

⚠️ Reverse arb is harder in India due to short-selling constraints (SLB costs).
""")
    else:
        st.info(f"No significant arbitrage — basis ({basis:+.2f}) within normal range.")

    st.subheader("Basis Behaviour Through the Month")
    days=np.arange(T_days,0,-1)
    np.random.seed(5)
    basis_path=basis*(days/T_days)+np.random.normal(0,1.5,len(days))
    fig=go.Figure(go.Scatter(x=T_days-days,y=basis_path,mode='lines+markers',line=dict(color='#6A0DAD',width=2)))
    fig.add_hline(y=0,line_color='black',line_dash='dash')
    fig.update_layout(title="Basis Converges to Zero at Expiry",xaxis_title="Days from Today",
                      yaxis_title="Basis (points)",height=300)
    st.plotly_chart(fig,use_container_width=True)

    st.warning("""
⚠️ **Execution speed matters:** Basis mispricings are usually small and
short-lived. HFT/prop desks capture these in milliseconds. By the time a
retail trader sees the opportunity, it's often gone — this is a textbook
case for ALGORITHMIC execution.
""")

# =========================================================
elif menu == "Calendar Spread Algo":
    st.header("📅 Calendar Spread Algo — Term Structure Trading")
    st.markdown("""
## The Concept

Options at different expiries have different IVs — the **volatility term
structure**. Algo systems monitor this structure and trade calendar spreads
when it's mispriced.

$$\\text{Term Structure Signal} = IV_{\\text{near}} - IV_{\\text{far}}$$

- **Steep backwardation** (near IV >> far IV): often near an event — SELL near, BUY far
- **Flat/inverted**: monitor for term structure normalisation trades
""")

    col1,col2 = st.columns(2)
    with col1:
        S=st.number_input("Spot",value=22000.0,key="cal_s")
        K=st.number_input("Strike (ATM)",value=22000.0,key="cal_k")
        T_near=st.number_input("Near expiry (days)",value=7,key="cal_tn")
        T_far=st.number_input("Far expiry (days)",value=30,key="cal_tf")
        iv_near=st.number_input("Near IV %",value=22.0,key="cal_ivn")/100
        iv_far=st.number_input("Far IV %",value=16.0,key="cal_ivf")/100

    r=0.07
    near_call=bsm(S,K,T_near/365,r,iv_near,"call")
    far_call=bsm(S,K,T_far/365,r,iv_far,"call")
    net_cost=far_call-near_call
    term_signal=iv_near-iv_far

    g_near=greeks(S,K,T_near/365,r,iv_near)
    g_far=greeks(S,K,T_far/365,r,iv_far)
    theta_near=g_near["tc"]
    theta_far=g_far["tc"]

    with col2:
        st.metric("Near Call Price (Sell)",cr(near_call))
        st.metric("Far Call Price (Buy)",cr(far_call))
        st.metric("Net Debit",cr(net_cost))
        st.metric("Term Structure Signal (Near-Far IV)",pct(term_signal*100))

    if term_signal>3:
        st.success(f"""
✅ **SIGNAL: Near IV elevated relative to Far IV (+{pct(term_signal*100)})**

Likely near-term event priced in. Algo action: SELL near-term {K:.0f} Call
@ ₹{near_call:.2f}, BUY far-term {K:.0f} Call @ ₹{far_call:.2f}.
Net cost: ₹{net_cost:.2f}/unit.

Profits if: Near-term IV collapses post-event (vol crush) while far-term
holds value.
""")
    else:
        st.info(f"Term structure signal ({pct(term_signal*100)}) within normal range — no strong calendar signal.")

    st.subheader("Theta Comparison")
    col1,col2,col3=st.columns(3)
    col1.metric("Near-term Theta/day",cr(theta_near*25))
    col2.metric("Far-term Theta/day",cr(theta_far*25))
    col3.metric("Net Theta Advantage",cr((abs(theta_near)-abs(theta_far))*25))
    st.info(f"Near-term option decays {cr(abs((theta_near-theta_far)*25))} faster per day than the far-term option — this is the calendar spread's daily edge if spot stays near {K:.0f}.")

# =========================================================
elif menu == "Theta Harvesting (Short Premium Systematic)":
    st.header("⏳ Theta Harvesting — Systematic Premium Selling")
    st.markdown("""
## The Strategy

Systematically SELL OTM options (e.g., weekly Nifty options 1-2% OTM) to
collect time decay (theta), with strict risk rules:

1. **Entry:** Sell OTM strangle/iron condor at fixed delta (e.g., 0.15-0.20 delta)
2. **Hold:** Collect theta daily
3. **Exit:** Close at profit target (e.g., 50% of premium) OR stop-loss (e.g., 2x premium)
4. **Risk control:** Hard stop on Greeks limits (delta, vega exposure)

$$\\text{Daily Theta Income} = |\\Theta_{\\text{call}}| + |\\Theta_{\\text{put}}|$$
""")

    col1,col2 = st.columns(2)
    with col1:
        S=st.number_input("Spot",value=22000.0,key="th_s")
        T_days=st.number_input("Days to Expiry",value=7,key="th_t")
        sigma=st.number_input("IV %",value=15.0,key="th_iv")/100
        delta_target=st.number_input("Target Delta for strikes",value=0.15,key="th_delta")
        n_lots=st.number_input("Lots",value=5,key="th_n")
        profit_target_pct=st.slider("Profit Target (% of premium)",30,80,50)
        stop_loss_mult=st.slider("Stop Loss (x premium)",1.5,3.0,2.0,0.5)

    T=T_days/365
    r=0.07

    # Find strikes for target delta (approximate)
    def find_strike_for_delta(S,T,r,sigma,target_delta,opt="call"):
        K=S
        for _ in range(50):
            g=greeks(S,K,T,r,sigma)
            d=g["dc"] if opt=="call" else abs(g["dp"])
            if abs(d-target_delta)<0.005: break
            if opt=="call":
                K += 10 if d>target_delta else -10
            else:
                K -= 10 if d>target_delta else -10
        return K

    K_call=find_strike_for_delta(S,T,r,sigma,delta_target,"call")
    K_put=find_strike_for_delta(S,T,r,sigma,delta_target,"put")
    call_premium=bsm(S,K_call,T,r,sigma,"call")
    put_premium=bsm(S,K_put,T,r,sigma,"put")
    total_premium=(call_premium+put_premium)*n_lots*25
    g_call=greeks(S,K_call,T,r,sigma)
    g_put=greeks(S,K_put,T,r,sigma)
    daily_theta=(abs(g_call["tc"])+abs(g_put["tp"]))*n_lots*25

    with col2:
        st.metric("Call Strike (sell)",f"{K_call:,.0f}")
        st.metric("Put Strike (sell)",f"{K_put:,.0f}")
        st.metric("Total Premium Collected",cr(total_premium))
        st.metric("Daily Theta Income",cr(daily_theta))

    profit_exit=total_premium*profit_target_pct/100
    loss_exit=-total_premium*(stop_loss_mult-1)

    col1,col2,col3=st.columns(3)
    col1.metric("Profit Exit Trigger",cr(profit_exit))
    col2.metric("Stop-Loss Trigger",cr(loss_exit))
    col3.metric("Max Risk (defined by stop)",cr(abs(loss_exit)))

    st.subheader("Simulated P&L Distribution (1000 expiries)")
    np.random.seed(42)
    n_sims=1000
    final_pnls=[]
    for _ in range(n_sims):
        move=np.random.normal(0,sigma*np.sqrt(T))
        ST=S*np.exp(move)
        call_payoff=max(ST-K_call,0)
        put_payoff=max(K_put-ST,0)
        pnl=total_premium-(call_payoff+put_payoff)*n_lots*25
        # apply stop loss/profit cap (simplified)
        pnl=max(min(pnl,profit_exit),loss_exit)
        final_pnls.append(pnl)

    final_pnls=np.array(final_pnls)
    win_rate=(final_pnls>0).mean()*100
    avg_pnl=final_pnls.mean()
    fig=go.Figure(go.Histogram(x=final_pnls,nbinsx=40,marker_color='#02B4AC'))
    fig.add_vline(x=0,line_color='red',line_dash='dash')
    fig.update_layout(title="Simulated Strategy P&L Distribution (per expiry)",height=350,
                      xaxis_title="P&L (₹)",yaxis_title="Frequency")
    st.plotly_chart(fig,use_container_width=True)

    col1,col2,col3=st.columns(3)
    col1.metric("Win Rate",pct(win_rate))
    col2.metric("Average P&L per expiry",cr(avg_pnl))
    col3.metric("Worst case (1st percentile)",cr(np.percentile(final_pnls,1)))

    st.warning("""
⚠️ **The Theta Harvesting Trap:** High win rate (often 70-85%) but
asymmetric risk — a handful of large losing trades (gap moves, vol spikes)
can erase MANY winning trades. Position sizing and hard stop-losses are
NON-NEGOTIABLE for systematic premium selling.
""")

# =========================================================
elif menu == "VIX-Based Strategy Switching":
    st.header("🔄 VIX-Based Strategy Switching")
    st.markdown("""
## The Concept

Algo systems switch between strategy MODES based on India VIX level —
matching the strategy to the volatility regime.

| VIX Level | Regime | Strategy Mode |
|---|---|---|
| VIX < 12 | Very Low Vol | Buy cheap options (straddles/strangles for events) |
| 12-18 | Normal | Balanced — directional spreads, moderate theta selling |
| 18-25 | Elevated | Favour premium selling (Iron Condors, credit spreads) |
| > 25 | High Stress | Reduce size, widen strikes, or stand aside |
| > 35 | Crisis | Defensive — close short vol, consider long vol hedges |
""")

    col1,col2 = st.columns(2)
    with col1:
        current_vix=st.slider("Current India VIX",8.0,45.0,18.0,0.5)
        capital=st.number_input("Trading Capital (₹)",value=1000000.0)

    def get_regime(vix):
        if vix<12: return "Very Low Vol","Buy options (cheap insurance/straddles)","#02B4AC",1.0
        elif vix<18: return "Normal","Balanced — directional spreads + moderate theta selling","#157A42",1.0
        elif vix<25: return "Elevated","Favour premium selling (Iron Condor, credit spreads)","#F5A623",0.7
        elif vix<35: return "High Stress","Reduce size 50%, widen strikes, defensive","#C03B3B",0.4
        else: return "Crisis","STAND ASIDE / close short vol / hedge with long puts","#6A0DAD",0.0

    regime,action,color,size_mult = get_regime(current_vix)

    with col2:
        st.metric("Current VIX",current_vix)
        st.metric("Regime",regime)
        st.metric("Recommended Position Size",pct(size_mult*100))
        st.metric("Adjusted Capital Deployment",cr(capital*size_mult))

    st.markdown(f"""
<div style="background-color:{color};padding:20px;border-radius:10px;color:white;text-align:center;">
<h3>Algo Action: {action}</h3>
</div>
""",unsafe_allow_html=True)

    # VIX regime chart
    vix_range=np.arange(8,45,0.5)
    size_mults=[get_regime(v)[3] for v in vix_range]
    fig=go.Figure(go.Scatter(x=vix_range,y=size_mults,mode='lines',line=dict(color='#174EA6',width=3),fill='tozeroy'))
    fig.add_vline(x=current_vix,line_dash='dash',line_color='red',annotation_text='Current VIX')
    for v,label in [(12,'12'),(18,'18'),(25,'25'),(35,'35')]:
        fig.add_vline(x=v,line_dash='dot',line_color='gray')
    fig.update_layout(title="Position Size Multiplier vs VIX (Regime-Based Sizing)",
                      xaxis_title="India VIX",yaxis_title="Size Multiplier",height=350)
    st.plotly_chart(fig,use_container_width=True)

    st.info("""
**Why this matters:** Premium-selling strategies look most attractive when
VIX is HIGH (rich premiums) — but that's also when GAP RISK is highest.
Regime-based sizing systematically REDUCES exposure exactly when raw
premium looks most tempting — a key discipline that manual traders often
fail to maintain.
""")

# =========================================================
elif menu == "Options Market Making":
    st.header("🏪 Options Market Making")
    st.markdown("""
## The Concept

Market makers continuously quote BOTH a bid and ask price for options,
profiting from the spread while managing inventory (Greeks) risk.

$$\\text{Quote} = \\text{Theoretical Price} \\pm \\frac{\\text{Spread}}{2} \\pm \\text{Skew Adjustment}$$

**Skew Adjustment:** market makers widen/shift quotes based on their current
inventory — if they're long calls, they'll quote calls cheaper (to sell more)
and more expensive to buy (less eager to buy more).
""")

    col1,col2 = st.columns(2)
    with col1:
        S=st.number_input("Spot",value=22000.0,key="mm_s")
        K=st.number_input("Strike",value=22000.0,key="mm_k")
        T_days=st.number_input("Days to Expiry",value=10,key="mm_t")
        sigma=st.number_input("IV %",value=16.0,key="mm_iv")/100
        base_spread_pct=st.number_input("Base Spread (% of price)",value=2.0,key="mm_spread")
        inventory=st.slider("Current Inventory (contracts, +ve=long)",-20,20,0)

    T=T_days/365
    r=0.07
    theo_price=bsm(S,K,T,r,sigma,"call")
    base_spread=theo_price*base_spread_pct/100

    # Skew adjustment based on inventory
    skew_factor=inventory*0.002  # shift mid price
    adjusted_mid=theo_price*(1-skew_factor)
    bid=adjusted_mid-base_spread/2
    ask=adjusted_mid+base_spread/2

    with col2:
        st.metric("Theoretical Price",cr(theo_price))
        st.metric("Adjusted Mid (inventory skew)",cr(adjusted_mid))
        st.metric("Bid Quote",cr(bid))
        st.metric("Ask Quote",cr(ask))

    if inventory>0:
        st.warning(f"📦 Long {inventory} contracts — quotes SKEWED DOWN to encourage selling (reduce long inventory)")
    elif inventory<0:
        st.warning(f"📦 Short {abs(inventory)} contracts — quotes SKEWED UP to encourage buying (reduce short inventory)")
    else:
        st.success("📦 Flat inventory — quotes centred on theoretical price")

    st.subheader("Spread Capture Simulation")
    n_trades=st.slider("Number of round-trip trades",10,200,50)
    spread_capture=base_spread*n_trades*25
    st.metric(f"Theoretical Spread Capture over {n_trades} round-trips",cr(spread_capture))
    st.info(f"""
Each round-trip (buy at bid ₹{bid:.2f}, sell at ask ₹{ask:.2f}) captures
₹{base_spread:.2f}/unit = ₹{base_spread*25:.2f}/lot.
Over {n_trades} round trips: ₹{spread_capture:,.0f} gross — BEFORE accounting
for adverse selection (informed traders hitting your quotes when you're wrong)
and inventory risk from unbalanced flow.
""")

# =========================================================
elif menu == "Greeks Risk Dashboard":
    st.header("📊 Greeks Risk Dashboard — Portfolio View")
    st.markdown("Aggregate Greeks across a multi-leg options book — the foundation of any derivatives algo's risk engine.")

    if "greeks_positions" not in st.session_state:
        st.session_state.greeks_positions=[
            {"type":"call","K":22200,"qty":-5,"T":10},
            {"type":"put","K":21800,"qty":-5,"T":10},
            {"type":"call","K":22600,"qty":3,"T":10},
            {"type":"put","K":21400,"qty":3,"T":10},
        ]

    S=st.number_input("Current Spot",value=22000.0,key="grd_s")
    sigma=st.number_input("IV %",value=16.0,key="grd_iv")/100
    r=0.07

    st.subheader("Position Builder")
    col1,col2,col3,col4,col5=st.columns(5)
    with col1: new_type=st.selectbox("Type",["call","put"])
    with col2: new_K=st.number_input("Strike",value=22000.0,key="new_k")
    with col3: new_qty=st.number_input("Qty (lots, +/-)",value=1)
    with col4: new_T=st.number_input("Days to Expiry",value=10,key="new_t")
    with col5:
        st.write("")
        if st.button("➕ Add"):
            st.session_state.greeks_positions.append({"type":new_type,"K":new_K,"qty":new_qty,"T":new_T})
            st.rerun()

    if st.button("🗑️ Clear All"):
        st.session_state.greeks_positions=[]
        st.rerun()

    if st.session_state.greeks_positions:
        rows=[]
        total_delta=total_gamma=total_theta=total_vega=0
        for p in st.session_state.greeks_positions:
            T=p["T"]/365
            g=greeks(S,p["K"],T,r,sigma)
            if p["type"]=="call":
                delta=g["dc"]; theta=g["tc"]
            else:
                delta=g["dp"]; theta=g["tp"]
            mult=p["qty"]*25
            d=delta*mult; gm=g["gamma"]*mult; th=theta*mult; vg=g["vega"]*mult
            total_delta+=d; total_gamma+=gm; total_theta+=th; total_vega+=vg
            rows.append({"Type":p["type"].upper(),"Strike":p["K"],"Qty (lots)":p["qty"],
                         "DTE":p["T"],"Delta":round(d,2),"Gamma":round(gm,5),
                         "Theta (₹/day)":round(th,2),"Vega":round(vg,2)})

        df_pos=pd.DataFrame(rows)
        st.dataframe(df_pos,use_container_width=True)

        st.subheader("📈 Aggregate Portfolio Greeks")
        col1,col2,col3,col4=st.columns(4)
        col1.metric("Net Delta",round(total_delta,2),
                    "Long bias" if total_delta>5 else ("Short bias" if total_delta<-5 else "Neutral"))
        col2.metric("Net Gamma",round(total_gamma,5),
                    "Long Gamma" if total_gamma>0 else "Short Gamma")
        col3.metric("Net Theta (₹/day)",round(total_theta,2),
                    "Collecting" if total_theta>0 else "Paying")
        col4.metric("Net Vega",round(total_vega,2),
                    "Long Vol" if total_vega>0 else "Short Vol")

        # Hedge recommendation
        if abs(total_delta)>5:
            hedge_action="SELL" if total_delta>0 else "BUY"
            st.warning(f"⚠️ **Delta Hedge Needed:** {hedge_action} {abs(total_delta):.1f} units of Nifty futures to neutralise delta")
        else:
            st.success("✅ Portfolio is delta-neutral within tolerance")

        # P&L scenario
        st.subheader("Scenario: Spot Moves ±1%, ±2%")
        scenarios=[-2,-1,0,1,2]
        pnl_data=[]
        for move_pct in scenarios:
            new_S=S*(1+move_pct/100)
            pnl_total=0
            for p in st.session_state.greeks_positions:
                T=p["T"]/365
                old_price=bsm(S,p["K"],T,r,sigma,p["type"])
                new_price=bsm(new_S,p["K"],T,r,sigma,p["type"])
                pnl_total+=(new_price-old_price)*p["qty"]*25
            pnl_data.append({"Spot Move":f"{move_pct:+}%","New Spot":round(new_S,1),"Portfolio P&L (₹)":round(pnl_total,2)})
        st.dataframe(pd.DataFrame(pnl_data),use_container_width=True)
    else:
        st.info("Add positions above to see aggregate Greeks.")

# =========================================================
elif menu == "Iron Condor Automation":
    st.header("🦅 Iron Condor Automation — Rule-Based Entry & Exit")
    st.markdown("""
## Automated Iron Condor Logic

An algo runs this EVERY TRADING DAY:

```
IF days_to_expiry == target_dte (e.g., 7):
    Find strikes at target deltas (e.g., 0.15 delta calls/puts)
    SELL OTM Call + SELL OTM Put (short strangle)
    BUY further OTM Call + BUY further OTM Put (wings)
    Record entry credit

EVERY DAY until expiry:
    Recalculate position P&L
    IF P&L >= profit_target (e.g., 50% of credit):
        CLOSE ALL LEGS — book profit
    IF P&L <= -stop_loss (e.g., -100% of credit):
        CLOSE ALL LEGS — cut loss
    IF days_to_expiry == 0:
        CLOSE ALL LEGS — expiry settlement
```
""")

    col1,col2 = st.columns(2)
    with col1:
        S=st.number_input("Spot at Entry",value=22000.0,key="ic_s")
        dte=st.number_input("Days to Expiry at Entry",value=7,key="ic_dte")
        sigma=st.number_input("IV %",value=15.0,key="ic_iv")/100
        wing_width=st.number_input("Wing Width (points)",value=200,key="ic_wing")
        short_delta=st.number_input("Short Strike Target Delta",value=0.15,key="ic_delta")
        n_lots=st.number_input("Lots",value=3,key="ic_n")
        profit_target=st.slider("Profit Target (%)",30,70,50,key="ic_pt")
        stop_loss=st.slider("Stop Loss (% of credit)",100,200,150,key="ic_sl")

    r=0.07; T=dte/365

    def find_strike(S,T,r,sigma,target_delta,opt):
        K=S
        for _ in range(60):
            g=greeks(S,K,T,r,sigma)
            d=g["dc"] if opt=="call" else abs(g["dp"])
            if abs(d-target_delta)<0.003: break
            if opt=="call": K += 5 if d>target_delta else -5
            else: K -= 5 if d>target_delta else -5
        return round(K/50)*50

    K_short_call=find_strike(S,T,r,sigma,short_delta,"call")
    K_short_put=find_strike(S,T,r,sigma,short_delta,"put")
    K_long_call=K_short_call+wing_width
    K_long_put=K_short_put-wing_width

    sc=bsm(S,K_short_call,T,r,sigma,"call")
    sp=bsm(S,K_short_put,T,r,sigma,"put")
    lc=bsm(S,K_long_call,T,r,sigma,"call")
    lp=bsm(S,K_long_put,T,r,sigma,"put")

    net_credit=(sc+sp-lc-lp)
    max_loss=wing_width-net_credit
    profit_exit=net_credit*profit_target/100
    loss_exit=net_credit*stop_loss/100

    with col2:
        st.metric(f"Short Call ({K_short_call:.0f})",cr(sc))
        st.metric(f"Short Put ({K_short_put:.0f})",cr(sp))
        st.metric(f"Long Call ({K_long_call:.0f})",cr(lc))
        st.metric(f"Long Put ({K_long_put:.0f})",cr(lp))

    col1,col2,col3,col4=st.columns(4)
    col1.metric("Net Credit",cr(net_credit*n_lots*25))
    col2.metric("Max Loss",cr(max_loss*n_lots*25))
    col3.metric("Profit Target",cr(profit_exit*n_lots*25))
    col4.metric("Stop Loss Level",cr(-loss_exit*n_lots*25))

    st.subheader("Daily P&L Simulation Through Expiry")
    days=np.arange(dte,-1,-1)
    np.random.seed(8)
    spot_path=[S]
    for _ in range(dte):
        spot_path.append(spot_path[-1]*(1+np.random.normal(0,sigma/np.sqrt(252))))

    pnls=[]
    exit_day=None
    for i,d in enumerate(days):
        T_rem=max(d/365,0.0001)
        spot_now=spot_path[i]
        sc_now=bsm(spot_now,K_short_call,T_rem,r,sigma,"call")
        sp_now=bsm(spot_now,K_short_put,T_rem,r,sigma,"put")
        lc_now=bsm(spot_now,K_long_call,T_rem,r,sigma,"call")
        lp_now=bsm(spot_now,K_long_put,T_rem,r,sigma,"put")
        current_value=(sc_now+sp_now-lc_now-lp_now)
        pnl=(net_credit-current_value)*n_lots*25
        pnls.append(pnl)
        if exit_day is None:
            if pnl>=profit_exit*n_lots*25:
                exit_day=i; exit_reason="PROFIT TARGET HIT"
            elif pnl<=-loss_exit*n_lots*25:
                exit_day=i; exit_reason="STOP LOSS HIT"

    fig=go.Figure(go.Scatter(x=list(range(len(pnls))),y=pnls,mode='lines+markers',line=dict(color='#02B4AC',width=2)))
    fig.add_hline(y=profit_exit*n_lots*25,line_dash='dash',line_color='green',annotation_text='Profit Target')
    fig.add_hline(y=-loss_exit*n_lots*25,line_dash='dash',line_color='red',annotation_text='Stop Loss')
    fig.add_hline(y=0,line_color='gray')
    if exit_day is not None:
        fig.add_vline(x=exit_day,line_dash='dot',line_color='orange',annotation_text=exit_reason)
    fig.update_layout(title="Simulated Iron Condor P&L Path",xaxis_title="Day",yaxis_title="P&L (₹)",height=400)
    st.plotly_chart(fig,use_container_width=True)

    if exit_day is not None:
        st.success(f"🤖 Algo would have EXITED on Day {exit_day} due to: {exit_reason}, P&L = ₹{pnls[exit_day]:,.0f}")
    else:
        st.info(f"🤖 Algo holds to expiry. Final P&L = ₹{pnls[-1]:,.0f}")

# =========================================================
elif menu == "Risk Management for Derivatives Algos":
    st.header("🛡️ Risk Management for Derivatives Algos")
    st.markdown("""
## Risk Layers Specific to Derivatives Algos

| Layer | Risk | Control |
|---|---|---|
| **Greeks Limits** | Unintended directional/vol exposure | Hard limits on net Delta, Vega, Gamma |
| **Margin Monitoring** | Margin calls from adverse moves | Real-time SPAN margin tracking, buffer |
| **Gap Risk** | Overnight/weekend gaps blow through strikes | Position sizing for worst-case gap |
| **Liquidity Risk** | Can't exit multi-leg position at fair price | Trade only liquid strikes (near ATM) |
| **Pin Risk** | Spot pins exactly at short strike at expiry | Close before expiry, don't hold to settlement |
| **Model Risk** | BSM assumptions break (fat tails, jumps) | Stress test with historical crash scenarios |
| **Correlation Risk** | Multiple positions move together in crisis | Portfolio-level VaR, not just per-position |
""")

    st.subheader("🔢 Margin & Gap Risk Calculator")
    col1,col2 = st.columns(2)
    with col1:
        capital=st.number_input("Trading Capital (₹)",value=2000000.0,key="rm_cap")
        margin_per_lot=st.number_input("SPAN Margin per Iron Condor Lot (₹)",value=60000.0)
        n_lots=st.number_input("Number of Lots",value=10,key="rm_lots")
        gap_scenario=st.slider("Overnight Gap Scenario (%)",1.0,8.0,3.0,0.5)
        S=st.number_input("Spot",value=22000.0,key="rm_s")

    total_margin=margin_per_lot*n_lots
    margin_utilisation=total_margin/capital*100
    gap_points=S*gap_scenario/100

    with col2:
        st.metric("Total Margin Required",cr(total_margin))
        st.metric("Margin Utilisation",pct(margin_utilisation))
        st.metric("Gap Scenario Move",f"{gap_points:.0f} points")

    if margin_utilisation>70:
        st.error(f"❌ {pct(margin_utilisation)} margin utilisation — DANGEROUS. A single adverse move could trigger margin call/forced liquidation.")
    elif margin_utilisation>50:
        st.warning(f"⚠️ {pct(margin_utilisation)} margin utilisation — Limited buffer for adverse moves.")
    else:
        st.success(f"✅ {pct(margin_utilisation)} margin utilisation — Adequate buffer maintained.")

    st.subheader("Crash Scenario Stress Test")
    crash_scenarios = [
        ("Normal day", 0.5, "#157A42"),
        ("Bad day", 2.0, "#F5A623"),
        ("Flash crash (2020-style)", 8.0, "#C03B3B"),
        ("Black swan (2008-style)", 15.0, "#6A0DAD"),
    ]
    stress_data=[]
    for name,move_pct,color in crash_scenarios:
        move_points=S*move_pct/100
        # Approximate: short strangle loses roughly linear beyond short strike with delta acceleration
        approx_loss_per_lot=move_points*25*0.5*(move_pct/2)  # simplified quadratic-ish
        total_loss=approx_loss_per_lot*n_lots
        stress_data.append({"Scenario":name,"Move":f"-{move_pct}%","Points":f"{move_points:.0f}",
                            "Approx Loss":cr(total_loss),"% of Capital":pct(total_loss/capital*100)})
    st.dataframe(pd.DataFrame(stress_data),use_container_width=True)

    st.error("""
⚠️ **The 2024 Lesson:** Several Indian retail algo/quant funds running
systematic options-selling strategies suffered severe drawdowns during
sharp single-day moves. The lesson: SIZE FOR THE BLACK SWAN, not the
average day. If a 15% move would wipe out your capital, you are
OVER-LEVERAGED — regardless of how good your average-day Sharpe ratio looks.
""")

# =========================================================
elif menu == "Backtesting a Vol-Selling Strategy":
    st.header("📉 Backtesting a Systematic Vol-Selling Strategy")
    st.markdown("""
## Backtest Setup

Simulate: Every week, sell a 0.15-delta strangle on Nifty, hold to expiry
(no profit target / stop for this simplified backtest), across 100 weeks
of simulated market history including occasional vol spikes.
""")

    col1,col2 = st.columns(2)
    with col1:
        n_weeks=st.slider("Number of weeks to backtest",20,150,100)
        base_iv=st.number_input("Base IV %",value=14.0,key="bt_iv")/100
        crash_prob=st.slider("Probability of vol-spike week (%)",1,15,5)
        crash_magnitude=st.number_input("Vol-spike move size (% of spot)",value=6.0,key="bt_crash")/100
        n_lots_bt=st.number_input("Lots",value=5,key="bt_lots")
        seed_bt=st.number_input("Seed",value=99,min_value=1,key="bt_seed")

    np.random.seed(seed_bt)
    S=22000
    r=0.07
    T=7/365
    weekly_pnls=[]
    spot_log=[S]

    for week in range(n_weeks):
        # find strikes at ~0.15 delta
        K_call=round((S*(1+1.5*base_iv*np.sqrt(T)))/50)*50
        K_put=round((S*(1-1.5*base_iv*np.sqrt(T)))/50)*50
        call_prem=bsm(S,K_call,T,r,base_iv,"call")
        put_prem=bsm(S,K_put,T,r,base_iv,"put")
        credit=(call_prem+put_prem)*n_lots_bt*25

        # simulate outcome
        is_crash = np.random.random()<crash_prob/100
        if is_crash:
            move=np.random.choice([-1,1])*crash_magnitude*(1+np.random.random())
        else:
            move=np.random.normal(0,base_iv*np.sqrt(T))
        ST=S*(1+move)

        call_payout=max(ST-K_call,0)
        put_payout=max(K_put-ST,0)
        pnl=credit-(call_payout+put_payout)*n_lots_bt*25
        weekly_pnls.append(pnl)
        S=ST
        spot_log.append(S)

    cum_pnl=np.cumsum(weekly_pnls)

    col1,col2,col3,col4=st.columns(4)
    col1.metric("Total P&L",cr(cum_pnl[-1]))
    col2.metric("Win Rate",pct((np.array(weekly_pnls)>0).mean()*100))
    col3.metric("Best Week",cr(max(weekly_pnls)))
    col4.metric("Worst Week",cr(min(weekly_pnls)))

    fig=go.Figure(go.Scatter(x=list(range(len(cum_pnl))),y=cum_pnl,mode='lines',
                              line=dict(color='#157A42' if cum_pnl[-1]>0 else '#C03B3B',width=2),
                              fill='tozeroy'))
    fig.add_hline(y=0,line_color='black')
    fig.update_layout(title="Cumulative P&L — Systematic Vol-Selling",xaxis_title="Week",
                      yaxis_title="Cumulative P&L (₹)",height=350)
    st.plotly_chart(fig,use_container_width=True)

    # Drawdown
    running_max=np.maximum.accumulate(cum_pnl)
    drawdown=cum_pnl-running_max
    max_dd=drawdown.min()
    st.metric("Maximum Drawdown",cr(max_dd))

    fig2=go.Figure(go.Scatter(x=list(range(len(drawdown))),y=drawdown,mode='lines',
                              fill='tozeroy',line=dict(color='#C03B3B')))
    fig2.update_layout(title="Drawdown Over Time",height=250,yaxis_title="Drawdown (₹)")
    st.plotly_chart(fig2,use_container_width=True)

    st.warning(f"""
**Try increasing the crash probability** from {crash_prob}% to 10-15% and
re-running — notice how a strategy that looked steadily profitable can
suddenly show a massive drawdown from just ONE or TWO bad weeks. This is
the central challenge of backtesting short-volatility strategies: tail
events are RARE in any given sample, so backtests can dramatically
UNDERSTATE true risk.
""")

# =========================================================
elif menu == "Step-by-Step Solver":
    st.header("🧠 Step-by-Step Solver")
    problem=st.selectbox("Choose Problem",[
        "Delta Hedge Quantity",
        "Gamma Scalping P&L",
        "Futures Fair Value & Basis",
        "Breakeven Realised Volatility",
        "Iron Condor Net Credit",
        "VRP (Vol Risk Premium)",
    ])

    if problem=="Delta Hedge Quantity":
        delta=st.number_input("Option Delta",value=0.45)
        lots=st.number_input("Lots Short",value=4)
        lot_size=25
        hedge=delta*lots*lot_size
        st.write("**Hedge Units = Delta × Lots × Lot Size**")
        st.latex(f"= {delta} \\times {lots} \\times {lot_size} = {round(hedge,2)}")
        st.success(f"BUY {round(hedge,1)} units of futures to hedge (≈ {round(hedge/25,2)} lots)")

    elif problem=="Gamma Scalping P&L":
        gamma=st.number_input("Position Gamma",value=0.05)
        move=st.number_input("Spot Move (points)",value=100.0)
        pnl=0.5*gamma*move**2
        st.write("**Gamma P&L ≈ 0.5 × Gamma × (ΔS)²**")
        st.latex(f"= 0.5 \\times {gamma} \\times {move}^2 = {round(pnl,2)}")
        st.success(f"Gamma P&L from this move ≈ ₹{round(pnl,2)}")

    elif problem=="Futures Fair Value & Basis":
        S=st.number_input("Spot",value=22000.0)
        r=st.number_input("Risk-free rate %",value=7.0)/100
        d=st.number_input("Dividend yield %",value=1.2)/100
        T=st.number_input("Days to expiry",value=20)/365
        F_market=st.number_input("Market Futures Price",value=22080.0)
        F_fair=S*np.exp((r-d)*T)
        basis=F_market-F_fair
        st.write("**Fair Futures = S × e^((r-d)×T)**")
        st.latex(f"= {S} \\times e^{{({r}-{d})\\times{round(T,4)}}} = {round(F_fair,2)}")
        st.write("**Basis = Market - Fair**")
        st.latex(f"= {F_market} - {round(F_fair,2)} = {round(basis,2)}")
        if basis>0: st.success(f"Basis = {round(basis,2)} → SELL Futures, BUY Spot")
        else: st.error(f"Basis = {round(basis,2)} → BUY Futures, SELL Spot")

    elif problem=="Breakeven Realised Volatility":
        gamma=st.number_input("Position Gamma",value=0.06,key="be_g")
        theta=st.number_input("Position Theta (₹/day, negative)",value=-500.0,key="be_t")
        S=st.number_input("Spot",value=22000.0,key="be_s")
        be_move=np.sqrt(2*abs(theta)/gamma)
        be_vol=be_move/S*np.sqrt(252)*100
        st.write("**Breakeven move: 0.5×Γ×ΔS² = |Θ| → ΔS = sqrt(2|Θ|/Γ)**")
        st.latex(f"\\Delta S = \\sqrt{{2\\times{abs(theta)}/{gamma}}} = {round(be_move,2)}")
        st.write("**Annualised: σ_BE = ΔS/S × sqrt(252)**")
        st.latex(f"= {round(be_move,2)}/{S} \\times \\sqrt{{252}} = {round(be_vol,2)}\\%")
        st.success(f"Breakeven realised volatility ≈ {round(be_vol,2)}%")

    elif problem=="Iron Condor Net Credit":
        sc=st.number_input("Short Call Premium",value=45.0)
        sp=st.number_input("Short Put Premium",value=40.0)
        lc=st.number_input("Long Call Premium",value=15.0)
        lp=st.number_input("Long Put Premium",value=12.0)
        net=sc+sp-lc-lp
        st.write("**Net Credit = (Short Call + Short Put) - (Long Call + Long Put)**")
        st.latex(f"= ({sc}+{sp}) - ({lc}+{lp}) = {round(net,2)}")
        st.success(f"Net Credit per unit = ₹{round(net,2)} = ₹{round(net*25,2)} per lot")

    elif problem=="VRP (Vol Risk Premium)":
        iv=st.number_input("Implied Volatility %",value=18.0)
        rv=st.number_input("Realised Volatility %",value=14.0)
        vrp=iv-rv
        st.write("**VRP = IV - RV**")
        st.latex(f"= {iv} - {rv} = {round(vrp,2)}\\%")
        if vrp>0: st.success(f"VRP = +{round(vrp,2)}% → SELL volatility (premium rich)")
        else: st.error(f"VRP = {round(vrp,2)}% → BUY volatility (premium cheap)")

# =========================================================
elif menu == "Quiz Engine":
    st.header("📝 Algo Trading with Derivatives — Quiz")
    if "deriv_algo_quiz_idx" not in st.session_state: st.session_state.deriv_algo_quiz_idx=0

    questions=[
        {"q":"Option Delta=0.40, Short 3 lots (25 units/lot). How many futures units to BUY for delta-neutral?",
         "ans":round(0.40*3*25,2),"hint":"Hedge units = |Delta| × Lots × Lot Size"},
        {"q":"Position Gamma=0.08, Spot moves 80 points. What is the Gamma P&L (₹)?",
         "ans":round(0.5*0.08*80**2,2),"hint":"Gamma P&L ≈ 0.5 × Gamma × (ΔS)²"},
        {"q":"Spot=22000, r=7%, dividend=1%, T=15/365. What is the fair futures price?",
         "ans":round(22000*np.exp((0.07-0.01)*(15/365)),2),"hint":"F_fair = S × e^((r-d)×T)"},
        {"q":"Implied Vol=20%, Realised Vol=15%. What is the Vol Risk Premium (VRP)?",
         "ans":5.0,"hint":"VRP = IV - RV"},
        {"q":"Iron Condor: Sell Call=50, Sell Put=45, Buy Call=18, Buy Put=15. Net credit per unit?",
         "ans":round(50+45-18-15,2),"hint":"Net Credit = (ShortCall+ShortPut)-(LongCall+LongPut)"},
        {"q":"Gamma=0.05, Theta=-400 (₹/day). What's the breakeven daily move (points)?",
         "ans":round(np.sqrt(2*400/0.05),2),"hint":"ΔS = sqrt(2|Θ|/Γ)"},
    ]
    if st.button("🔄 New Question"):
        st.session_state.deriv_algo_quiz_idx=random.randint(0,len(questions)-1); st.rerun()
    qd=questions[st.session_state.deriv_algo_quiz_idx]
    st.markdown(f"**Q:** {qd['q']}")
    ans=st.number_input("Your Answer",value=0.0,step=0.01,key="deriv_algo_ans")
    if st.button("Submit"):
        if abs(ans-qd["ans"])<max(0.5,abs(qd["ans"])*0.02):
            st.success(f"✅ Correct! = {qd['ans']}")
            st.balloons()
        else: st.error(f"❌ Answer = {qd['ans']}")
    if st.checkbox("Hint"): st.info(f"💡 {qd['hint']}")

# =========================================================
elif menu == "Formula Cheat Sheet":
    st.header("📘 Algo Trading with Derivatives — Formula Cheat Sheet")
    formulas="""
ALGO TRADING WITH DERIVATIVES — COMPLETE REFERENCE
=====================================================

GREEKS-BASED P&L DECOMPOSITION
-----------------------------------------------------
ΔP ≈ Δ×ΔS + 0.5×Γ×(ΔS)² + Θ×Δt + ν×Δσ

DELTA HEDGING
-----------------------------------------------------
Hedge Units = -Δ_option × Quantity × Lot Size
Re-hedge when: |Δ_current - Δ_hedged| > threshold

GAMMA SCALPING
-----------------------------------------------------
Gamma P&L (per move) ≈ 0.5 × Γ × (ΔS)²
Theta Cost (per day) = |Θ| × 1 day
Breakeven Move: ΔS_BE = sqrt(2|Θ|/Γ)
Breakeven Realised Vol: σ_BE = (ΔS_BE/S) × sqrt(252)

VOLATILITY ARBITRAGE
-----------------------------------------------------
VRP (Vol Risk Premium) = IV - RV_forecast
VRP > 0 → SELL volatility (premium rich)
VRP < 0 → BUY volatility (premium cheap)

FUTURES-CASH BASIS (CASH AND CARRY)
-----------------------------------------------------
Fair Futures = S × e^((r-d)×T)
Basis = F_market - F_fair
Basis > 0 → SELL Futures + BUY Spot
Basis < 0 → BUY Futures + SELL Spot (reverse arb)
Basis → 0 as T → 0 (convergence at expiry)

CALENDAR SPREAD / TERM STRUCTURE
-----------------------------------------------------
Term Structure Signal = IV_near - IV_far
Steep positive signal → SELL near, BUY far (theta capture)

THETA HARVESTING (SHORT PREMIUM)
-----------------------------------------------------
Daily Theta Income = |Θ_call| + |Θ_put|
Profit Target Exit: close at X% of credit received
Stop Loss Exit: close at Y× credit received as loss

VIX-BASED REGIME SIZING
-----------------------------------------------------
VIX<12: Buy vol (cheap) | 12-18: Balanced
18-25: Favour selling premium | >25: Reduce size
>35: Defensive/stand-aside

OPTIONS MARKET MAKING
-----------------------------------------------------
Quote = Theoretical Price ± Spread/2 ± Inventory Skew
Inventory Skew: long inventory → quote lower (encourage selling)

IRON CONDOR (NET CREDIT)
-----------------------------------------------------
Net Credit = (Short Call + Short Put) - (Long Call + Long Put)
Max Loss = Wing Width - Net Credit
Max Profit = Net Credit (at expiry, between short strikes)

RISK MANAGEMENT
-----------------------------------------------------
Margin Utilisation = Total Margin / Capital
Position sized for WORST CASE (gap/crash), not average day
Portfolio VaR aggregates correlated positions, not just per-trade

KEY RULES
-----------------------------------------------------
- Greeks change continuously — hedge systems must run in real time
- High win-rate ≠ good strategy if tail losses are large (asymmetric risk)
- VRP exists on average, but crash days can wipe out months of gains
- Size for the black swan: a 10-15% gap shouldn't threaten survival
- Backtest with REALISTIC tail event frequency, not just "normal" data
=====================================================
"""
    st.text_area("Reference",formulas,height=700)
    st.download_button("📥 Download",data=formulas,file_name="Algo_Derivatives_Formulas.txt")

# =========================================================
elif menu == "Case Study — Systematic Short Straddle on Nifty":
    st.header("📚 Case Study: Systematic Short Straddle on Nifty Weekly Options")
    st.markdown("""
## The Strategy (Popular Among Indian Retail/Quant Algos)

**Every Thursday (Nifty weekly expiry):**
1. Sell ATM Call + ATM Put (short straddle) for next week's expiry
2. Hold until expiry (or exit at stop-loss)
3. Collect premium if Nifty stays within a range

This became EXTREMELY popular in India 2019-2023 as Nifty/BankNifty weekly
options launched, with many "algo" platforms (Sensibull, Streak, Tradetron)
offering ready-made versions.

## What Happened — A Realistic Narrative

| Period | Market Condition | Outcome |
|---|---|---|
| 2021-2022 (most weeks) | Range-bound, moderate vol | Steady small profits — "free money" feeling |
| Budget Day / RBI Day | Occasional large move | Some losing weeks, manageable |
| **Specific shock weeks** | Sharp single-day moves (geopolitical, global cues) | LARGE losses — sometimes erasing 10-20 weeks of gains in ONE day |
""")

    col1,col2 = st.columns(2)
    with col1:
        weekly_premium=st.number_input("Average Weekly Premium Collected (₹, per lot)",value=2500.0)
        n_lots_case=st.number_input("Lots",value=10,key="case_lots")
        good_weeks_pct=st.slider("% of weeks: small profit (~full premium)",60,90,75)
        bad_week_loss_mult=st.slider("Bad week loss = X × weekly premium",3,15,8)
        bad_weeks_pct=st.slider("% of weeks: bad loss",1,10,3)

    neutral_pct=100-good_weeks_pct-bad_weeks_pct
    weekly_income=weekly_premium*n_lots_case
    expected_weekly=(good_weeks_pct/100*weekly_income +
                     neutral_pct/100*0 +
                     bad_weeks_pct/100*(-weekly_income*bad_week_loss_mult))

    with col2:
        st.metric("Weekly Income (good week)",cr(weekly_income))
        st.metric("Bad Week Loss",cr(-weekly_income*bad_week_loss_mult))
        st.metric("Expected Weekly P&L",cr(expected_weekly))
        st.metric("Expected Annual P&L (52 weeks)",cr(expected_weekly*52))

    if expected_weekly>0:
        st.success(f"✅ Positive expectancy: ₹{expected_weekly:,.0f}/week on average")
    else:
        st.error(f"❌ Negative expectancy: ₹{expected_weekly:,.0f}/week — strategy LOSES money despite high win rate!")

    # Simulate 52 weeks
    np.random.seed(2023)
    weekly_pnls=[]
    for _ in range(52):
        r=np.random.random()
        if r<good_weeks_pct/100:
            weekly_pnls.append(weekly_income*np.random.uniform(0.7,1.0))
        elif r<(good_weeks_pct+neutral_pct)/100:
            weekly_pnls.append(weekly_income*np.random.uniform(-0.3,0.3))
        else:
            weekly_pnls.append(-weekly_income*bad_week_loss_mult*np.random.uniform(0.8,1.3))

    cum=np.cumsum(weekly_pnls)
    fig=go.Figure(go.Scatter(x=list(range(1,53)),y=cum,mode='lines+markers',
                              line=dict(color='#157A42' if cum[-1]>0 else '#C03B3B',width=2)))
    fig.add_hline(y=0,line_color='black')
    fig.update_layout(title="52-Week Simulated P&L — Systematic Short Straddle",
                      xaxis_title="Week",yaxis_title="Cumulative P&L (₹)",height=400)
    st.plotly_chart(fig,use_container_width=True)

    n_bad_weeks=sum(1 for p in weekly_pnls if p<-weekly_income)
    st.metric("Number of 'Bad Weeks' in this simulation",n_bad_weeks)
    st.metric("Final 52-week P&L",cr(cum[-1]))

    st.subheader("🎓 Discussion Questions")
    st.markdown("""
1. If 75% of weeks are profitable but the strategy still loses money overall,
   what does that tell you about the RELATIVE SIZE of wins vs losses?

2. SEBI data (2023-24) showed that the VAST MAJORITY of retail F&O traders
   in India lose money. How might "set and forget" algo platforms running
   strategies like this contribute to that statistic?

3. How would adding a STOP-LOSS (e.g., exit if loss exceeds 1.5× premium
   received) change the P&L distribution? What's the trade-off?

4. Many traders report this strategy "worked great for 2 years" before a
   single bad week erased the gains. Is 2 years of data enough to validate
   a strategy with this risk profile? Why or why not?

5. How does this relate to the "Volatility Risk Premium" — is VRP a real,
   persistent edge, or is it compensation for tail risk that simply hasn't
   shown up yet in a given sample?
""")

    st.warning("""
**The Honest Conclusion:** Systematic premium-selling strategies can have
genuine positive expectancy over LONG horizons (the VRP is empirically real
on average). But the P&L distribution is HIGHLY SKEWED — many small wins,
occasional large losses. Position sizing, hedging tail risk (e.g., buying
far OTM "tail" options), and realistic expectations about drawdowns are
essential. "It worked for 2 years" is NOT the same as "it works."
""")
