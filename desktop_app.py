import tkinter as tk
from tkinter import ttk, messagebox
import schwabdev
import pandas as pd
import numpy as np
import altair as alt
import datetime
import tempfile, os, webbrowser
from math import log, sqrt, exp
from scipy.stats import norm    

# === Schwab credentials ===
APP_KEY = ""        # your Schwab app key
SECRET = ""           # your Schwab secret
CALLBACK_URL = "https://127.0.0.1"

client = schwabdev.Client(APP_KEY, SECRET, CALLBACK_URL)

exp_data_map = {}
current_symbol = ""
current_price = 0.0

# === Black-Scholes Gamma (with dividend yield) ===
def bs_gamma(S, K, T, r, q, sigma):
    try:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0
        dp = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        gamma = np.exp(-q*T) * norm.pdf(dp) / (S * sigma * np.sqrt(T))
        return gamma
    except Exception:
        return 0.0

# === Fetch Option Chain ===
def fetch_option_chain(symbol):
    try:
        resp = client.option_chains(
            symbol=symbol.upper(),
            contractType="ALL",
            strikeCount=40,
            includeUnderlyingQuote=True
        )
        data = resp.json()

                # === DEBUG: print just one option sample ===
        print(f"\n=== DEBUG: Raw OptionChain keys for {symbol} ===")
        print("Available keys:", list(data.keys()))

        # Print one sample call & put entry if available
        calls = data.get("callExpDateMap", {})
        puts = data.get("putExpDateMap", {})

        if calls:
            first_exp = next(iter(calls))
            first_strike = next(iter(calls[first_exp]))
            print("\nSample CALL JSON:")
            print(calls[first_exp][first_strike][0])
        else:
            print("No calls found.")

        if puts:
            first_exp = next(iter(puts))
            first_strike = next(iter(puts[first_exp]))
            print("\nSample PUT JSON:")
            print(puts[first_exp][first_strike][0])
        else:
            print("No puts found.")
        print("=== END DEBUG ===\n")
        
        all_expirations = sorted(set(list(calls.keys()) + list(puts.keys())))

        exp_to_df = {}
        for exp in all_expirations:
            call_opts, put_opts = [], []

            if exp in calls:
                for strike, optlist in calls[exp].items():
                    opt = optlist[0]
                    call_opts.append({
                        "Strike": float(strike),
                        "Bid_Call": opt.get("bid", 0.0),
                        "Ask_Call": opt.get("ask", 0.0),
                        "Delta_Call": opt.get("delta", 0.0),
                        "Theta_Call": opt.get("theta", 0.0),
                        "Gamma_Call": opt.get("gamma", 0.0),
                        "IV_Call": opt.get("volatility", 0.0),
                        "OI_Call": opt.get("openInterest", 0.0)
                    })

            if exp in puts:
                for strike, optlist in puts[exp].items():
                    opt = optlist[0]
                    put_opts.append({
                        "Strike": float(strike),
                        "Bid_Put": opt.get("bid", 0.0),
                        "Ask_Put": opt.get("ask", 0.0),
                        "Delta_Put": opt.get("delta", 0.0),
                        "Theta_Put": opt.get("theta", 0.0),
                        "Gamma_Put": opt.get("gamma", 0.0),
                        "IV_Put": opt.get("volatility", 0.0),
                        "OI_Put": opt.get("openInterest", 0.0)
                    })

            df_calls = pd.DataFrame(call_opts).sort_values("Strike")
            df_puts = pd.DataFrame(put_opts).sort_values("Strike")
            df_merged = pd.merge(df_calls, df_puts, on="Strike", how="outer").fillna("")
            exp_to_df[exp] = df_merged

        return exp_to_df, all_expirations

    except Exception as e:
        messagebox.showerror("Error", f"Failed to fetch data for {symbol}: {e}")
        return {}, []

# === Fetch Stock Price ===
def fetch_stock_price(symbol):
    try:
        resp = client.quotes(symbol)
        data = resp.json()
        sym_data = data.get(symbol.upper(), {})
        price = (
            sym_data.get("quote", {}).get("lastPrice")
            or sym_data.get("regular", {}).get("regularMarketLastPrice")
            or sym_data.get("extended", {}).get("lastPrice")
            or sym_data.get("quote", {}).get("mark")
            or 0.0
        )
        return float(price)
    except Exception as e:
        messagebox.showwarning("Quote Error", f"Could not fetch quote for {symbol}: {e}")
        return 0.0

# === GUI Setup ===
root = tk.Tk()
root.title("Schwab Option Chain Dashboard")
root.geometry("1400x750")

sidebar_visible = True
sidebar_frame = tk.Frame(root, bg="#2b2b2b", width=200)
sidebar_frame.pack(side=tk.LEFT, fill=tk.Y)

def toggle_sidebar():
    global sidebar_visible
    if sidebar_visible:
        sidebar_frame.pack_forget()
        toggle_btn.config(text="☰ Show Menu")
        sidebar_visible = False
    else:
        main_frame.pack_forget()
        sidebar_frame.pack(side=tk.LEFT, fill=tk.Y)
        main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        toggle_btn.config(text="☰ Hide Menu")
        sidebar_visible = True

toggle_btn = tk.Button(root, text="☰ Hide Menu", command=toggle_sidebar)
toggle_btn.pack(anchor=tk.NW)

tk.Label(sidebar_frame, text="Tools", bg="#2b2b2b", fg="white", font=("Arial", 12, "bold")).pack(pady=10)
update_gamma_btn = tk.Button(sidebar_frame, text="Generate Gamma Chart", command=lambda: generate_gamma_chart())
update_gamma_btn.pack(pady=5, padx=10, fill=tk.X)

# === Main Frame ===
main_frame = tk.Frame(root)
main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Top controls
top_frame = tk.Frame(main_frame)
top_frame.pack(pady=10)

tk.Label(top_frame, text="Stock Symbol:").pack(side=tk.LEFT, padx=5)
symbol_entry = tk.Entry(top_frame, width=10)
symbol_entry.pack(side=tk.LEFT, padx=5)
symbol_entry.insert(0, "SPY")

price_var = tk.StringVar(value="—")
price_label = tk.Label(top_frame, textvariable=price_var, fg="blue", font=("Arial", 12, "bold"))
price_label.pack(side=tk.LEFT, padx=10)

tk.Label(top_frame, text="Expiration:").pack(side=tk.LEFT, padx=5)
exp_var = tk.StringVar()
exp_dropdown = ttk.Combobox(top_frame, textvariable=exp_var, state="readonly", width=25)
exp_dropdown.pack(side=tk.LEFT, padx=5)

# === Fetch options ===
def on_fetch():
    global exp_data_map, current_symbol, current_price
    symbol = symbol_entry.get().strip().upper()
    if not symbol:
        messagebox.showwarning("Input required", "Please enter a stock symbol.")
        return
    messagebox.showinfo("Fetching", f"Fetching option chain for {symbol}...")
    current_symbol = symbol

    current_price = fetch_stock_price(symbol)
    price_var.set(f"Price: ${current_price:.2f}" if current_price else "Price: —")

    exp_data_map, expirations = fetch_option_chain(symbol)
    if not expirations:
        messagebox.showwarning("No data", f"No option data returned for {symbol}.")
        return

    previous_exp = exp_var.get()
    exp_dropdown["values"] = expirations
    if previous_exp in expirations:
        exp_var.set(previous_exp)
    else:
        exp_var.set(expirations[0])
    update_table(exp_var.get())

def auto_refresh_price():
    global current_symbol, current_price
    if current_symbol:
        new_price = fetch_stock_price(current_symbol)
        if new_price > 0:
            diff = new_price - current_price
            current_price = new_price
            color = "green" if diff > 0 else "red" if diff < 0 else "blue"
            price_var.set(f"Price: ${current_price:.2f}")
            price_label.config(fg=color)
    root.after(10000, auto_refresh_price)

def auto_refresh_options():
    global exp_data_map, current_symbol
    if current_symbol:
        exp_data_map, expirations = fetch_option_chain(current_symbol)
        if expirations:
            previous_exp = exp_var.get()
            exp_dropdown["values"] = expirations
            if previous_exp in expirations:
                exp_var.set(previous_exp)
            else:
                exp_var.set(expirations[0])
            update_table(exp_var.get())
    root.after(120000, auto_refresh_options)

fetch_btn = tk.Button(top_frame, text="Fetch Options", command=on_fetch)
fetch_btn.pack(side=tk.LEFT, padx=5)

# === Table ===
cols = [
    "Bid_Call", "Ask_Call", "Delta_Call", "Theta_Call", "Gamma_Call", "IV_Call", "OI_Call",
    "Strike",
    "Bid_Put", "Ask_Put", "Delta_Put", "Theta_Put", "Gamma_Put", "IV_Put", "OI_Put"
]
headers = [
    "Call Bid", "Call Ask", "Δ(Call)", "Θ(Call)", "Γ(Call)", "IV(Call)", "OI(Call)",
    "Strike",
    "Put Bid", "Put Ask", "Δ(Put)", "Θ(Put)", "Γ(Put)", "IV(Put)", "OI(Put)"
]

frame = tk.Frame(main_frame)
frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
tree = ttk.Treeview(frame, columns=cols, show="headings")
for c, h in zip(cols, headers):
    tree.heading(c, text=h)
    tree.column(c, width=100, anchor=tk.CENTER)

vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
tree.configure(yscroll=vsb.set, xscroll=hsb.set)
vsb.pack(side=tk.RIGHT, fill=tk.Y)
hsb.pack(side=tk.BOTTOM, fill=tk.X)
tree.pack(fill=tk.BOTH, expand=True)

def update_table(selected_exp):
    df = exp_data_map.get(selected_exp, pd.DataFrame())
    tree.delete(*tree.get_children())
    if not df.empty:
        for _, row in df.iterrows():
            vals = [row.get(c, "") for c in cols]
            tree.insert("", tk.END, values=vals)

def on_exp_change(event):
    update_table(exp_var.get())

exp_dropdown.bind("<<ComboboxSelected>>", on_exp_change)

# === Gamma Chart ===
def generate_gamma_chart():
    if not current_symbol:
        messagebox.showwarning("Missing", "Please fetch a symbol first.")
        return
    selected_exp = exp_var.get()
    if not selected_exp or selected_exp not in exp_data_map:
        messagebox.showwarning("No Expiration", "Select a valid expiration.")
        return

    df = exp_data_map[selected_exp]
    if df.empty:
        messagebox.showwarning("No Data", "No data for this expiration.")
        return

    CONTRACT_MULTIPLIER = 100
    r = 0.05
    q = 0.015
    T = 7 / 365  # ~1 week default
    gamma_data = []

    for _, row in df.iterrows():
        strike = float(row.get("Strike", 0) or 0)
        if strike <= 0:
            continue

        iv_call = float(row.get("IV_Call", 0) or 0.2)
        iv_put = float(row.get("IV_Put", 0) or 0.2)
        oi_call = float(row.get("OI_Call", 0) or 1)
        oi_put = float(row.get("OI_Put", 0) or 1)

        gamma_call = bs_gamma(current_price, strike, T, r, q, iv_call)
        gamma_put = bs_gamma(current_price, strike, T, r, q, iv_put)

        # === Correct CBOE-style exposure scaling ===
        gex_call = gamma_call * oi_call * CONTRACT_MULTIPLIER * (current_price ** 2) * 0.01
        gex_put = -gamma_put * oi_put * CONTRACT_MULTIPLIER * (current_price ** 2) * 0.01

        gamma_data.append({"Strike": strike, "Type": "CALL", "Exposure": gex_call})
        gamma_data.append({"Strike": strike, "Type": "PUT", "Exposure": gex_put})

    df_plot = pd.DataFrame(gamma_data)
    total_gex = df_plot["Exposure"].sum() / 1e9
    df_plot["Exposure_Bn"] = df_plot["Exposure"] / 1e9

    chart = (
        alt.Chart(df_plot)
        .mark_bar(size=14)
        .encode(
            x=alt.X("Strike:Q", title="Strike Price"),
            y=alt.Y("Exposure_Bn:Q", title="Spot Gamma Exposure ($ billions / 1% move)", scale=alt.Scale(domainMid=0)),
            color=alt.Color("Type:N", scale=alt.Scale(domain=["CALL", "PUT"], range=["green", "red"]))
        )
        .properties(
            title={
                "text": [f"${current_symbol} Gamma Exposure ({selected_exp.split(':')[0]})"],
                "subtitle": [f"Total Gamma: {total_gex:+.2f} Bn | Updated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Black–Scholes, S²×0.01 scaling)"],
                "fontSize": 18
            },
            width=850,
            height=450
        )
    )

    html_path = os.path.join(tempfile.gettempdir(), f"{current_symbol}_{selected_exp.replace(':','-')}_gamma.html")
    chart.save(html_path)
    webbrowser.open(f"file://{html_path}")
    messagebox.showinfo("Gamma Chart", "Gamma chart generated and opened in browser.")

# === Start Refresh Loops ===
auto_refresh_price()
auto_refresh_options()
root.mainloop()
