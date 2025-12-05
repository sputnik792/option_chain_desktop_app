import tkinter as tk
from tkinter import filedialog
from tkinter import ttk, messagebox
from tkhtmlview import HTMLLabel
import schwabdev
import pandas as pd
import numpy as np
import altair as alt
import datetime
import tempfile, os, webbrowser
from math import log, sqrt, exp
from scipy.stats import norm    
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# === Schwab credentials ===
APP_KEY = "NWy77tZSfVAMzZBMkZ6HBoliNH7NX6Rh"        # your Schwab app key
SECRET = "448aaPGZUahJKaMv"           # your Schwab secret
CALLBACK_URL = "https://127.0.0.1"

client = schwabdev.Client(APP_KEY, SECRET, CALLBACK_URL)

def load_csv_index():
    """
    Loads csv index options from local CSV file and inserts
    the results into the same data structures as Schwab live options.
    """
    global exp_data_map, current_symbol, current_price

    symbol = index_symbol_var.get()

    # Determine file source
    if csv_mode_var.get() == "Default File":
        filename = f"{symbol.lower()}_quotedata.csv"
    else:
        filename = filedialog.askopenfilename(
            title=f"Select {symbol} CSV File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not filename:
            messagebox.showwarning("Cancelled", "No CSV file selected.")
            return

    try:
        with open(filename, "r") as f:
            lines = f.readlines()
    except Exception as e:
        messagebox.showerror("File Error", f"Could not read CSV file:\n{e}")
        return

    # --- Parse Spot Price (Line 2 in CBOE format) ---
    try:
        spotLine = lines[1]
        current_price = float(spotLine.split("Last:")[1].split(",")[0])
    except:
        current_price = 0.0

    # Update UI
    current_symbol = symbol + " (CSV)"
    price_var.set(f"Price: ${current_price:.2f}")

    # --- Parse Options Table starting at row 4 ---
    try:
        df = pd.read_csv(filename, sep=",", header=None, skiprows=4)
    except Exception as e:
        messagebox.showerror("CSV Error", f"Unable to parse CSV:\n{e}")
        return

    df.columns = [
        'ExpirationDate','Calls','CallLastSale','CallNet','CallBid','CallAsk','CallVol',
        'CallIV','CallDelta','CallGamma','CallOpenInt',
        'Strike','Puts','PutLastSale','PutNet','PutBid','PutAsk','PutVol',
        'PutIV','PutDelta','PutGamma','PutOpenInt'
    ]

    # Convert numeric fields
    for col in ["Strike","CallIV","PutIV","CallDelta","PutDelta",
                "CallGamma","PutGamma","CallOpenInt","PutOpenInt"]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Convert expiration formats
    df["ExpirationDate"] = pd.to_datetime(df["ExpirationDate"], errors="coerce")

    # Rebuild expiration list (Schwab-style keys)
    exp_data_map = {}
    expirations = []

    for exp_date, group in df.groupby("ExpirationDate"):
        if pd.isna(exp_date):
            continue

        exp_key = exp_date.strftime("%Y-%m-%d") + ":0"
        expirations.append(exp_key)

        # Build standard schema expected by the rest of the system
        clean_df = pd.DataFrame({
            "Strike": group["Strike"],
            "Bid_Call": group["CallBid"].replace(0, "N/A"),
            "Ask_Call": group["CallAsk"].replace(0, "N/A"),
            "Delta_Call": group["CallDelta"],
            "Theta_Call": np.zeros(len(group)),     # CSV does not contain theta → set 0
            "Gamma_Call": group["CallGamma"],
            "IV_Call": group["CallIV"],
            "OI_Call": group["CallOpenInt"],

            "Bid_Put": group["PutBid"].replace(0, "N/A"),
            "Ask_Put": group["PutAsk"].replace(0, "N/A"),
            "Delta_Put": group["PutDelta"],
            "Theta_Put": np.zeros(len(group)),      # CSV lacks theta → set 0
            "Gamma_Put": group["PutGamma"],
            "IV_Put": group["PutIV"],
            "OI_Put": group["PutOpenInt"],
        })

        exp_data_map[exp_key] = clean_df.sort_values("Strike")

    # Update expiration dropdown
    exp_dropdown["values"] = expirations
    exp_var.set(expirations[0])

    # Update table UI
    update_table(exp_var.get())
    stats_btn.config(state="normal")
    messagebox.showinfo("CSV Loaded", f"{symbol} index options loaded successfully.")


def show_timed_message(title, message, duration_ms):
    # Use your existing global root window
    global root  

    dialog = tk.Toplevel(root)
    dialog.title(title)
    dialog.geometry("300x100")
    dialog.resizable(False, False)

    dialog.update_idletasks()

    # Center on screen
    screen_w = dialog.winfo_screenwidth()
    screen_h = dialog.winfo_screenheight()
    win_w = dialog.winfo_width()
    win_h = dialog.winfo_height()

    x = (screen_w // 2) - (win_w // 2)
    y = (screen_h // 2) - (win_h // 2)
    dialog.geometry(f"{win_w}x{win_h}+{x}+{y}")

    ttk.Label(dialog, text=message, wraplength=280).pack(
        expand=True, fill="both", padx=10, pady=10
    )
    dialog.update_idletasks()
    dialog.update()

    dialog.after(duration_ms, dialog.destroy)

    dialog.transient(root)
    dialog.grab_set()
    # root.wait_window(dialog)

exp_data_map = {}
current_symbol = ""
current_price = 0.0

def open_stats_breakdown():
    global exp_data_map, current_price

    selected_exp = exp_var.get()
    if selected_exp not in exp_data_map:
        messagebox.showwarning("No Data", "No data for this expiration.")
        return

    df = exp_data_map[selected_exp]
    if df.empty:
        messagebox.showwarning("Empty", "No option data loaded.")
        return

    # === Replace blanks with zero ===
    df_num = df.replace("", 0.0)

    # === Time to expiration (days extracted from "YYYY-MM-DD:XX") ===
    try:
        exp_date_str = selected_exp.split(":")[0]
        exp_date = datetime.datetime.strptime(exp_date_str, "%Y-%m-%d").date()
        today = datetime.date.today()
        T = max((exp_date - today).days / 365, 1/365)
    except:
        T = 7/365  # fallback

    r = 0.05
    q = 0.015
    CONTRACT_MULT = 100

    # === Black-Scholes Vega ===
    def bs_vega_with_mult(S, K, T, r, q, sigma):
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        vega = S * np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T)
        return vega * CONTRACT_MULT

    # === Compute totals ===
    total_call_oi = df_num["OI_Call"].sum()
    total_put_oi  = df_num["OI_Put"].sum()

    total_call_gamma = df_num["Gamma_Call"].sum()
    total_put_gamma  = df_num["Gamma_Put"].sum()

    total_call_delta = df_num["Delta_Call"].sum()
    total_put_delta  = df_num["Delta_Put"].sum()

    # Real model-based Vega
    call_vega_list = []
    put_vega_list  = []

    for _, row in df_num.iterrows():
        strike = float(row["Strike"])
        iv_call = float(row["IV_Call"])
        iv_put  = float(row["IV_Put"])

        call_vega_list.append(bs_vega_with_mult(current_price, strike, T, r, q, iv_call))
        put_vega_list.append(bs_vega_with_mult(current_price, strike, T, r, q, iv_put))

    total_call_vega = np.sum(call_vega_list)
    total_put_vega  = np.sum(put_vega_list)

    # IV-weighted PCR
    total_call_iv = df_num["IV_Call"].sum()
    total_put_iv  = df_num["IV_Put"].sum()

    # Theta totals
    total_call_theta = df_num["Theta_Call"].sum()
    total_put_theta  = df_num["Theta_Put"].sum()

    def safe_ratio(a, b):
        return (a / b) if b != 0 else 0.0

    # === Standard Ratios ===
    pcr = safe_ratio(total_put_oi, total_call_oi)
    pcr_gamma = safe_ratio(total_put_gamma, total_call_gamma)
    pcr_delta = safe_ratio(total_put_delta, total_call_delta)
    pcr_vega  = safe_ratio(total_put_vega, total_call_vega)

    # === Weighted Ratios ===
    pcr_theta = safe_ratio(total_put_theta, total_call_theta)
    pcr_iv    = safe_ratio(total_put_iv,   total_call_iv)

    weighted_pcr_gamma = safe_ratio(total_put_gamma * total_put_oi,
                                    total_call_gamma * total_call_oi)

    weighted_pcr_vega = safe_ratio(total_put_vega * total_put_oi,
                                   total_call_vega * total_call_oi)

    # === Prepare table data ===
    stats_sections = [
        ("STANDARD RATIOS", [
            ("Put/Call OI Ratio", f"{pcr:.3f}"),
            ("PCR x Gamma",        f"{pcr_gamma:.3f}"),
            ("PCR x Delta",        f"{pcr_delta:.3f}"),
            ("PCR x Vega",         f"{pcr_vega:.3f}"),
        ]),

        ("WEIGHTED RATIOS", [
            ("PCR x Theta",        f"{pcr_theta:.3f}"),
            ("PCR x IV",           f"{pcr_iv:.3f}"),
            ("Weighted PCR Gamma",     f"{weighted_pcr_gamma:.3f}"),
            ("Weighted PCR Vega",  f"{weighted_pcr_vega:.3f}"),
        ])
    ]

    # === Build window ===
    win = tk.Toplevel(root)
    win.title(f"{current_symbol} Stats Breakdown — {selected_exp}")
    win.geometry("520x420")

    header = tk.Label(
        win,
        text=f"{current_symbol} Stats Breakdown\nExpiration: {selected_exp}",
        font=("Arial", 15, "bold")
    )
    header.pack(pady=10)

    table_frame = tk.Frame(win)
    table_frame.pack(pady=10)

    row_index = 0

    # Build table row-by-row
    for section_title, rows in stats_sections:
        # Section header
        tk.Label(
            table_frame,
            text=section_title,
            font=("Arial", 12, "bold"),
            fg="#333"
        ).grid(row=row_index, column=0, columnspan=2, pady=(10, 5))
        row_index += 1

        for metric, value in rows:
            bg_color = "#F4F4F4" if row_index % 2 == 0 else "#FFFFFF"

            tk.Label(
                table_frame,
                text=metric,
                font=("Arial", 11),
                bg=bg_color,
                width=28,
                anchor="w"
            ).grid(row=row_index, column=0, padx=10, pady=2)

            tk.Label(
                table_frame,
                text=value,
                font=("Arial", 11, "bold"),
                bg=bg_color,
                width=12,
                anchor="e"
            ).grid(row=row_index, column=1, padx=10, pady=2)

            row_index += 1



# === Black-Scholes Algorithms (with dividend yield) ===
def bs_gamma(S, K, T, r, q, sigma):
    try:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0
        dp = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        gamma = np.exp(-q*T) * norm.pdf(dp) / (S * sigma * np.sqrt(T))
        return gamma
    except Exception:
        return 0.0
    
def bs_vanna(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S/K) + (r - q + 0.5*sigma*sigma)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-q*T) * norm.pdf(d1) * d2 / sigma

def bs_vega(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S/K) + (r - q + 0.5*sigma*sigma)*T) / (sigma*np.sqrt(T))
    return S * np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T)

def bs_volga(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S/K) + (r - q + 0.5*sigma*sigma)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    vega = bs_vega(S, K, T, r, q, sigma)
    return vega * d1 * d2 / sigma

def bs_charm(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S/K) + (r - q + 0.5*sigma*sigma)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    term1 = q * np.exp(-q*T) * norm.cdf(d1)
    term2 = np.exp(-q*T) * norm.pdf(d1)
    term3 = (2*(r-q)*T - d2*sigma*np.sqrt(T)) / (2*T*sigma*np.sqrt(T))
    return term1 - term2 * term3

EXPOSURE_MODELS = {
    "Gamma":  lambda S,K,T,r,q,sigma: bs_gamma(S,K,T,r,q,sigma),
    "Vanna":  lambda S,K,T,r,q,sigma: bs_vanna(S,K,T,r,q,sigma),
    "Volga":  lambda S,K,T,r,q,sigma: bs_volga(S,K,T,r,q,sigma),
    "Charm":  lambda S,K,T,r,q,sigma: bs_charm(S,K,T,r,q,sigma),
}

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
        # print(f"\n=== DEBUG: Raw OptionChain keys for {symbol} ===")
        # print("Available keys:", list(data.keys()))

        # Print one sample call & put entry if available
        calls = data.get("callExpDateMap", {})
        puts = data.get("putExpDateMap", {})

        # if calls:
        #     first_exp = next(iter(calls))
        #     first_strike = next(iter(calls[first_exp]))
        #     print("\nSample CALL JSON:")
        #     print(calls[first_exp][first_strike][0])
        # else:
        #     print("No calls found.")

        # if puts:
        #     first_exp = next(iter(puts))
        #     first_strike = next(iter(puts[first_exp]))
        #     print("\nSample PUT JSON:")
        #     print(puts[first_exp][first_strike][0])
        # else:
        #     print("No puts found.")
        # print("=== END DEBUG ===\n")
        
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

tk.Label(sidebar_frame, text="Chart Output:", bg="#2b2b2b", fg="white").pack(pady=5)
chart_output_var = tk.StringVar(value="Desktop")
chart_output_dropdown = ttk.Combobox(
    sidebar_frame,
    textvariable=chart_output_var,
    values=["Browser", "Desktop"],
    state="readonly",
    width=10
)
chart_output_dropdown.pack(pady=5)

stats_btn = tk.Button(
    sidebar_frame,
    text="Get Stats Breakdown",
    command=lambda: open_stats_breakdown(),
    state="disabled"
)
stats_btn.pack(pady=5, padx=10, fill=tk.X)

model_btn = tk.Menubutton(sidebar_frame, text="Exposure Model", relief=tk.RAISED)
model_menu = tk.Menu(model_btn, tearoff=0)
model_btn.config(menu=model_menu)

# Submenu: Black-Scholes
bs_menu = tk.Menu(model_menu, tearoff=0)
model_menu.add_cascade(label="Black-Scholes", menu=bs_menu)

# Model variable
selected_model_var = tk.StringVar(value="Gamma")

# Add items
for greek in ["Gamma", "Vanna", "Volga", "Charm"]:
    bs_menu.add_radiobutton(
        label=greek,
        variable=selected_model_var,
        value=greek
    )

model_btn.pack(pady=5, padx=10, fill=tk.X)

tk.Label(sidebar_frame, text="Tools", bg="#2b2b2b", fg="white", font=("Arial", 12, "bold")).pack(pady=10)
update_gamma_btn = tk.Button(sidebar_frame, text="Generate Chart", command=lambda: generate_selected_chart())
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
    show_timed_message("Fetching...", f"Fetched Data for {symbol} complete", 3000)
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
    stats_btn.config(state="normal")

def auto_refresh_price():
    global current_symbol, current_price
    if current_symbol and "(CSV)" not in current_symbol:
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
    if current_symbol and "(CSV)" not in current_symbol:
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

# --- Separator ---
tk.Label(top_frame, text=" | ", font=("Arial", 14, "bold")).pack(side=tk.LEFT, padx=10)

# --- CSV Index Controls ---
tk.Label(top_frame, text="CSV Index:").pack(side=tk.LEFT, padx=5)

index_symbol_var = tk.StringVar()
index_dropdown = ttk.Combobox(
    top_frame,
    textvariable=index_symbol_var,
    state="readonly",
    values=["SPX", "NDX"],
    width=6
)
index_dropdown.pack(side=tk.LEFT, padx=5)
index_dropdown.set("SPX")

# CSV Load Mode Selector
csv_mode_var = tk.StringVar(value="Default File")
csv_mode_dropdown = ttk.Combobox(
    top_frame,
    textvariable=csv_mode_var,
    state="readonly",
    values=["Default File", "Choose CSV File"],
    width=14
)
csv_mode_dropdown.pack(side=tk.LEFT, padx=5)

# Fetch Index Button
fetch_index_btn = tk.Button(
    top_frame,
    text="Fetch CS Index",
    command=lambda: load_csv_index()
)
fetch_index_btn.pack(side=tk.LEFT, padx=5)

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
def generate_selected_chart():
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

    model_name = selected_model_var.get()
    model_func = EXPOSURE_MODELS[model_name]

    for _, row in df.iterrows():
        strike = float(row.get("Strike", 0) or 0)
        if strike <= 0:
            continue

        iv_call = float(row.get("IV_Call", 0) or 0.2)
        iv_put  = float(row.get("IV_Put", 0) or 0.2)
        oi_call = float(row.get("OI_Call", 0) or 1)
        oi_put  = float(row.get("OI_Put", 0) or 1)

        # --- Compute raw Greek values ---
        call_val = model_func(current_price, strike, T, r, q, iv_call)
        put_val  = model_func(current_price, strike, T, r, q, iv_put)

        # -----------------------------
        # APPLY MODEL-SPECIFIC SCALING
        # -----------------------------
        if model_name == "Gamma":
            scale_call = oi_call * CONTRACT_MULTIPLIER * (current_price**2) * 0.01
            scale_put  = oi_put  * CONTRACT_MULTIPLIER * (current_price**2) * 0.01

            exp_call = call_val * scale_call
            exp_put  = -put_val * scale_put   # ONLY Gamma gets sign flip

        elif model_name == "Vanna":
            scale_call = oi_call * CONTRACT_MULTIPLIER * current_price * iv_call
            scale_put  = oi_put  * CONTRACT_MULTIPLIER * current_price * iv_put

            exp_call = abs(call_val * scale_call)
            exp_put  = -abs(put_val * scale_put)    # NO sign flip

        elif model_name == "Volga":
            vega_call = bs_vega(current_price, strike, T, r, q, iv_call)
            vega_put  = bs_vega(current_price, strike, T, r, q, iv_put)

            scale_call = oi_call * vega_call
            scale_put  = oi_put  * vega_put

            exp_call = abs(call_val * scale_call)
            exp_put  = -abs(put_val * scale_put)  # NO sign flip

        elif model_name == "Charm":
            scale_call = oi_call * CONTRACT_MULTIPLIER * current_price
            scale_put  = oi_put  * CONTRACT_MULTIPLIER * current_price

            exp_call = abs(call_val * scale_call)
            exp_put  = -abs(put_val * scale_put)  # NO sign flip

        # Add to output list
        gamma_data.append({"Strike": strike, "Type": "CALL", "Exposure": exp_call})
        gamma_data.append({"Strike": strike, "Type": "PUT",  "Exposure": exp_put})


    ### the scale is currently incorrect, it is just for gamma by default
    df_plot = pd.DataFrame(gamma_data)
    total_gex = df_plot["Exposure"].sum() / 1e9
    df_plot["Exposure_Bn"] = df_plot["Exposure"] / 1e9

    min_strike = df_plot["Strike"].min()
    max_strike = df_plot["Strike"].max()

    chart = (
        alt.Chart(df_plot)
        .mark_bar(size=14)
        .encode(
            x=alt.X("Strike:Q", title="Strike Price", scale=alt.Scale(domain=[min_strike, max_strike])),
            y=alt.Y("Exposure_Bn:Q", title=f"Spot {model_name} Exposure (Bn)", scale=alt.Scale(domainMid=0)),
            color=alt.Color("Type:N", scale=alt.Scale(domain=["CALL", "PUT"], range=["green", "red"]))
        )
        .properties(
            title={
                "text": [f"${current_symbol} {model_name} Exposure ({selected_exp.split(':')[0]})"],
                "subtitle": [
                    f"Total {model_name}: {total_gex:+.2f} Bn | Updated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Black–Scholes Model)"
                ],
            },
            width=850,
            height=450
        )
    )

    html_path = os.path.join(tempfile.gettempdir(), f"{current_symbol}_{selected_exp.replace(':','-')}_gamma.html")
    chart.save(html_path)
    mode = chart_output_var.get()

    if mode == "Browser":
        webbrowser.open(f"file://{html_path}")
        show_timed_message("Chart Created", "Chart opened in browser.", 3000)
        # messagebox.showinfo("Gamma Chart", "Gamma chart opened in browser.")
        return
    elif mode == "Desktop":
        win = tk.Toplevel(root)
        win.title(f"{current_symbol} Chart - {selected_exp}")
        win.geometry("950x700")

        df_calls = df_plot[df_plot["Type"] == "CALL"].sort_values("Strike")
        df_puts  = df_plot[df_plot["Type"] == "PUT"].sort_values("Strike")

        # === Auto-scale using min/max strikes ===
        strikes = df_plot["Strike"].sort_values().unique()
        min_strike = strikes.min()
        max_strike = strikes.max()

        even_start = int(min_strike + (min_strike % 2))    # next even ≥ min_strike
        even_end   = int(max_strike - (max_strike % 2))    # last even ≤ max_strike

        even_ticks = list(range(even_start, even_end + 1, 2))

        unique_strikes = sorted(df_plot["Strike"].unique())
        if len(unique_strikes) > 1:
            spacing = unique_strikes[1] - unique_strikes[0]
        else:
            spacing = 1

        # Dynamic bar width — normalized across different strike densities
        # These proportions make SPX, NDX, SPY, QQQ, TSLA all look consistent
        if spacing <= 1:
            bar_width = spacing * 0.7    # typical equity options
        elif spacing <= 2.5:
            bar_width = spacing * 0.55   # some ETF chains
        elif spacing <= 5:
            bar_width = spacing * 0.35   # SPX/NDX weekly strikes (5-wide)
        else:
            bar_width = spacing * 0.25   # Large increment strikes (25-wide)

        fig = Figure(figsize=(9, 6), dpi=100)
        ax = fig.add_subplot(111)

        # === Plot CALLS (green) ===
        ax.bar(
            df_calls["Strike"],
            df_calls["Exposure_Bn"],
            width=bar_width,
            color="#2ECC71",
            edgecolor="black",
            linewidth=0.6,
            label="CALL"
        )

        ax.bar(
            df_puts["Strike"],
            df_puts["Exposure_Bn"],
            width=bar_width,
            color="#E74C3C",
            edgecolor="black",
            linewidth=0.6,
            label="PUT"
        )
        # Zero line
        ax.axhline(0, color="black", linewidth=1)

        # Labels and title
        ax.set_title(f"{current_symbol} {model_name} Exposure ({selected_exp.split(':')[0]})", fontsize=14)
        fig.suptitle(f"Model: Black–Scholes ({model_name})", fontsize=11, y=0.98)


        ax.set_xlabel("Strike Price", fontsize=12)
        ax.set_ylabel(f"{model_name} Exposure (Bn)", fontsize=12)

        # === X-axis tick marks every 2 strikes ===
        # --- Determine natural strike spacing ---
        unique_strikes = sorted(df_plot["Strike"].unique())
        if len(unique_strikes) > 1:
            spacing = unique_strikes[1] - unique_strikes[0]
        else:
            spacing = 1

        # --- Decide tick interval based on spacing ---
        if spacing <= 1:
            tick_interval = 2
        elif spacing <= 2.5:
            tick_interval = 5
        elif spacing <= 5:
            tick_interval = 10
        else:
            tick_interval = spacing * 2

        # --- Generate clean ticks ---
        min_s = min(unique_strikes)
        max_s = max(unique_strikes)

        # Round to nearest interval
        start_tick = (int(min_s) // tick_interval) * tick_interval
        end_tick   = (int(max_s + tick_interval) // tick_interval) * tick_interval

        tick_list = list(range(start_tick, end_tick + tick_interval, tick_interval))

        ax.set_xticks(tick_list)

        # ax.set_xticks(even_ticks)

        # Avoid scientific notation
        ax.ticklabel_format(style='plain', axis='x')

        # === Apply scaled limits ===
        ax.set_xlim(min_strike, max_strike)

        # Cleaner grid (Altair-like)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        # Legend
        ax.legend()

        # Embed into Tkinter
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, win)
        toolbar.update()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # show_timed_message("Gamma Chart", "Gamma chart opened in new Tkinter window.", 7000)
        return

# === Start Refresh Loops ===
auto_refresh_price()
auto_refresh_options()
root.mainloop()
