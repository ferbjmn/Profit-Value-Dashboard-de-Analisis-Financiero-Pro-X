# -------------------------------------------------------------
#  📊 DASHBOARD FINANCIERO AVANZADO
# -------------------------------------------------------------
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import time

# Configuración global
st.set_page_config(
    page_title="📊 Dashboard Financiero Avanzado",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

# Parámetros editables
Rf = 0.0435   # riesgo libre
Rm = 0.085    # retorno mercado
Tc0 = 0.21    # tasa impositiva por defecto

# Orden de sectores
SECTOR_RANK = {
    "Consumer Defensive": 1,
    "Consumer Cyclical": 2,
    "Healthcare": 3,
    "Technology": 4,
    "Financial Services": 5,
    "Industrials": 6,
    "Communication Services": 7,
    "Energy": 8,
    "Real Estate": 9,
    "Utilities": 10,
    "Basic Materials": 11,
    "Unknown": 99,
}

MAX_TICKERS_PER_CHART = 10

# =============================================================
# FUNCIONES AUXILIARES
# =============================================================
def safe_first(obj):
    if obj is None:
        return None
    if hasattr(obj, "dropna"):
        obj = obj.dropna()
    return obj.iloc[0] if hasattr(obj, "iloc") and not obj.empty else obj

def seek_row(df, keys):
    for k in keys:
        if k in df.index:
            return df.loc[k]
    return pd.Series([0], index=df.columns[:1])

def format_number(x, decimals=2, is_percent=False):
    if pd.isna(x):
        return "N/D"
    if is_percent:
        return f"{x*100:.{decimals}f}%"
    return f"{x:.{decimals}f}"

def calc_ke(beta):
    return Rf + beta * (Rm - Rf)

def calc_kd(interest, debt):
    return interest / debt if debt else 0

def calc_wacc(mcap, debt, ke, kd, t):
    total = (mcap or 0) + (debt or 0)
    return (mcap/total)*ke + (debt/total)*kd*(1-t) if total else None

def cagr4(fin, metric):
    if metric not in fin.index:
        return None
    v = fin.loc[metric].dropna().iloc[:4]
    return (v.iloc[0]/v.iloc[-1])**(1/(len(v)-1))-1 if len(v)>1 and v.iloc[-1] else None

def chunk_df(df, size=MAX_TICKERS_PER_CHART):
    if df.empty:
        return []
    return [df.iloc[i:i+size] for i in range(0, len(df), size)]

def auto_ylim(ax, values, pad=0.10):
    """Ajuste automático del eje Y."""
    if isinstance(values, pd.DataFrame):
        arr = values.to_numpy(dtype="float64")
    else:
        arr = np.asarray(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if vmax == vmin:
        ymin = vmin - abs(vmin)*pad - 1
        ymax = vmax + abs(vmax)*pad + 1
        ax.set_ylim(ymin, ymax)
        return
    if vmin >= 0:
        ymin = 0
        ymax = vmax * (1 + pad)
    elif vmax <= 0:
        ymax = 0
        ymin = vmin * (1 + pad)
    else:
        m = max(abs(vmin), abs(vmax)) * (1 + pad)
        ymin, ymax = -m, m
    ax.set_ylim(ymin, ymax)

def obtener_datos_financieros(tk, Tc_def):
    try:
        tkr = yf.Ticker(tk)
        info = tkr.info
        bs = tkr.balance_sheet
        fin = tkr.financials
        cf = tkr.cashflow
        
        # Datos básicos
        beta = info.get("beta", 1)
        ke = calc_ke(beta)
        
        debt = safe_first(seek_row(bs, ["Total Debt", "Long Term Debt"])) or info.get("totalDebt", 0)
        cash = safe_first(seek_row(bs, [
            "Cash And Cash Equivalents",
            "Cash And Cash Equivalents At Carrying Value",
            "Cash Cash Equivalents And Short Term Investments",
        ]))
        equity = safe_first(seek_row(bs, ["Common Stock Equity", "Total Stockholder Equity"]))

        interest = safe_first(seek_row(fin, ["Interest Expense"]))
        ebt = safe_first(seek_row(fin, ["Ebt", "EBT"]))
        tax_exp = safe_first(seek_row(fin, ["Income Tax Expense"]))
        ebit = safe_first(seek_row(fin, ["EBIT", "Operating Income",
                                       "Earnings Before Interest and Taxes"]))

        kd = calc_kd(interest, debt)
        tax = tax_exp / ebt if ebt else Tc_def
        mcap = info.get("marketCap", 0)
        wacc = calc_wacc(mcap, debt, ke, kd, tax)

        nopat = ebit * (1 - tax) if ebit is not None else None
        invested = (equity or 0) + ((debt or 0) - (cash or 0))
        roic = nopat / invested if (nopat is not None and invested) else None
        
        # CALCULAR CREACIÓN DE VALOR (WACC vs ROIC) en lugar de EVA
        creacion_valor = (roic - wacc) * 100 if all(v is not None for v in (roic, wacc)) else None

        price = info.get("currentPrice")
        fcf = safe_first(seek_row(cf, ["Free Cash Flow"]))
        shares = info.get("sharesOutstanding")
        pfcf = price / (fcf/shares) if (fcf and shares) else None

        # Cálculo de ratios
        current_ratio = info.get("currentRatio")
        quick_ratio = info.get("quickRatio")
        debt_eq = info.get("debtToEquity")
        lt_debt_eq = info.get("longTermDebtToEquity")
        oper_margin = info.get("operatingMargins")
        profit_margin = info.get("profitMargins")
        roa = info.get("returnOnAssets")
        roe = info.get("returnOnEquity")
        
        # Dividendos
        div_yield = info.get("dividendYield")
        payout = info.get("payoutRatio")
        
        # Crecimiento
        revenue_growth = cagr4(fin, "Total Revenue")
        eps_growth = cagr4(fin, "Net Income")
        fcf_growth = cagr4(cf, "Free Cash Flow") or cagr4(cf, "Operating Cash Flow")

        # Obtener datos para la nueva sección de estructura de capital
        # Obtener datos históricos de balance sheet para los últimos 4 años
        balance_sheet_4y = tkr.balance_sheet.iloc[:, :4] if len(tkr.balance_sheet.columns) >= 4 else tkr.balance_sheet
        
        # Extraer datos de estructura de capital
        try:
            total_assets = seek_row(balance_sheet_4y, ["Total Assets"])
            total_liabilities = seek_row(balance_sheet_4y, ["Total Liabilities Net Minority Interest", "Total Liabilities"])
            total_equity = seek_row(balance_sheet_4y, ["Total Equity Gross Minority Interest", "Stockholders Equity"])
            total_debt = seek_row(balance_sheet_4y, ["Total Debt", "Long Term Debt"])
        except:
            total_assets = pd.Series([0] * len(balance_sheet_4y.columns), index=balance_sheet_4y.columns)
            total_liabilities = pd.Series([0] * len(balance_sheet_4y.columns), index=balance_sheet_4y.columns)
            total_equity = pd.Series([0] * len(balance_sheet_4y.columns), index=balance_sheet_4y.columns)
            total_debt = pd.Series([0] * len(balance_sheet_4y.columns), index=balance_sheet_4y.columns)
        
        # Obtener datos para ratios de sostenibilidad de deuda
        income_statement_4y = tkr.financials.iloc[:, :4] if len(tkr.financials.columns) >= 4 else tkr.financials
        cash_flow_4y = tkr.cashflow.iloc[:, :4] if len(tkr.cashflow.columns) >= 4 else tkr.cashflow
        
        try:
            ebitda = seek_row(income_statement_4y, ["EBITDA"])
            interest_expense = seek_row(income_statement_4y, ["Interest Expense"])
            operating_cash_flow = seek_row(cash_flow_4y, ["Operating Cash Flow"])
            capital_expenditure = seek_row(cash_flow_4y, ["Capital Expenditure"])
        except:
            ebitda = pd.Series([0] * len(income_statement_4y.columns), index=income_statement_4y.columns)
            interest_expense = pd.Series([0] * len(income_statement_4y.columns), index=income_statement_4y.columns)
            operating_cash_flow = pd.Series([0] * len(cash_flow_4y.columns), index=cash_flow_4y.columns)
            capital_expenditure = pd.Series([0] * len(cash_flow_4y.columns), index=cash_flow_4y.columns)
        
        # Calcular ratios de sostenibilidad de deuda para el último año
        if len(total_debt) > 0 and len(total_equity) > 0:
            debt_to_equity = total_debt.iloc[0] / total_equity.iloc[0] if total_equity.iloc[0] != 0 else None
        else:
            debt_to_equity = None
            
        if len(ebitda) > 0 and len(interest_expense) > 0 and interest_expense.iloc[0] != 0:
            interest_coverage = ebitda.iloc[0] / abs(interest_expense.iloc[0])
        else:
            interest_coverage = None
            
        if len(total_debt) > 0 and len(ebitda) > 0 and ebitda.iloc[0] != 0:
            debt_to_ebitda = total_debt.iloc[0] / ebitda.iloc[0]
        else:
            debt_to_ebitda = None
            
        if len(operating_cash_flow) > 0 and len(total_debt) > 0 and total_debt.iloc[0] != 0:
            cash_flow_to_debt = operating_cash_flow.iloc[0] / total_debt.iloc[0]
        else:
            cash_flow_to_debt = None
            
        if len(total_debt) > 0 and len(total_assets) > 0 and total_assets.iloc[0] != 0:
            leverage_ratio = total_debt.iloc[0] / total_assets.iloc[0]
        else:
            leverage_ratio = None
            
        # Calcular Free Cash Flow
        if len(operating_cash_flow) > 0 and len(capital_expenditure) > 0:
            free_cash_flow = operating_cash_flow.iloc[0] + capital_expenditure.iloc[0]  # CapEx es negativo
            fcf_to_debt = free_cash_flow / total_debt.iloc[0] if total_debt.iloc[0] != 0 else None
        else:
            free_cash_flow = None
            fcf_to_debt = None

        return {
            "Ticker": tk,
            "Nombre": info.get("longName") or info.get("shortName") or info.get("displayName") or tk,
            "País": info.get("country") or info.get("countryCode") or "N/D",
            "Industria": info.get("industry") or info.get("industryKey") or info.get("industryDisp") or "N/D",
            "Sector": info.get("sector", "Unknown"),
            "Precio": price,
            "P/E": info.get("trailingPE"),
            "P/B": info.get("priceToBook"),
            "P/FCF": pfcf,
            "Dividend Yield %": div_yield,
            "Payout Ratio": payout,
            "ROA": roa,
            "ROE": roe,
            "Current Ratio": current_ratio,
            "Quick Ratio": quick_ratio,
            "Debt/Eq": debt_eq,
            "LtDebt/Eq": lt_debt_eq,
            "Oper Margin": oper_margin,
            "Profit Margin": profit_margin,
            "WACC": wacc,
            "ROIC": roic,
            "Creacion Valor (Wacc vs Roic)": creacion_valor,
            "Revenue Growth": revenue_growth,
            "EPS Growth": eps_growth,
            "FCF Growth": fcf_growth,
            "MarketCap": mcap,
            # Nuevos datos para la sección de estructura de capital
            "Total Assets": total_assets,
            "Total Liabilities": total_liabilities,
            "Total Equity": total_equity,
            "Total Debt": total_debt,
            "EBITDA": ebitda,
            "Interest Expense": interest_expense,
            "Operating Cash Flow": operating_cash_flow,
            "Capital Expenditure": capital_expenditure,
            "Debt to Equity": debt_to_equity,
            "Interest Coverage": interest_coverage,
            "Debt to EBITDA": debt_to_ebitda,
            "Cash Flow to Debt": cash_flow_to_debt,
            "Leverage Ratio": leverage_ratio,
            "FCF to Debt": fcf_to_debt
        }
    except Exception as e:
        st.error(f"Error obteniendo datos para {tk}: {str(e)}")
        return None

# =============================================================
# INTERFAZ PRINCIPAL
# =============================================================
def main():
    st.title("📊 Dashboard de Análisis Financiero Avanzado")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        t_in = st.text_area("Tickers (separados por comas)", 
                          "HRL, AAPL, MSFT, ABT, O, XOM, KO, JNJ, CLX, CHD, CB, DDOG")
        max_t = st.slider("Máximo de tickers", 1, 100, 50)
        
        st.markdown("---")
        st.markdown("**Parámetros WACC**")
        global Rf, Rm, Tc0
        Rf = st.number_input("Tasa libre de riesgo (%)", 0.0, 20.0, 4.35)/100
        Rm = st.number_input("Retorno esperado del mercado (%)", 0.0, 30.0, 8.5)/100
        Tc0 = st.number_input("Tasa impositiva corporativa (%)", 0.0, 50.0, 21.0)/100

    if st.button("🔍 Analizar Acciones", type="primary"):
        tickers = [t.strip().upper() for t in t_in.split(",") if t.strip()][:max_t]
        
        # Obtener datos
        datos = []
        errs = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        with st.spinner("Obteniendo datos financieros..."):
            for i, tk in enumerate(tickers):
                try:
                    status_text.text(f"⏳ Procesando {tk} ({i+1}/{len(tickers)})...")
                    data = obtener_datos_financieros(tk, Tc0)
                    if data:
                        datos.append(data)
                except Exception as e:
                    errs.append({"Ticker": tk, "Error": str(e)})
                progress_bar.progress((i + 1) / len(tickers))
                time.sleep(1)  # Evitar rate limiting

        status_text.text("✅ Análisis completado!")
        time.sleep(0.5)
        status_text.empty()
        progress_bar.empty()

        if not datos:
            st.error("No se pudieron obtener datos para los tickers proporcionados")
            if errs:
                st.table(pd.DataFrame(errs))
            return

        df = pd.DataFrame(datos)
        df["SectorRank"] = df["Sector"].map(SECTOR_RANK).fillna(99).astype(int)
        df = df.sort_values(["SectorRank", "Sector", "Ticker"])
        
        # Formatear valores para visualización
        df_disp = df.copy()
        
        # Columnas con 2 decimales
        for col in ["P/E", "P/B", "P/FCF", "Current Ratio", "Quick Ratio", "Debt/Eq", "LtDebt/Eq"]:
            df_disp[col] = df_disp[col].apply(lambda x: format_number(x, 2))
            
        # Porcentajes con 2 decimales
        for col in ["Dividend Yield %", "Payout Ratio", "ROA", "ROE", "Oper Margin", 
                   "Profit Margin", "WACC", "ROIC", "Revenue Growth", "EPS Growth", "FCF Growth"]:
            df_disp[col] = df_disp[col].apply(lambda x: format_number(x, 2, is_percent=True))
            
        # Creación de Valor con 2 decimales y porcentaje
        df_disp["Creacion Valor (Wacc vs Roic)"] = df_disp["Creacion Valor (Wacc vs Roic)"].apply(
            lambda x: format_number(x/100, 2, is_percent=True) if pd.notnull(x) else "N/D"
        )
            
        # Precio y MarketCap con 2 decimales
        df_disp["Precio"] = df_disp["Precio"].apply(lambda x: f"${float(x):,.2f}" if pd.notnull(x) else "N/D")
        df_disp["MarketCap"] = df_disp["MarketCap"].apply(lambda x: f"${float(x)/1e9:,.2f}B" if pd.notnull(x) else "N/D")
        
        # Formatear nuevos ratios de sostenibilidad de deuda
        for col in ["Debt to Equity", "Interest Coverage", "Debt to EBITDA", 
                   "Cash Flow to Debt", "Leverage Ratio", "FCF to Debt"]:
            df_disp[col] = df_disp[col].apply(lambda x: format_number(x, 2) if pd.notnull(x) else "N/D")
        
        # Asegurar que las columnas de texto no sean None
        for c in ["Nombre", "País", "Industria"]:
            df_disp[c] = df_disp[c].fillna("N/D").replace({None: "N/D", "": "N/D"})

        # =====================================================
        # SECCIÓN 1: RESUMEN GENERAL
        # =====================================================
        st.header("📋 Resumen General (agrupado por Sector)")
        
        # Mostrar tabla con la columna Nombre fijada
        columnas_resumen = [
            "Ticker", "Nombre", "País", "Industria", "Sector",
            "Precio", "P/E", "P/B", "P/FCF",
            "Dividend Yield %", "Payout Ratio", "ROA", "ROE",
            "Current Ratio", "Debt/Eq", "Oper Margin", "Profit Margin",
            "WACC", "ROIC", "Creacion Valor (Wacc vs Roic)", "MarketCap"
        ]
        
        # Configurar el DataFrame para que la columna Nombre esté fija
        st.dataframe(
            df_disp[columnas_resumen],
            use_container_width=True,
            height=500,
            column_config={
                "Nombre": st.column_config.Column(
                    width="medium",
                    help="Nombre completo de la empresa"
                )
            }
        )

        if errs:
            st.subheader("🚫 Tickers con error")
            st.table(pd.DataFrame(errs))

        sectors_ordered = df["Sector"].unique()

        # =====================================================
        # SECCIÓN 2: ANÁLISIS DE VALORACIÓN
        # =====================================================
        st.header("💰 Análisis de Valoración (por Sector)")
        
        for sec in sectors_ordered:
            sec_df = df[df["Sector"] == sec]
            if sec_df.empty:
                continue
                
            with st.expander(f"Sector: {sec} ({len(sec_df)} empresas)", expanded=False):
                fig, ax = plt.subplots(figsize=(10, 4))
                val = sec_df[["Ticker", "P/E", "P/B", "P/FCF"]].set_index("Ticker").apply(pd.to_numeric, errors="coerce")
                val.plot(kind="bar", ax=ax, rot=45)
                ax.set_ylabel("Ratio")
                auto_ylim(ax, val)
                st.pyplot(fig)
                plt.close()

        # =============================================================
        # SECCIÓN 3: RENTABILIDAD Y EFICIENCIA
        # =============================================================
        st.header("📈 Rentabilidad y Eficiencia")

        tabs = st.tabs(["ROE vs ROA", "Márgenes", "WACC vs ROIC"])

        with tabs[0]:
            for sec in sectors_ordered:
                sec_df = df[df["Sector"] == sec]
                if sec_df.empty:
                    continue
                    
                with st.expander(f"Sector: {sec}", expanded=False):
                    fig, ax = plt.subplots(figsize=(10, 5))
                    rr = pd.DataFrame({
                        "ROE": (sec_df["ROE"]*100).values,
                        "ROA": (sec_df["ROA"]*100).values
                    }, index=sec_df["Ticker"])
                    rr.plot(kind="bar", ax=ax, rot=45)
                    ax.set_ylabel("%")
                    auto_ylim(ax, rr)
                    st.pyplot(fig)
                    plt.close()

        with tabs[1]:
            for sec in sectors_ordered:
                sec_df = df[df["Sector"] == sec]
                if sec_df.empty:
                    continue
                    
                with st.expander(f"Sector: {sec}", expanded=False):
                    fig, ax = plt.subplots(figsize=(10, 5))
                    mm = pd.DataFrame({
                        "Oper Margin": (sec_df["Oper Margin"]*100).values,
                        "Profit Margin": (sec_df["Profit Margin"]*100).values
                    }, index=sec_df["Ticker"])
                    mm.plot(kind="bar", ax=ax, rot=45)
                    ax.set_ylabel("%")
                    auto_ylim(ax, mm)
                    st.pyplot(fig)
                    plt.close()

        with tabs[2]:
            # WACC vs ROIC dividido por sectores
            for sec in sectors_ordered:
                sec_df = df[df["Sector"] == sec]
                if sec_df.empty:
                    continue
                    
                with st.expander(f"Sector: {sec} - WACC vs ROIC", expanded=False):
                    fig, ax = plt.subplots(figsize=(12, 6))
                    rw = pd.DataFrame({
                        "ROIC": (sec_df["ROIC"]*100).values,
                        "WACC": (sec_df["WACC"]*100).values
                    }, index=sec_df["Ticker"])
                    rw.plot(kind="bar", ax=ax, rot=45)
                    ax.set_ylabel("%")
                    ax.set_title(f"Creación de Valor: ROIC vs WACC - Sector {sec}")
                    auto_ylim(ax, rw)
                    
                    # Añadir línea de referencia en 0
                    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    
                    # Calcular y mostrar creación/destrucción de valor promedio del sector
                    valor_creado = (sec_df["ROIC"] - sec_df["WACC"]).mean() * 100
                    color = "green" if valor_creado > 0 else "red"
                    ax.axhline(y=valor_creado, color=color, linestyle='--', alpha=0.7, 
                              label=f'Valor creado promedio: {valor_creado:.2f}%')
                    ax.legend()
                    
                    st.pyplot(fig)
                    plt.close()
                    
                    # Mostrar estadísticas del sector
                    st.markdown(f"**Estadísticas del Sector {sec}:**")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_roic = sec_df["ROIC"].mean() * 100
                        avg_wacc = sec_df["WACC"].mean() * 100
                        st.metric("ROIC Promedio", f"{avg_roic:.2f}%")
                        st.metric("WACC Promedio", f"{avg_wacc:.2f}%")
                    
                    with col2:
                        median_roic = sec_df["ROIC"].median() * 100
                        median_wacc = sec_df["WACC"].median() * 100
                        st.metric("ROIC Mediano", f"{median_roic:.2f}%")
                        st.metric("WACC Mediano", f"{median_wacc:.2f}%")
                    
                    with col3:
                        empresas_creadoras = (sec_df["ROIC"] > sec_df["WACC"]).sum()
                        total_empresas = len(sec_df)
                        porcentaje_creadoras = (empresas_creadoras / total_empresas) * 100
                        st.metric("Empresas que crean valor", f"{empresas_creadoras}/{total_empresas} ({porcentaje_creadoras:.1f}%)")

        # =====================================================
        # NUEVA SECCIÓN: ESTRUCTURA DE CAPITAL Y SOSTENIBILIDAD DE DEUDA
        # =====================================================
        st.header("🏦 Estructura de Capital y Sostenibilidad de Deuda")
        
        # Subsección 1: Evolución de Activos, Pasivos y Patrimonio por Sector
        st.subheader("Evolución de Activos, Pasivos y Patrimonio Neto (por Sector)")
        
        for sec in sectors_ordered:
            sec_df = df[df["Sector"] == sec]
            if sec_df.empty:
                continue
                
            with st.expander(f"Sector: {sec} - Estructura de Capital", expanded=False):
                for i, chunk in enumerate(chunk_df(sec_df), 1):
                    st.caption(f"Bloque {i}")
                    
                    # Crear gráficos para cada empresa en el chunk
                    for _, empresa in chunk.iterrows():
                        st.markdown(f"**{empresa['Ticker']} - {empresa['Nombre']}**")
                        
                        # Obtener datos para los últimos 4 años
                        total_assets = empresa["Total Assets"]
                        total_liabilities = empresa["Total Liabilities"]
                        total_equity = empresa["Total Equity"]
                        
                        if total_assets.empty or total_liabilities.empty or total_equity.empty:
                            st.warning("Datos insuficientes para mostrar la estructura de capital")
                            continue
                        
                        # Crear gráfico de barras
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Preparar datos para el gráfico
                        years = [col.strftime('%Y') for col in total_assets.index]
                        x_pos = np.arange(len(years))
                        width = 0.25
                        
                        # Graficar barras
                        bars1 = ax.bar(x_pos - width, total_assets.values / 1e6, width, label='Activos Totales', color='#0074D9')
                        bars2 = ax.bar(x_pos, total_liabilities.values / 1e6, width, label='Pasivos Totales', color='#FF4136')
                        bars3 = ax.bar(x_pos + width, total_equity.values / 1e6, width, label='Patrimonio Neto', color='#2ECC40')
                        
                        # Configurar el gráfico
                        ax.set_xlabel('Año')
                        ax.set_ylabel('Millones USD')
                        ax.set_title(f'Estructura de Capital - {empresa["Ticker"]}')
                        ax.set_xticks(x_pos)
                        ax.set_xticklabels(years)
                        ax.legend()
                        
                        # Añadir valores en las barras
                        for bars in [bars1, bars2, bars3]:
                            for bar in bars:
                                height = bar.get_height()
                                ax.annotate(f'{height:,.0f}',
                                            xy=(bar.get_x() + bar.get_width() / 2, height),
                                            xytext=(0, 3),
                                            textcoords="offset points",
                                            ha='center', va='bottom', fontsize=8)
                        
                        st.pyplot(fig)
                        plt.close()
        
        # Subsección 2: Ratios de Sostenibilidad de Deuda
        st.subheader("Ratios de Sostenibilidad de Deuda (por Sector)")
        
        # Definir umbrales para los ratios
        umbrales = {
            'Debt to Equity': 1.0,
            'Interest Coverage': 3.0,
            'Debt to EBITDA': 3.0,
            'Cash Flow to Debt': 0.2,
            'FCF to Debt': 0.1
        }
        
        for sec in sectors_ordered:
            sec_df = df[df["Sector"] == sec]
            if sec_df.empty:
                continue
                
            with st.expander(f"Sector: {sec} - Sostenibilidad de Deuda", expanded=False):
                # Mostrar tabla de ratios
                ratios_df = sec_df[["Ticker", "Debt to Equity", "Interest Coverage", 
                                  "Debt to EBITDA", "Cash Flow to Debt", "FCF to Debt"]].copy()
                
                # Calcular evaluación de sostenibilidad
                evaluacion = []
                for _, row in ratios_df.iterrows():
                    sostenible = all([
                        row.get("Debt to Equity", float('inf')) < umbrales['Debt to Equity'],
                        row.get("Interest Coverage", 0) > umbrales['Interest Coverage'],
                        row.get("Debt to EBITDA", float('inf')) < umbrales['Debt to EBITDA'],
                        row.get("Cash Flow to Debt", 0) > umbrales['Cash Flow to Debt'],
                        row.get("FCF to Debt", 0) > umbrales['FCF to Debt']
                    ])
                    evaluacion.append("✅ Sostenible" if sostenible else "❌ No Sostenible")
                
                ratios_df["Evaluación"] = evaluacion
                st.dataframe(ratios_df.set_index("Ticker"), use_container_width=True)
                
                # Gráfico de ratios comparativos
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.flatten()
                
                ratios = ["Debt to Equity", "Interest Coverage", "Debt to EBITDA", 
                         "Cash Flow to Debt", "FCF to Debt"]
                
                for i, ratio in enumerate(ratios):
                    if i < len(axes):
                        ax = axes[i]
                        values = sec_df[ratio].values
                        tickers = sec_df["Ticker"].values
                        
                        bars = ax.bar(tickers, values)
                        ax.set_title(ratio)
                        ax.tick_params(axis='x', rotation=45)
                        
                        # Añadir línea de umbral
                        if ratio in umbrales:
                            ax.axhline(y=umbrales[ratio], color='r', linestyle='--', alpha=0.7)
                
                # Ocultar el último subplot si no se uses
                if len(ratios) < len(axes):
                    axes[-1].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        # =====================================================
        # SECCIÓN 5: CRECIMIENTO
        # =====================================================
        st.header("🚀 Crecimiento (CAGR 3-4 años, por sector)")
        
        for sec in sectors_ordered:
            sec_df = df[df["Sector"] == sec]
            if sec_df.empty:
                continue
                
            with st.expander(f"Sector: {sec}", expanded=False):
                for i, chunk in enumerate(chunk_df(sec_df), 1):
                    st.caption(f"Bloque {i}")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    gdf = pd.DataFrame({
                        "Revenue Growth": (chunk["Revenue Growth"]*100).values,
                        "EPS Growth": (chunk["EPS Growth"]*100).values,
                        "FCF Growth": (chunk["FCF Growth"]*100).values
                    }, index=chunk["Ticker"])
                    gdf.plot(kind="bar", ax=ax, rot=45)
                    ax.axhline(0, color="black", linewidth=0.8)
                    ax.set_ylabel("%")
                    auto_ylim(ax, gdf)
                    st.pyplot(fig)
                    plt.close()

        # =====================================================
        # SECCIÓN 6: ANÁLISIS INDIVIDUAL
        # =====================================================
        st.header("🔍 Análisis por Empresa")
        pick = st.selectbox("Selecciona empresa", df_disp["Ticker"].unique())
        det_disp = df_disp[df_disp["Ticker"] == pick].iloc[0]
        det_raw = df[df["Ticker"] == pick].iloc[0]

        st.markdown(f"""
        **{det_raw['Nombre']}**  
        **Sector:** {det_raw['Sector']}  
        **País:** {det_raw['País']}  
        **Industria:** {det_raw['Industria']}
        """)

        cA, cB, cC = st.columns(3)
        with cA:
            st.metric("Precio", det_disp["Precio"])
            st.metric("P/E", det_disp["P/E"])
            st.metric("P/B", det_disp["P/B"])
            st.metric("P/FCF", det_disp["P/FCF"])
            
        with cB:
            st.metric("Market Cap", det_disp["MarketCap"])
            st.metric("ROIC", det_disp["ROIC"])
            st.metric("WACC", det_disp["WACC"])
            st.metric("Creación Valor", det_disp["Creacion Valor (Wacc vs Roic)"])
            
        with cC:
            st.metric("ROE", det_disp["ROE"])
            st.metric("Dividend Yield", det_disp["Dividend Yield %"])
            st.metric("Current Ratio", det_disp["Current Ratio"])
            st.metric("Debt/Eq", det_disp["Debt/Eq"])

        # Nueva sección de análisis individual de estructura de capital
        st.subheader("🏦 Análisis de Estructura de Capital y Sostenibilidad de Deuda")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Ratios de Sostenibilidad de Deuda**")
            ratios_data = {
                "Ratio": ["Debt to Equity", "Interest Coverage", "Debt to EBITDA", 
                         "Cash Flow to Debt", "FCF to Debt"],
                "Valor": [
                    det_disp["Debt to Equity"],
                    det_disp["Interest Coverage"],
                    det_disp["Debt to EBITDA"],
                    det_disp["Cash Flow to Debt"],
                    det_disp["FCF to Debt"]
                ],
                "Umbral": [
                    umbrales["Debt to Equity"],
                    umbrales["Interest Coverage"],
                    umbrales["Debt to EBITDA"],
                    umbrales["Cash Flow to Debt"],
                    umbrales["FCF to Debt"]
                ],
                "Evaluación": [
                    "✅" if pd.notnull(det_raw["Debt to Equity"]) and det_raw["Debt to Equity"] < umbrales["Debt to Equity"] else "❌",
                    "✅" if pd.notnull(det_raw["Interest Coverage"]) and det_raw["Interest Coverage"] > umbrales["Interest Coverage"] else "❌",
                    "✅" if pd.notnull(det_raw["Debt to EBITDA"]) and det_raw["Debt to EBITDA"] < umbrales["Debt to EBITDA"] else "❌",
                    "✅" if pd.notnull(det_raw["Cash Flow to Debt"]) and det_raw["Cash Flow to Debt"] > umbrales["Cash Flow to Debt"] else "❌",
                    "✅" if pd.notnull(det_raw["FCF to Debt"]) and det_raw["FCF to Debt"] > umbrales["FCF to Debt"] else "❌"
                ]
            }
            
            ratios_df = pd.DataFrame(ratios_data)
            st.dataframe(ratios_df, hide_index=True, use_container_width=True)
            
            # Evaluación general
            sostenible = all([
                pd.notnull(det_raw["Debt to Equity"]) and det_raw["Debt to Equity"] < umbrales["Debt to Equity"],
                pd.notnull(det_raw["Interest Coverage"]) and det_raw["Interest Coverage"] > umbrales["Interest Coverage"],
                pd.notnull(det_raw["Debt to EBITDA"]) and det_raw["Debt to EBITDA"] < umbrales["Debt to EBITDA"],
                pd.notnull(det_raw["Cash Flow to Debt"]) and det_raw["Cash Flow to Debt"] > umbrales["Cash Flow to Debt"],
                pd.notnull(det_raw["FCF to Debt"]) and det_raw["FCF to Debt"] > umbrales["FCF to Debt"]
            ])
            
            if sostenible:
                st.success("✅ La deuda de esta empresa es SOSTENIBLE según los ratios analizados")
            else:
                st.error("❌ La deuda de esta empresa podría NO SER SOSTENIBLE")
        
        with col2:
            st.markdown("**Evolución de la Estructura de Capital**")
            
            # Gráfico de estructura de capital
            if not det_raw["Total Assets"].empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                years = [col.strftime('%Y') for col in det_raw["Total Assets"].index]
                x_pos = np.arange(len(years))
                width = 0.25
                
                bars1 = ax.bar(x_pos - width, det_raw["Total Assets"].values / 1e6, width, label='Activos Totales', color='#0074D9')
                bars2 = ax.bar(x_pos, det_raw["Total Liabilities"].values / 1e6, width, label='Pasivos Totales', color='#FF4136')
                bars3 = ax.bar(x_pos + width, det_raw["Total Equity"].values / 1e6, width, label='Patrimonio Neto', color='#2ECC40')
                
                ax.set_xlabel('Año')
                ax.set_ylabel('Millones USD')
                ax.set_title(f'Estructura de Capital - {det_raw["Ticker"]}')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(years)
                ax.legend()
                
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("Datos insuficientes para mostrar la estructura de capital")

        st.subheader("ROIC vs WACC")
        if pd.notnull(det_raw["ROIC"]) and pd.notnull(det_raw["WACC"]):
            fig, ax = plt.subplots(figsize=(5, 4))
            comp = pd.DataFrame({
                "ROIC": [det_raw["ROIC"]*100],
                "WACC": [det_raw["WACC"]*100]
            }, index=[pick])
            comp.plot(kind="bar", ax=ax, rot=0, legend=False, 
                     color=["green" if det_raw["ROIC"] > det_raw["WACC"] else "red", "gray"])
            ax.set_ylabel("%")
            auto_ylim(ax, comp)
            st.pyplot(fig)
            plt.close()
            
            if det_raw["ROIC"] > det_raw["WACC"]:
                st.success("✅ Crea valor (ROIC > WACC)")
            else:
                st.error("❌ Destruye valor (ROIC < WACC)")
        else:
            st.warning("Datos insuficientes para comparar ROIC/WACC")

if __name__ == "__main__":
    main()
