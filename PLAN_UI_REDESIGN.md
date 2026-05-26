# Rediseño UI — Global Steel Designer

> **Para Hermes:** Usar subagent-driven-development para implementar este plan.

**Goal:** Rediseño completo de la interfaz de `app.py` con layout 2-columnas, sistema visual profesional, inputs agrupados en secciones, y resultados mejorados con color-coding.

**Architecture:** Una sola columna izquierda (35%) con inputs agrupados en tarjetas expandibles + columna derecha (65%) con dashboard de resultados. Sidebar reducida a configuración esencial. CSS inline con paleta de ingeniería.

**Tech Stack:** Streamlit, Plotly, Pandas (ya instalados). Solo se modifica `app.py`.

---

### Tareas (6 tareas, archivo único app.py)

#### Task 1: CSS y sistema visual

**Objective:** Reemplazar el CSS actual con un tema profesional de ingeniería estructural.

**File:** `app.py` (reemplazar bloque CSS en líneas ~30-31)

**Implementación:**
- Paleta principal: azul ingenieril `#1a365d`, acento naranja `#ed8936`, verde pass `#38a169`, rojo fail `#e53e3e`
- Sidebar: gradiente oscuro `#0f1f38` → `#1a365d`
- Cards: `.input-card` con borde izquierdo azul, fondo blanco, sombra sutil
- Métricas: `.metric-card` con icono, valor grande, unidad pequeña, color condicional
- Tablas: `.results-table` con filas alternadas, color D/C ratio condicional
- Botón calcular: full-width, gradiente azul, texto blanco
- Header: gradiente, título + subtítulo

```python
st.markdown("""
<style>
/* === SIDEBAR === */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1f38 0%, #1a365d 100%);
}
[data-testid="stSidebar"] * {color: #cbd5e0 !important;}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {color: #63b3ed !important;}
[data-testid="stSidebar"] hr {border-color: #2d4a6e !important;}
[data-testid="stSidebar"] .stSelectbox label, 
[data-testid="stSidebar"] .stRadio label {color: #a0aec0 !important;}

/* === HEADER === */
.app-header {
    background: linear-gradient(135deg, #1a365d, #2b6cb0);
    padding: 24px 32px; border-radius: 12px; margin-bottom: 24px;
}
.app-header h1 {color: white !important; font-size: 1.6rem; margin: 0;}
.app-header p {color: #90cdf4 !important; margin: 6px 0 0; font-size: 0.85rem;}

/* === INPUT CARDS === */
.input-card {
    background: #ffffff; border-left: 4px solid #3182ce;
    border-radius: 6px 10px 10px 6px; padding: 16px 20px;
    margin-bottom: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.input-card h3 {
    color: #1a365d !important; font-size: 0.95rem; margin: 0 0 10px 0;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.input-card label {font-size: 0.82rem !important; color: #4a5568 !important;}

/* === METRIC CARDS === */
.metric-row {display: flex; gap: 14px; margin-bottom: 18px;}
.metric-card {
    flex: 1; background: #ffffff; border-radius: 10px;
    padding: 16px 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    text-align: center;
}
.metric-card .label {font-size: 0.72rem; color: #718096; text-transform: uppercase; letter-spacing: 0.6px;}
.metric-card .value {font-size: 1.35rem; font-weight: 700; color: #1a365d;}
.metric-card .unit {font-size: 0.75rem; color: #a0aec0; margin-left: 4px;}
.metric-card.pass {border-top: 3px solid #38a169;}
.metric-card.fail {border-top: 3px solid #e53e3e;}
.metric-card.warn {border-top: 3px solid #ed8936;}

/* === RESULTS === */
.results-section {margin-top: 18px;}
.results-section h2 {color: #1a365d; font-size: 1.15rem; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px;}

/* === BUTTON === */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #1a365d, #2b6cb0) !important;
    color: white !important; border: none !important; width: 100% !important;
    font-weight: 600 !important; padding: 12px !important; border-radius: 8px !important;
    font-size: 1rem !important; transition: all 0.2s;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #2b6cb0, #3182ce) !important;
    box-shadow: 0 4px 12px rgba(49,130,206,0.3);
}

/* === TABLAS === */
[data-testid="stDataFrame"] {font-size: 0.82rem !important;}

/* === FOOTER === */
.app-footer {
    margin-top: 28px; padding: 12px 20px;
    background: #f7fafc; border-radius: 8px; text-align: center;
    font-size: 0.78rem; color: #718096;
}
</style>
""", unsafe_allow_html=True)
```

#### Task 2: Header y sidebar simplificada

**Objective:** Añadir header profesional y simplificar sidebar solo a configuración esencial.

**File:** `app.py` (reemplazar st.sidebar actual ~líneas 32-38 y añadir header)

**Implementación:**
- Sidebar: título, units, engine mode, strictness. (Quitar Family y Source de sidebar — van al panel de inputs)
- Header: `app-header` div con título "🌍 Global Steel Designer" + subtítulo "AISC 360-22 · LRFD · Master Engine v2 + v8"

```python
# ── HEADER ──
st.markdown("""
<div class="app-header">
  <h1>🌍 Global Steel Designer</h1>
  <p>AISC 360-22 &nbsp;|&nbsp; LRFD &nbsp;|&nbsp; Master Engine v2 + v8 &nbsp;|&nbsp; 
     I-Shape · Channel · Angle · Tee</p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    units = st.selectbox("Units", ["kN, mm, MPa", "kip, in, ksi"])
    engine_mode = st.radio("Engine mode", ["best_effort", "strict"])
    strictness = Strictness.STRICT if st.radio(
        "Normative strictness", ["STRICT", "PRACTICAL"]
    ) == "STRICT" else Strictness.PRACTICAL
    st.divider()
    st.caption("AISC 360-22 · LRFD · v2.0")
```

#### Task 3: Panel de inputs agrupado en tarjetas

**Objective:** Reorganizar todos los inputs en tarjetas colapsables con secciones lógicas.

**File:** `app.py` (reemplazar líneas 40-90, el `left` column completo + mover Family/Source aquí)

**Layout del panel izquierdo:**

```
┌─ GEOMETRÍA ──────────────────────────┐
│ Family: [I Shape ▾]  Source: [Table ▾]│
│ Section: [W18X35 ▾]                   │
│ (o dims geométricas si Geometric)     │
└───────────────────────────────────────┘
┌─ MATERIAL ───────────────────────────┐
│ Fy (MPa): [345]  Fu (MPa): [450]      │
└───────────────────────────────────────┘
┌─ LONGITUDES ─────────────────────────┐
│ Lx: [6000]  Ly: [6000]  Lb: [3000]   │
│ Kx: [1.0]  Ky: [1.0]  Cb: [1.0]     │
└───────────────────────────────────────┘
┌─ CARGAS ─────────────────────────────┐
│ Pu: [150]  Tu: [0]  Mux: [250]       │
│ Muy: [0]   Vux: [80]                 │
│ U: [1.0]  Hole area: [0]             │
└───────────────────────────────────────┘
┌─ AVANZADO (colapsable) ──────────────┐
│ a stiffeners, stiffeners checkbox,    │
│ TFA checkbox, e_x, e_y,              │
│ Block shear toggle + inputs          │
│ Stem in tension checkbox             │
└───────────────────────────────────────┘
[       🧮 CALCULAR       ]
```

**Implementación con `st.expander` o `st.markdown` con secciones HTML:**
Usar `st.markdown('<div class="input-card">...')` para cada sección, con título h3 y `st.columns` para campos en línea.

```python
left, right = st.columns([0.35, 0.65])

with left:
    # ── GEOMETRÍA ──
    st.markdown('<div class="input-card"><h3>📐 GEOMETRÍA</h3>', unsafe_allow_html=True)
    family = st.selectbox("Family", ["I Shape", "Channel", "Angle", "Tee"])
    source = st.selectbox("Source", ["Table", "Geometric"])
    # ... section selection or geometric dims ...
    
    filters = {"I Shape": ["W","M","S","HP","IPE","IPN","HEA","HEB","HEM","IN","HN"],
               "Channel": ["C","MC"], "Tee": ["WT","MT","ST"], "Angle": []}
    sec_in = None; row = None
    if source == "Table" and family != "Angle":
        cands = df[df["family"].astype(str).isin(filters[family])]
        name = st.selectbox("Section", cands["name"].tolist())
        row = cands[cands["name"] == name].iloc[0]
    else:
        if family == "I Shape":
            d = inp(units, 300.0, 11.81, "d (mm)", "d (in)")
            bf = inp(units, 150.0, 5.91, "bf (mm)", "bf (in)")
            tf = inp(units, 12.0, 0.47, "tf (mm)", "tf (in)")
            tw = inp(units, 8.0, 0.31, "tw (mm)", "tw (in)")
            if units == "kip, in, ksi": d,bf,tf,tw = d*25.4,bf*25.4,tf*25.4,tw*25.4
            sec_in = SectionPropertyCalculator.i_shape(IShapeDims(d=d,bf=bf,tf=tf,tw=tw,name="Custom I"))
        elif family == "Angle":
            d = inp(units, 150.0, 5.91, "d (mm)", "d (in)")
            b = inp(units, 100.0, 3.94, "b (mm)", "b (in)")
            t = inp(units, 10.0, 0.39, "t (mm)", "t (in)")
            if units == "kip, in, ksi": d,b,t = d*25.4,b*25.4,t*25.4
            sec_in = SectionPropertyCalculator.angle(AngleDims(d=d,b=b,t=t,name="Custom L"))
        else:
            st.warning("Use table for this family.")
            st.stop()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ── MATERIAL ──
    st.markdown('<div class="input-card"><h3>🧪 MATERIAL</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: Fy = inp(units, 345.0, 50.0, "Fy (MPa)", "Fy (ksi)")
    with c2: Fu = inp(units, 450.0, 65.0, "Fu (MPa)", "Fu (ksi)")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ── LONGITUDES ──
    st.markdown('<div class="input-card"><h3>📏 LONGITUDES &amp; COEFICIENTES</h3>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: Lx = inp(units, 6000.0, 236.22, "Lx (mm)", "Lx (in)")
    with c2: Ly = inp(units, 6000.0, 236.22, "Ly (mm)", "Ly (in)")
    with c3: Lb = inp(units, 3000.0, 118.11, "Lb (mm)", "Lb (in)")
    c1, c2, c3 = st.columns(3)
    with c1: Kx = st.number_input("Kx", value=1.0)
    with c2: Ky = st.number_input("Ky", value=1.0)
    with c3: Cb = st.number_input("Cb", value=1.0)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ── CARGAS ──
    st.markdown('<div class="input-card"><h3>⚡ CARGAS &amp; DEMANDA</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        Pu = inp(units, 150.0, 33.72, "Pu (kN)", "Pu (kip)")
        Tu = inp(units, 0.0, 0.0, "Tu (kN)", "Tu (kip)")
        Mux = inp(units, 250.0, 184.39, "Mux (kN·m)", "Mux (kip-ft)")
    with c2:
        Muy = inp(units, 0.0, 0.0, "Muy (kN·m)", "Muy (kip-ft)")
        Vux = inp(units, 80.0, 17.98, "Vux (kN)", "Vux (kip)")
    c1, c2 = st.columns(2)
    with c1: U = st.number_input("U (shear lag)", value=1.0, min_value=0.0, max_value=1.0)
    with c2: holes = inp(units, 0.0, 0.0, "Hole area (mm²)", "Hole area (in²)")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ── AVANZADO ──
    with st.expander("🔧 ADVANCED OPTIONS"):
        c1, c2 = st.columns(2)
        with c1: a = inp(units, 0.0, 0.0, "a stiffeners (mm)", "a stiffeners (in)")
        with c2: 
            stiff = st.checkbox("Stiffeners", value=False)
            tfa = st.checkbox("Tension field action", value=False)
        c1, c2 = st.columns(2)
        with c1: eccx = inp(units, 0.0, 0.0, "e_x (mm)", "e_x (in)")
        with c2: eccy = inp(units, 0.0, 0.0, "e_y (mm)", "e_y (in)")
        stem_in_tension = st.checkbox("Stem in tension (tees)", value=True)
        block_enabled = st.checkbox("Evaluate block shear", value=False)
        if block_enabled:
            c1, c2 = st.columns(2)
            with c1:
                Avg = st.number_input("Avg (mm²)", value=1200.0)
                Avn = st.number_input("Avn (mm²)", value=1000.0)
            with c2:
                Atg = st.number_input("Atg (mm²)", value=500.0)
                Atn = st.number_input("Atn (mm²)", value=420.0)
            block_vals = {"Avg": Avg, "Avn": Avn, "Atg": Atg, "Atn": Atn, "Ubs": 1.0}
        else:
            block_vals = {"Avg": 0.0, "Avn": 0.0, "Atg": 0.0, "Atn": 0.0, "Ubs": 1.0}
    
    st.button("🧮 CALCULAR", type="primary", use_container_width=True)
```

#### Task 4: Panel de resultados con métricas mejoradas

**Objective:** Reemplazar las 4 métricas simples con KPI cards estilizadas y color-coding condicional.

**File:** `app.py` (reemplazar líneas 122-165, el bloque `mid` + `right`)

**Implementación:**
- Columnas 2-column: métricas arriba, tabs abajo con color-coding en tablas
- P-M diagram abajo de los tabs (ancho completo)
- Tablas de resultados con ratio coloreado (verde ≤0.9, naranja 0.9-1.0, rojo >1.0)
- Croquis de sección básico con Plotly

```python
with right:
    if err:
        st.error(f"❌ Error: {err}")
        st.stop()
    elif result is not None:
        inter = result.get("interaction_ratio", 0.0)
        passed = inter <= 1.0
        
        # ── MÉTRICAS ──
        status_class = "pass" if passed else "fail"
        ratio_color = "#38a169" if inter <= 0.9 else ("#ed8936" if inter <= 1.0 else "#e53e3e")
        
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="label">Section</div>
                <div class="value">{getattr(section, 'name', '-')}</div>
            </div>
            <div class="metric-card">
                <div class="label">Family</div>
                <div class="value">{family}</div>
            </div>
            <div class="metric-card {status_class}">
                <div class="label">H1 Interaction Ratio</div>
                <div class="value" style="color:{ratio_color}">{inter:.3f}</div>
            </div>
            <div class="metric-card {status_class}">
                <div class="label">Status</div>
                <div class="value" style="color:{ratio_color}">{'✅ PASS' if passed else '❌ FAIL'}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # ── TABS ──
        tabs = st.tabs(["📊 Results", "📋 Audit", "📐 Geometry", "⚠️ Gaps"])
        
        with tabs[0]:
            for key, title, moment in [
                ("tension", "Tension", False),
                ("compression", "Compression", False),
                ("flexure_major", "Flexure Major", True),
                ("flexure_minor", "Flexure Minor", True),
                ("flexure", "Flexure", True),
                ("shear_major", "Shear Major", False),
                ("shear", "Shear", False),
            ]:
                bundle = result.get(key)
                if bundle is not None:
                    st.subheader(title)
                    x = bundle_df(bundle, moment=moment)
                    if not x.empty:
                        # Color-code ratio column
                        def color_ratio(val):
                            try:
                                v = float(val)
                                if v <= 0.9: return 'color: #38a169'
                                elif v <= 1.0: return 'color: #ed8936'
                                else: return 'color: #e53e3e; font-weight: bold'
                            except: return ''
                        styled = x.style.applymap(color_ratio, subset=['Ratio'])
                        st.dataframe(styled, hide_index=True, use_container_width=True)
                    for note in getattr(bundle, "notes", []):
                        st.info(note)
        
        with tabs[1]:
            audit = result.get("audit_report")
            if audit is not None and audit.items:
                st.dataframe(pd.DataFrame(audit.as_dicts()), hide_index=True, use_container_width=True)
                if audit.has_blockers:
                    st.error("Blocking exact-data branches detected.")
                elif audit.has_warnings:
                    st.warning("Warnings detected for approximate or geometric idealized logic.")
            else:
                st.success("No audit flags detected.")
        
        with tabs[2]:
            st.json(result.get("geometry_source", {}))
        
        with tabs[3]:
            rows = [{"Family": i.family, "Chapter": i.chapter, "Description": i.description,
                     "Required Inputs": ", ".join(i.required_inputs)} 
                    for i in normative_gap_registry_v2()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        
        # ── P-M DIAGRAM + PROPIEDADES ──
        st.markdown('<div class="results-section"><h2>📈 P-M Interaction Diagram</h2></div>', unsafe_allow_html=True)
        
        pm_col1, pm_col2 = st.columns([1.5, 1])
        
        with pm_col1:
            pc = result["compression"].controlling.design_strength if result.get("compression") and result["compression"].controlling else 0.0
            mc = result["flexure_major"].controlling.design_strength if result.get("flexure_major") and result["flexure_major"].controlling else (
                result["flexure"].controlling.design_strength if result.get("flexure") and result["flexure"].controlling else 0.0
            )
            fig = go.Figure()
            pc_kn = n_to_kn(pc)
            mc_knm = nmm_to_knm(mc)
            # Better P-M curve shape
            pm_x = [0, mc_knm*0.3, mc_knm*0.65, mc_knm*0.9, mc_knm, mc_knm*0.9, mc_knm*0.65, mc_knm*0.3, 0]
            pm_y = [pc_kn, pc_kn*0.92, pc_kn*0.72, pc_kn*0.38, 0, -pc_kn*0.38, -pc_kn*0.72, -pc_kn*0.92, -pc_kn]
            fig.add_scatter(x=pm_x, y=pm_y, mode='lines', name='φPn-Mn',
                           fill='toself', fillcolor='rgba(49,130,206,0.12)',
                           line=dict(color='#2b6cb0', width=2.5))
            fig.add_scatter(x=[nmm_to_knm(vals["Mux"])], y=[n_to_kn(vals["Pu"])],
                           mode='markers', marker=dict(size=12, color='#e53e3e', symbol='x-thin'),
                           name='Demand (Pu, Mux)')
            fig.update_layout(
                height=350, margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="M (kN·m)", yaxis_title="P (kN)",
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with pm_col2:
            st.markdown('<div class="input-card"><h3>📋 SECTION PROPERTIES</h3>', unsafe_allow_html=True)
            st.dataframe(props_df(section), hide_index=True, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # ── FOOTER ──
    st.markdown("""
    <div class="app-footer">
        Global Steel Designer v2.0 · AISC 360-22 · LRFD · Master Engine<br>
        ⚠️ For engineering verification only. Not a substitute for professional judgment.
    </div>
    """, unsafe_allow_html=True)
```

#### Task 5: Añadir croquis de sección con Plotly

**Objective:** Mostrar un croquis esquemático de la sección seleccionada usando Plotly shapes.

**File:** `app.py` (añadir función `plot_section_sketch` y llamarla en results)

**Implementación:**
```python
def plot_section_sketch(sec, family_name):
    """Draw a schematic cross-section using Plotly shapes."""
    fig = go.Figure()
    fig.update_layout(
        height=300, showlegend=False,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=10),
    )
    
    d = getattr(sec, 'd', 300)
    bf = getattr(sec, 'bf', 150)
    tf = getattr(sec, 'tf', 12)
    tw = getattr(sec, 'tw', 8)
    
    if family_name in ("I Shape", "Tee"):
        # Web
        fig.add_shape(type="rect", x0=-tw/2, y0=tf if family_name=="Tee" else tf,
                      x1=tw/2, y1=d-tf if family_name=="I Shape" else d,
                      line=dict(color="#3182ce", width=1.5),
                      fillcolor="#ebf4ff")
        # Top flange
        fig.add_shape(type="rect", x0=-bf/2, y0=d-tf if family_name=="Tee" else d-tf,
                      x1=bf/2, y1=d if family_name=="Tee" else d,
                      line=dict(color="#3182ce", width=1.5),
                      fillcolor="#bee3f8")
        if family_name == "I Shape":
            # Bottom flange
            fig.add_shape(type="rect", x0=-bf/2, y0=0, x1=bf/2, y1=tf,
                          line=dict(color="#3182ce", width=1.5),
                          fillcolor="#bee3f8")
    elif family_name == "Channel":
        fig.add_shape(type="rect", x0=-tw/2, y0=tf, x1=tw/2, y1=d-tf,
                      line=dict(color="#3182ce", width=1.5), fillcolor="#ebf4ff")
        fig.add_shape(type="rect", x0=-bf+tw/2, y0=d-tf, x1=bf/2, y1=d,
                      line=dict(color="#3182ce", width=1.5), fillcolor="#bee3f8")
        fig.add_shape(type="rect", x0=-bf+tw/2, y0=0, x1=bf/2, y1=tf,
                      line=dict(color="#3182ce", width=1.5), fillcolor="#bee3f8")
    elif family_name == "Angle":
        fig.add_shape(type="rect", x0=-tw/2, y0=0, x1=tw/2, y1=d-tf,
                      line=dict(color="#3182ce", width=1.5), fillcolor="#ebf4ff")
        fig.add_shape(type="rect", x0=-tw/2, y0=d-tf, x1=bf, y1=d,
                      line=dict(color="#3182ce", width=1.5), fillcolor="#bee3f8")
    
    # Centerline
    fig.add_shape(type="line", x0=0, y0=-d*0.05, x1=0, y1=d*1.05,
                  line=dict(color="#a0aec0", width=1, dash="dot"))
    fig.add_annotation(x=bf*0.15, y=d*0.95, text="C.G.", showarrow=False,
                       font=dict(size=9, color="#718096"))
    
    return fig
```

Insertar en el results panel, en la sección de geometría:
```python
with tabs[2]:
    st.json(result.get("geometry_source", {}))
    st.markdown("**Section Sketch**")
    sketch = plot_section_sketch(section, family)
    st.plotly_chart(sketch, use_container_width=True)
```

#### Task 6: Configuración del tema

**Objective:** Actualizar `.streamlit/config.toml` con tema más moderno.

**File:** `.streamlit/config.toml`

```toml
[server]
headless = true
port = 8501

[theme]
base = "light"
primaryColor = "#1a365d"
backgroundColor = "#f7fafc"
secondaryBackgroundColor = "#ffffff"
textColor = "#1a202c"
font = "sans serif"
```

---

### Verificación

Después de todos los cambios:
```bash
cd global-steel-designer
python3 -c "import py_compile; py_compile.compile('app.py', doraise=True)"
streamlit run app.py  # verificar que inicia sin errores
```

### Notas
- Los módulos de engine (aisc360_*) NO se modifican — solo app.py
- Mantener compatibilidad con el engine existente — no cambiar firmas de función
- El bloque `run` del botón se mantiene igual lógicamente, solo cambia el layout
