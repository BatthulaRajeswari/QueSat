"""QueSat — Earth Observation Data Analysis with Quantum ML (UC-097).

Streamlit dashboard. Run with:  streamlit run app.py
Every number shown comes from code that actually ran; nothing is hard-coded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src import (INSTITUTION, PROJECT_CODE, PROJECT_NAME, PROJECT_TITLE, TEAM, __version__)
from src import data_loader as dl
from src import quantum_circuit as qc
from src import regional_visualization as rv
from src import visualization as viz
from src.evaluation import CLASSICAL_NAME, QUANTUM_NAME, comparison_table, neutral_summary
from src.feature_engineering import FEATURE_DESCRIPTIONS, band_summary, feature_columns, add_spectral_indices
from src.ibm_quantum import CONFIG_REQUIRED, ibm_status
from src.preprocessing import preprocess
from src.quantum_model import ENGINES
from src.simulation import (LOCAL_SIM_LABEL, aer_sample, environment_status, run_pipeline,
                            transpile_summary)

st.set_page_config(page_title="QueSat — EO + Quantum ML", page_icon="🛰️", layout="wide")

PAGES = ["Dashboard", "Data Analysis", "Quantum Circuit", "Quantum Model",
         "Classical vs Quantum", "Results & Visualization", "IBM Quantum", "About Project"]

CONCEPTS = {
    "Quantum Feature Map": "Transforms classical features into a quantum feature space.",
    "Entanglement": "Creates interactions between encoded features.",
    "Quantum Kernel": "Measures similarity between samples in the quantum feature space.",
    "QSVM": "Uses the quantum kernel within an SVM classification workflow.",
}

PIPELINE_STEPS = ["Satellite Data", "Preprocessing", "Feature Extraction", "Quantum Encoding",
                  "Quantum Feature Map", "Entanglement", "Quantum Kernel", "QSVM", "Change Detection"]

CSS = """
<style>
.block-container{padding-top:1rem;max-width:1280px}
[data-testid="stSidebar"]{background:#0B1F3A}
[data-testid="stSidebar"] *{color:#E6EEFB !important}
.qs-banner{background:linear-gradient(120deg,#0B1F3A 0%,#12306b 55%,#1f4fbf 100%);border-radius:14px;
 padding:18px 24px;color:#fff;display:flex;justify-content:space-between;align-items:center;gap:18px;
 flex-wrap:wrap;margin-bottom:14px;border-bottom:3px solid #14B8A6}
.qs-title{font-size:30px;font-weight:800;letter-spacing:.5px}
.qs-sub{font-size:15px;opacity:.95}
.qs-meta{font-size:12.5px;opacity:.75;margin-top:2px}
.qs-right{text-align:right;font-size:13px;line-height:1.5}
.pill{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px;margin:3px 0 0 6px;
 background:rgba(255,255,255,.14);color:#fff;border:1px solid rgba(255,255,255,.25)}
.pill.ok{background:rgba(20,184,166,.25);border-color:#14B8A6}
.pill.bad{background:rgba(249,115,22,.25);border-color:#F97316}
.pill.demo{background:rgba(124,58,237,.35);border-color:#a78bfa}
.card{background:#F3F6FB;border:1px solid #DCE5F3;border-radius:12px;padding:14px 16px;height:100%}
.card .lbl{font-size:12px;text-transform:uppercase;letter-spacing:.8px;color:#51617d}
.card .val{font-size:28px;font-weight:800;color:#0B1F3A;line-height:1.2}
.card .sub{font-size:12px;color:#51617d}
.card.blue{border-top:4px solid #2F6FED}.card.teal{border-top:4px solid #14B8A6}
.card.purple{border-top:4px solid #7C3AED}.card.orange{border-top:4px solid #F97316}
.flow{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:8px 0 14px 0}
.chip{background:#0B1F3A;color:#fff;border-radius:8px;padding:7px 12px;font-size:13px}
.chip.q{background:#7C3AED}.chip.ok{background:#0f766e}.chip.warn{background:#B45309}.chip.off{background:#64748b}
.arrow{color:#2F6FED;font-weight:700}
.badge-demo{display:inline-block;background:#EEE8FF;color:#5b21b6;border:1px solid #c4b5fd;border-radius:8px;
 padding:6px 12px;font-size:13px;font-weight:600;margin:4px 0 10px 0}
.concept{background:#fff;border:1px solid #DCE5F3;border-left:4px solid #7C3AED;border-radius:8px;
 padding:8px 12px;margin-bottom:8px;font-size:13.5px}
.concept b{color:#0B1F3A}
</style>
"""


# ------------------------------------------------------------------ helpers
def html(s: str):
    st.markdown(s, unsafe_allow_html=True)


def card(label, value, sub="", accent="blue"):
    return (f'<div class="card {accent}"><div class="lbl">{label}</div>'
            f'<div class="val">{value}</div><div class="sub">{sub}</div></div>')


def flow(steps, kinds=None):
    parts = []
    for i, s in enumerate(steps):
        kind = (kinds[i] if kinds else "") or ""
        parts.append(f'<span class="chip {kind}">{s}</span>')
    html('<div class="flow">' + '<span class="arrow">→</span>'.join(parts) + "</div>")


def vflow(steps):
    for i, s in enumerate(steps):
        html(f'<div class="chip q" style="display:inline-block">{s}</div>')
        if i < len(steps) - 1:
            html('<div class="arrow" style="margin-left:18px">↓</div>')


def concept(name):
    html(f'<div class="concept"><b>{name}</b><br>{CONCEPTS[name]}</div>')


def fmt(v):
    return "unavailable" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def persistent_select(label, options, key):
    cur = st.session_state.get(key, options[0])
    idx = options.index(cur) if cur in options else 0
    val = st.selectbox(label, options, index=idx, key=f"w_{key}")
    st.session_state[key] = val
    return val


def init_state():
    s = st.session_state
    s.setdefault("roi", dl.AP_REGION)
    s.setdefault("area", "Krishna")
    s.setdefault("scenario", "Water-body Change")
    s.setdefault("data_source", "demo")
    s.setdefault("cfg", {"n_components": 4, "test_size": 0.25, "seed": 42, "reps": 2,
                         "engine": "statevector", "use_indices": False})
    s.setdefault("result", None)
    s.setdefault("upload_report", None)
    s.setdefault("upload_key", None)


def current_dataset():
    """Return (df, is_synthetic, stats, report) for the dataset the next run will use."""
    s = st.session_state
    if s["data_source"] == "upload":
        rep = s.get("upload_report")
        if rep is not None and rep.ok:
            return rep.df, False, rep.stats, rep
        return None, False, None, rep
    df = dl.generate_demo_data(s["area"], s["scenario"])
    rep = dl.validate_dataframe(df)
    return df, True, rep.stats, rep


def run_analysis():
    s = st.session_state
    df, synthetic, _, rep = current_dataset()
    if df is None:
        st.error("No valid uploaded dataset is available. Upload a valid CSV on the **Data Analysis** "
                 "page, or switch to demo data.")
        return False
    with st.spinner("Running hybrid classical–quantum pipeline…"):
        s["result"] = run_pipeline(df, s["area"], s["scenario"], synthetic, **s["cfg"])
    return True


def reset_all():
    s = st.session_state
    for k in ("result", "upload_report", "upload_key"):
        s[k] = None
    s["area"], s["scenario"], s["data_source"] = "Krishna", "Water-body Change", "demo"
    s["cfg"] = {"n_components": 4, "test_size": 0.25, "seed": 42, "reps": 2,
                "engine": "statevector", "use_indices": False}


def render_header(env, demo_active):
    q_ok = env["simulator_ready"]
    pills = ['<span class="pill ok">● System Ready</span>']
    pills.append('<span class="pill ok">Qiskit Simulator: Ready</span>' if q_ok
                 else '<span class="pill bad">Qiskit Simulator: Not available</span>')
    pills.append('<span class="pill ok">Aer: Ready</span>' if env["aer"]["available"]
                 else '<span class="pill bad">Aer: Not installed</span>')
    if demo_active:
        pills.append('<span class="pill demo">Demo Mode · Synthetic Data</span>')
    html(f"""<div class="qs-banner"><div><div class="qs-title">🛰️ {PROJECT_NAME}</div>
<div class="qs-sub">{PROJECT_TITLE}</div>
<div class="qs-meta">{PROJECT_CODE} | Hybrid Classical–Quantum Earth Observation</div></div>
<div class="qs-right"><div><b>{TEAM}</b> · RGUKT Nuzvid</div><div>{''.join(pills)}</div></div></div>""")


def demo_banner():
    html(f'<div class="badge-demo">{dl.DEMO_BANNER}</div>')


def need_result():
    r = st.session_state.get("result")
    if r is None:
        st.info("No analysis has been run yet. Go to **Dashboard** and click **RUN ANALYSIS** "
                "(or use the Expo quick start).")
        return None
    if r.get("message"):
        st.error(r["message"])
        return None
    return r


def show_run_messages(r):
    for w in r.get("warnings", []):
        st.warning(w)
    if r["quantum"] and r["quantum"]["status"] == "error":
        st.error(r["quantum"]["message"])
    if r["classical"] and r["classical"]["status"] == "error":
        st.error(r["classical"]["message"])


def model_results(r):
    return {CLASSICAL_NAME: r["classical"], QUANTUM_NAME: r["quantum"]}


# ------------------------------------------------------------------ pages
def page_dashboard():
    s = st.session_state
    st.header("Andhra Pradesh Earth Observation Intelligence")
    html('<div style="color:#51617d;margin-top:-8px;letter-spacing:1.5px;font-size:12px">'
         "ANDHRA PRADESH EARTH OBSERVATION ANALYSIS</div>")
    c1, c2, c3 = st.columns(3)
    with c1:
        persistent_select("Region of Interest", [dl.AP_REGION], "roi")
    with c2:
        persistent_select("Area", dl.AREAS, "area")
    with c3:
        persistent_select("Analysis Scenario", list(dl.SCENARIOS), "scenario")
    st.caption(f"Scenario: {dl.SCENARIOS[s['scenario']]['description']}  ·  Areas are demonstration "
               "Areas of Interest unless a dataset with real coordinates is supplied.")

    b1, b2, b3 = st.columns([1, 1, 2])
    run = b1.button("RUN ANALYSIS", type="primary")
    reset = b2.button("RESET")
    quick = b3.button("▶ Expo quick start: Krishna · Water-body Change")

    if reset:
        reset_all()
        st.rerun()
    if quick:
        s["area"], s["scenario"], s["data_source"] = "Krishna", "Water-body Change", "demo"
        if run_analysis():
            st.rerun()
    elif run:
        run_analysis()

    r = s["result"]
    st.subheader("Hybrid pipeline")
    flow(PIPELINE_STEPS, ["", "", "", "q", "q", "q", "q", "q", "ok"])

    q_info = r["circuit"]["info"] if (r and r.get("circuit") and r["circuit"]["status"] == "ok") else None
    cols = st.columns(4)
    if r and not r.get("message"):
        status = {"complete": "Complete", "partial": "Partial", "failed": "Failed"}[r["status"]]
        cols[0].markdown(card("Samples Analyzed", r["n_samples"],
                              f"train {len(r['prepared'].y_train)} · test {len(r['prepared'].y_test)}"),
                         unsafe_allow_html=True)
        cols[1].markdown(card("Quantum Features", r["prepared"].n_components,
                              "PCA applied" if r["prepared"].used_pca else "no PCA needed", "teal"),
                         unsafe_allow_html=True)
        cols[2].markdown(card("Qubits Used", q_info["qubits"] if q_info else "Unavailable",
                              LOCAL_SIM_LABEL if q_info else "Qiskit circuit not built", "purple"),
                         unsafe_allow_html=True)
        cols[3].markdown(card("Model Status", status,
                              f"classical: {r['classical']['status']} · quantum: {r['quantum']['status']}",
                              "orange"), unsafe_allow_html=True)
    else:
        for col, (lbl, accent) in zip(cols, [("Samples Analyzed", "blue"), ("Quantum Features", "teal"),
                                              ("Qubits Used", "purple"), ("Model Status", "orange")]):
            col.markdown(card(lbl, "—", "run an analysis", accent), unsafe_allow_html=True)

    if r and not r.get("message"):
        st.write("")
        if r["is_synthetic"]:
            demo_banner()
        show_run_messages(r)
        tbl = comparison_table(r["classical"], r["quantum"])
        show = tbl.copy()
        for c in ["Accuracy", "Precision", "Recall", "F1"]:
            show[c] = show[c].map(fmt)
        st.markdown(f"**Latest run — {r['region']} · {r['scenario']}**")
        st.table(show.set_index("Model"))
        st.caption("Open **Results & Visualization** for the regional output and **Classical vs Quantum** "
                   "for the full comparison.")
    elif r and r.get("message"):
        st.error(r["message"])

    with st.expander("Expo demo guide (3–5 minutes)", expanded=r is None):
        st.markdown(
            "1. Select **Andhra Pradesh → Krishna → Water-body Change**, click **RUN ANALYSIS**\n"
            "2. **Data Analysis** — satellite-derived features and preprocessing\n"
            "3. **Quantum Circuit** — the RY encoding + CNOT feature map\n"
            "4. **Quantum Model** — kernel matrix heatmap and QSVM output\n"
            "5. **Classical vs Quantum** — measured comparison (no assumed winner)\n"
            "6. **Results & Visualization** — illustrative regional change map\n"
            "7. **IBM Quantum** — honest status and the future hardware path")


def page_data():
    s = st.session_state
    st.header("Earth Observation Data")
    c1, c2, _ = st.columns([1, 1, 2])
    if c1.button("USE DEMO DATA"):
        s["data_source"] = "demo"
    if c2.button("UPLOAD CSV"):
        s["data_source"] = "upload"

    if s["data_source"] == "upload":
        st.markdown("**Upload satellite-derived CSV** — required header: `Blue,Green,Red,NIR,Change` "
                    "(optional `Latitude,Longitude`).")
        up = st.file_uploader("CSV file", type=["csv"], key="uploader")
        if up is not None:
            key = (up.name, up.size)
            if s.get("upload_key") != key:
                s["upload_report"] = dl.load_csv(up)
                s["upload_key"] = key
        template = dl.generate_demo_data("Krishna", "Water-body Change", n=40).to_csv(index=False)
        st.download_button("Download CSV template (synthetic example rows)", template,
                           "quesat_template.csv", "text/csv")
    else:
        demo_banner()
        st.markdown(f"**Source:** synthetic demo dataset for **{s['area']} · {s['scenario']}**  \n"
                    f"{dl.SCENARIOS[s['scenario']]['signature']}")

    df, synthetic, stats, rep = current_dataset()
    if rep is not None and s["data_source"] == "upload":
        for e in rep.errors:
            st.error(e)
        for w in rep.warnings:
            st.warning(w)
        if rep.ok:
            st.success(f"Validation passed: {rep.stats['n_valid']} usable rows.")
            st.info(f"Area label **{s['area']}** is only a label: the file is not verified to cover this region."
                    + (f" {dl.fraction_inside_ap(rep.df) * 100:.0f}% of supplied coordinates fall inside the "
                       "rough Andhra Pradesh bounding box." if dl.has_geo(rep.df) else ""))
    if df is None:
        st.info("Upload a valid CSV to continue.")
        return

    m = st.columns(4)
    m[0].metric("Samples", stats["n_valid"])
    m[1].metric("Features", stats["n_features"])
    m[2].metric("Missing values", stats["missing_cells"])
    m[3].metric("Change / No Change",
                f"{stats['class_counts'].get(1, 0)} / {stats['class_counts'].get(0, 0)}")
    st.markdown("**Dataset preview**")
    st.dataframe(df.head(15))
    a, b = st.columns([1, 2])
    a.pyplot(viz.class_distribution_fig(df))
    b.pyplot(viz.feature_distribution_fig(df))

    st.subheader("Preprocessing")
    flow(["Raw features", "StandardScaler", "Feature extraction", "Dimensionality reduction (PCA if required)",
          "Angle scaling [0, π]"], ["", "", "", "", "q"])
    cfg = dict(s["cfg"])
    k1, k2, k3 = st.columns(3)
    cfg["use_indices"] = k1.checkbox("Add NDVI & NDWI spectral indices (6 input features)",
                                     value=cfg["use_indices"])
    n_in = 6 if cfg["use_indices"] else 4
    cfg["n_components"] = k2.slider("Quantum feature dimension (= qubits)", 1, n_in,
                                    min(cfg["n_components"], n_in))
    cfg["test_size"] = k3.slider("Test split", 0.1, 0.4, float(cfg["test_size"]), 0.05)
    k4, k5, k6 = st.columns(3)
    cfg["reps"] = k4.slider("Feature-map layers (reps)", 1, 3, int(cfg["reps"]))
    cfg["engine"] = k5.selectbox("Kernel engine", list(ENGINES), index=list(ENGINES).index(cfg["engine"]),
                                 format_func=lambda k: ENGINES[k])
    cfg["seed"] = int(k6.number_input("Random seed", 0, 99999, int(cfg["seed"])))
    s["cfg"] = cfg
    st.caption(f"{n_in} input features → {cfg['n_components']} quantum features → "
               f"{cfg['n_components']} qubits. Default: 4 features → 4 qubits.")

    try:
        work = add_spectral_indices(df) if cfg["use_indices"] else df
        cols = feature_columns(cfg["use_indices"])
        p = preprocess(work, cols, cfg["n_components"], cfg["test_size"], cfg["seed"])
        left, right = st.columns(2)
        with left:
            st.markdown("**StandardScaler (fit on training split)**")
            st.dataframe(pd.DataFrame({"mean": p.scaler.mean_, "std": p.scaler.scale_}, index=cols).round(4))
            for n in p.notes:
                st.caption("• " + n)
        with right:
            if p.used_pca:
                st.pyplot(viz.pca_variance_fig(p.explained_variance))
            st.markdown("**Example quantum feature vector (first test sample, RY angles in radians)**")
            st.dataframe(pd.DataFrame([p.X_test[0]], columns=p.component_names).round(4))
    except Exception as exc:
        st.error(f"Preprocessing preview failed: {exc}")


def page_circuit():
    s = st.session_state
    r = s.get("result")
    st.header("How the Quantum Circuit Works")
    st.caption("Only the actual workflow used by the model is shown.")
    flow(["Feature Vector [x₁ … xₙ]", "RY Encoding", "Quantum Feature Map", "CNOT Entanglement",
          "Quantum Kernel", "QSVM Classification"], ["", "q", "q", "q", "q", "ok"])
    cc = st.columns(4)
    for col, name in zip(cc, CONCEPTS):
        with col:
            concept(name)

    cfg = s["cfg"]
    n = r["prepared"].n_components if (r and not r.get("message")) else cfg["n_components"]
    reps = r["config"]["reps"] if (r and not r.get("message")) else cfg["reps"]
    st.markdown("**Example input**")
    st.code("[x₁, x₂, x₃, x₄]   →   RY(x₁) RY(x₂) RY(x₃) RY(x₄)  →  CNOT chain", language="text")
    try:
        circuit = qc.build_feature_map(n, reps)
    except Exception as exc:
        st.error("Quantum circuit could not be built. The classical baseline remains available. "
                 f"Check the Qiskit installation. ({exc})")
        return
    st.markdown(f"**Qiskit circuit** — {n} qubits, {reps} layer(s)")
    st.code(qc.circuit_text(circuit), language="text")
    fig = qc.circuit_figure(circuit)
    if fig is not None:
        st.pyplot(fig)

    info = {
        "Qubits": n, "Feature dimension": n, "Encoding": "RY(xᵢ) angle encoding",
        "Feature map": f"RY layer + linear CNOT chain, repeated {reps}×",
        "Entanglement": "Linear CNOT (qubit i → i+1)",
        "Kernel": "Fidelity quantum kernel  k(x,y) = |⟨φ(x)|φ(y)⟩|²",
        "Classifier": "SVM with precomputed quantum kernel (QSVC-equivalent)",
    }
    from src.simulation import circuit_info
    ci = circuit_info(circuit, reps)
    info.update({"Circuit depth": ci["depth"], "Gate count": ci["gate_count"],
                 "Gates": ", ".join(f"{k}×{v}" for k, v in ci["gate_breakdown"].items()),
                 "Simulation": LOCAL_SIM_LABEL})
    st.table(pd.DataFrame({"Property": list(info), "Value": [str(v) for v in info.values()]}).set_index("Property"))
    st.info("Why 2 layers by default? For a fidelity kernel, a single RY layer followed by CNOTs gives "
            "the same kernel as no entanglement (the final fixed CNOT layer cancels). Re-encoding the "
            "data after a CNOT layer is what makes entanglement affect the kernel. Set layers to 1 on the "
            "Data Analysis page to see the single-layer circuit.")

    if r and not r.get("message"):
        x = r["prepared"].X_test[0]
        st.markdown("**The same circuit bound to the first real test sample**")
        st.code(f"x = {np.round(x, 3).tolist()}", language="text")
        st.code(qc.circuit_text(qc.bind_features(circuit, x)), language="text")


def page_quantum_model():
    st.header("Quantum Kernel + QSVM")
    flow(["Reduced Satellite Feature Vector", "Quantum Feature Encoding", "Quantum Feature Map",
          "Entanglement", "Quantum Kernel", "QSVM", "Classification"], ["", "q", "q", "q", "q", "q", "ok"])
    r = need_result()
    if r is None:
        return
    if r["is_synthetic"]:
        demo_banner()
    q = r["quantum"]
    if q["status"] != "ok":
        st.error(q["message"])
        st.info("The classical baseline remains available on the **Classical vs Quantum** page.")
        return
    if q.get("engine_note"):
        st.warning(q["engine_note"])
    prepared = r["prepared"]
    m = st.columns(4)
    m[0].metric("Kernel engine", q["engine"])
    m[1].metric("Training kernel", f"{q['K_train'].shape[0]}×{q['K_train'].shape[1]}")
    m[2].metric("Fit + evaluate (s)", f"{q['fit_seconds']:.2f}")
    m[3].metric("Mode", LOCAL_SIM_LABEL)

    a, b = st.columns(2)
    a.pyplot(viz.kernel_heatmap_fig(q["K_train"], prepared.y_train))
    with b:
        st.pyplot(viz.decision_hist_fig(q["decision"], prepared.y_test, "QSVM decision values (test set)"))
        st.caption("The decision value is the signed SVM margin score of the quantum-kernel SVM. It is "
                   "not a probability and is not presented as one.")

    st.markdown("**QSVM output on test samples**")
    n_show = min(20, len(prepared.y_test))
    out = pd.DataFrame({
        "Test sample": np.arange(n_show),
        "Actual": ["Change" if v else "No Change" for v in prepared.y_test[:n_show]],
        "QSVM prediction": ["Change" if v else "No Change" for v in q["y_pred"][:n_show]],
        "Decision value": np.round(q["decision"][:n_show], 4),
    })
    st.dataframe(out)

    st.markdown("**Aer shot-based sampling (local simulation)**")
    idx = int(st.number_input("Test sample index", 0, len(prepared.y_test) - 1, 0))
    if st.button("Sample this sample's circuit on Qiskit Aer"):
        res = aer_sample(r["circuit"]["circuit"], prepared.X_test[idx])
        if res["status"] == "ok":
            st.pyplot(viz.counts_fig(res["counts"]))
            st.caption(f"{res['shots']} shots · {res['mode']}. The kernel itself is computed exactly from "
                       "statevectors; this view shows measurement statistics of the same circuit.")
        else:
            st.error(res["message"])


def page_compare():
    st.header("Classical vs Quantum ML")
    r = need_result()
    if r is None:
        return
    if r["is_synthetic"]:
        demo_banner()
    show_run_messages(r)
    p = r["prepared"]
    st.caption(f"Both models use the same processed features ({p.n_components} dimensions), the same "
               f"training set (n = {len(p.y_train)}) and the same test set (n = {len(p.y_test)}).")
    tbl = comparison_table(r["classical"], r["quantum"])
    show = tbl.copy()
    for c in ["Accuracy", "Precision", "Recall", "F1"]:
        show[c] = show[c].map(fmt)
    st.table(show.set_index("Model"))
    st.pyplot(viz.comparison_chart_fig(tbl))
    st.markdown(neutral_summary(r["classical"], r["quantum"]))
    cm_cols = st.columns(2)
    for col, (name, res) in zip(cm_cols, model_results(r).items()):
        with col:
            if res and res["status"] == "ok":
                st.pyplot(viz.confusion_matrix_fig(res["metrics"]["confusion_matrix"], name))
            else:
                st.info(f"{name}: result unavailable.")
    t = r["timings"]
    st.caption(f"Classical fit: {r['classical'].get('fit_seconds', float('nan')):.3f}s · "
               f"Quantum fit + evaluate: {r['quantum'].get('fit_seconds', float('nan')):.3f}s "
               f"({LOCAL_SIM_LABEL}; simulation cost is not representative of quantum hardware).")


def page_results():
    st.header("Andhra Pradesh Earth Observation Results")
    r = need_result()
    if r is None:
        return
    avail = {n: res for n, res in model_results(r).items() if res and res["status"] == "ok"}
    if not avail:
        show_run_messages(r)
        return
    default = QUANTUM_NAME if QUANTUM_NAME in avail else CLASSICAL_NAME
    names = list(avail)
    model_name = st.selectbox("Model shown", names, index=names.index(default))
    res = avail[model_name]

    c = st.columns(4)
    c[0].markdown(card("Selected Region", r["region"], "Andhra Pradesh AOI (demonstration)" if r["is_synthetic"]
                       else "label only — not verified"), unsafe_allow_html=True)
    c[1].markdown(card("Scenario", r["scenario"].replace(" Change", ""), "Change detection", "teal"),
                  unsafe_allow_html=True)
    c[2].markdown(card("Model Used", "QSVM" if model_name == QUANTUM_NAME else "SVM", model_name, "purple"),
                  unsafe_allow_html=True)
    c[3].markdown(card("Analysis Status", r["status"].title(), LOCAL_SIM_LABEL, "orange"),
                  unsafe_allow_html=True)
    st.write("")
    if r["is_synthetic"]:
        demo_banner()
    show_run_messages(r)

    summ = None
    if r["scene"] is not None and model_name in r["scene_preds"]:
        pred = r["scene_preds"][model_name]
        summ = rv.summarize_scene(r["scene"], pred)
        st.subheader("Regional visualization")
        st.pyplot(rv.regional_map_fig(r["scene"], pred, model_name))
        k = st.columns(3)
        k[0].metric("Cells flagged as Change", f"{summ['detected_pct']:.1f}%")
        k[1].metric("Scene definition (changed cells)", f"{summ['scene_change_pct']:.1f}%")
        k[2].metric("Per-cell agreement", f"{summ['cell_accuracy_pct']:.1f}%")
        st.caption("The Detected-change panel is the model's own classification of synthetic per-cell "
                   "band values. 'Scene definition' is the illustrative ground truth used to create them. "
                   "Map predictions use the exact statevector evaluation of the quantum kernel.")
    elif r["geo"] is not None and model_name in r["geo"]["preds"]:
        st.subheader("Regional visualization — supplied coordinates")
        st.pyplot(rv.geo_scatter_fig(r["geo"]["df"], r["geo"]["preds"][model_name], model_name))
    else:
        st.info("No geographic coordinates are available for this dataset, so no regional map is shown. "
                "Supply `Latitude` and `Longitude` columns in the CSV to enable one. "
                "(No map is fabricated.)")
        if r["scene_errors"].get(model_name):
            st.error(f"Regional classification failed: {r['scene_errors'][model_name]}")

    st.subheader("Classification result")
    p = r["prepared"]
    n_test = len(p.y_test)
    pred_change = int(np.sum(res["y_pred"]))
    s1, s2 = st.columns([1, 2])
    with s1:
        idx = int(st.number_input("Test sample", 0, n_test - 1, 0))
        label = "Change" if res["y_pred"][idx] else "No Change"
        actual = "Change" if p.y_test[idx] else "No Change"
        st.markdown(f"**Classification:** {label}  \n**Actual label:** {actual}  \n"
                    f"**Decision value (model output):** {res['decision'][idx]:.4f}")
    with s2:
        st.markdown(f"On the test set, **{pred_change} of {n_test}** samples were classified as Change "
                    f"({int(p.y_test.sum())} actually Change).")
        raw = r["data"].iloc[p.idx_test[idx]][r["feature_cols"]]
        st.dataframe(pd.DataFrame([raw.values], columns=r["feature_cols"]).round(4))

    st.subheader("Satellite-derived feature analysis")
    d1, d2 = st.columns(2)
    d1.pyplot(viz.band_means_fig(band_summary(r["data"])))
    with d2:
        st.markdown("**Quantum processing of this sample**")
        flow(["Feature vector", "Quantum encoding", "Feature map", "Entanglement", "Quantum kernel", "QSVM"],
             ["", "q", "q", "q", "q", "ok"])
        st.code(f"RY angles x = {np.round(p.X_test[idx], 3).tolist()}", language="text")
        if model_name == CLASSICAL_NAME:
            st.caption("Shown model is classical; the quantum steps apply to the QSVM.")

    e1, e2 = st.columns(2)
    e1.pyplot(viz.confusion_matrix_fig(res["metrics"]["confusion_matrix"], f"Confusion matrix — {model_name}"))
    e2.pyplot(viz.feature_space_fig(p))

    st.subheader("Final environmental insight")
    m = res["metrics"]
    scope = "synthetic, illustrative" if r["is_synthetic"] else "supplied"
    text = (f"For {r['region']} · {r['scenario']} ({scope} data), {model_name} reached accuracy "
            f"{m['accuracy']:.3f} and F1 {m['f1']:.3f} on {m['n_test']} held-out samples.")
    if summ:
        text += (f" It flagged {summ['detected_pct']:.1f}% of the illustrative scene cells as Change "
                 f"({summ['scene_change_pct']:.1f}% were defined as changed).")
    if r["is_synthetic"]:
        text += " These are properties of a synthetic demonstration, not observations of the real region."
    st.markdown(text)


def page_ibm():
    st.header("IBM Quantum Integration")
    env = environment_status()
    ibm = ibm_status()
    sim_ok = env["simulator_ready"]
    flow(["Local Qiskit Simulation", "Circuit Transpilation", "IBM Quantum Backend", "Hardware Execution",
          "Result Comparison"],
         ["ok" if sim_ok else "off", "ok" if sim_ok else "off", "warn", "off", "off"])
    c = st.columns(3)
    c[0].markdown(card("Local simulation", "Ready" if sim_ok else "Unavailable", LOCAL_SIM_LABEL, "teal"),
                  unsafe_allow_html=True)
    c[1].markdown(card("IBM backend", "Not connected", ibm["label"], "orange"), unsafe_allow_html=True)
    c[2].markdown(card("Hardware execution", "None", "No job has been submitted", "purple"),
                  unsafe_allow_html=True)
    st.write("")
    if not ibm["credentials_found"]:
        st.warning(CONFIG_REQUIRED)
    else:
        st.info(ibm["label"] + ". " + ibm["detail"])
    st.caption("This prototype never claims hardware results. The comparison step stays empty until a real "
               "hardware job is run.")

    st.subheader("Circuit transpilation preview (no backend contacted)")
    r = st.session_state.get("result")
    if r and r.get("circuit") and r["circuit"]["status"] == "ok":
        t = transpile_summary(r["circuit"]["circuit"])
        if t["status"] == "ok":
            st.write(f"Transpiled to a generic IBM-style basis {t['basis']}: depth **{t['depth']}**, "
                     f"gates **{t['gate_count']}** ({', '.join(f'{k}×{v}' for k, v in t['gate_breakdown'].items())}).")
        else:
            st.error(t["message"])
    else:
        st.info("Run an analysis first to preview transpilation of the feature-map circuit.")

    st.subheader("Secure configuration")
    st.markdown("Never put API keys in source code. Use an environment variable or a saved Qiskit account:")
    st.code("# Option A — environment variable (current shell only)\n"
            "export QISKIT_IBM_TOKEN='<your token>'\n\n"
            "# Option B — save once to ~/.qiskit/qiskit-ibm.json (outside the repository)\n"
            "pip install qiskit-ibm-runtime\n"
            "python -c \"from qiskit_ibm_runtime import QiskitRuntimeService as S; "
            "S.save_account(channel='ibm_quantum_platform', token='<your token>')\"", language="bash")
    st.caption("`.env`, `secrets.toml` and `qiskit-ibm.json` are already in .gitignore. See docs/ibm_quantum.md.")


def page_about():
    st.header("About Project")
    st.markdown(f"**{PROJECT_NAME} — {PROJECT_TITLE}** · {PROJECT_CODE} · {TEAM} · {INSTITUTION} · v{__version__}")
    sections = {
        "Problem": "Earth-observation analysts must separate genuine land-surface change (water-body shifts, "
                   "crop cycles, floods, land-cover conversion) from noise in multispectral measurements, "
                   "quickly and repeatably, across large regions.",
        "Classical limitation": "Kernel methods such as SVMs depend on a hand-chosen similarity function. Good "
                                "kernels for overlapping spectral signatures are hard to design, and the "
                                "choice strongly affects results.",
        "Research gap": "Quantum kernels offer a different, high-dimensional similarity measure, but there is "
                        "little application-level, honestly-measured comparison against a classical baseline "
                        "on Earth-observation features, particularly for Andhra Pradesh.",
        "Proposed solution": "A hybrid workflow: classical preprocessing and dimensionality reduction, a quantum "
                             "feature map and fidelity kernel simulated with Qiskit, and an SVM classifier — "
                             "evaluated side by side with a classical SVM on identical data splits.",
        "Quantum approach": "Features are rescaled to RY rotation angles on n qubits, entangled with a linear "
                            "CNOT chain (re-encoded for 2 layers), and the kernel is the state fidelity "
                            "|⟨φ(x)|φ(y)⟩|². The resulting Gram matrix is passed to an SVM (QSVM).",
        "Earth-observation application": "Blue, Green, Red and NIR band values (and optional NDVI/NDWI) "
                                         "classify each sample as Change or No Change for water-body, "
                                         "agricultural, flood-related and land-cover scenarios.",
        "Andhra Pradesh focus": "Selectable demonstration Areas of Interest: Krishna, Guntur, Vijayawada Region, "
                                "Visakhapatnam, Kakinada, Godavari Region, Nellore and Tirupati. Real "
                                "regional analysis requires user-supplied data for that region.",
        "Expected impact": "A transparent, reproducible testbed for evaluating quantum-kernel methods on "
                           "Earth-observation data, and a teaching demonstration of a complete hybrid pipeline.",
        "Future scope": "Sentinel-2 derived features, temporal observations, geographic coordinates and "
                        "region metadata, larger feature sets, repeated cross-validation, noise models, and "
                        "execution on IBM Quantum hardware once access is configured.",
    }
    for title, body in sections.items():
        st.markdown(f"**{title}**")
        st.write(body)
    st.warning("Limitations: the bundled data are synthetic; results demonstrate the pipeline, not the "
               "real condition of any region; the quantum model is a classical simulation; no quantum "
               "advantage is claimed or implied.")


# ------------------------------------------------------------------ main
def main():
    init_state()
    html(CSS)
    env = environment_status()
    s = st.session_state
    with st.sidebar:
        html('<div style="font-size:22px;font-weight:800;padding:6px 0">🛰️ QueSat</div>'
             '<div style="font-size:12px;opacity:.8;margin-bottom:10px">UC-097 · Earth Observation × Quantum ML</div>')
    page = st.sidebar.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")
    r = s.get("result")
    demo_active = (r["is_synthetic"] if (r and not r.get("message")) else s["data_source"] == "demo")
    st.sidebar.caption("● Demo Mode (synthetic data)" if demo_active else "● Using uploaded data")
    st.sidebar.caption(f"{LOCAL_SIM_LABEL} only · no IBM hardware used")
    render_header(env, demo_active)
    {"Dashboard": page_dashboard, "Data Analysis": page_data, "Quantum Circuit": page_circuit,
     "Quantum Model": page_quantum_model, "Classical vs Quantum": page_compare,
     "Results & Visualization": page_results, "IBM Quantum": page_ibm,
     "About Project": page_about}[page]()


main()
