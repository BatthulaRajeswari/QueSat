"""Generate sample figures + metrics from the SYNTHETIC demo data (no UI needed).

    python scripts/generate_sample_outputs.py                 # real Qiskit pipeline
    python scripts/generate_sample_outputs.py --reference-sim # only if Qiskit is unavailable

--reference-sim replaces the Qiskit statevector step with the independent NumPy simulator
from tests/reference_sim.py (same circuit maths). Outputs are then suffixed
"_reference-sim" and the metrics file records this. Do not present them as Qiskit results.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_loader as dl                                    # noqa: E402
from src import quantum_circuit as qc                                # noqa: E402
from src import regional_visualization as rv                         # noqa: E402
from src import visualization as viz                                 # noqa: E402
from src.evaluation import CLASSICAL_NAME, QUANTUM_NAME, comparison_table, neutral_summary  # noqa: E402
from src.simulation import run_pipeline                              # noqa: E402

CASES = [("Krishna", "Water-body Change"), ("Guntur", "Agricultural Change"),
         ("Godavari Region", "Flood-related Change"), ("Vijayawada Region", "Land-cover Change")]


class _FakeCircuit:  # only used with --reference-sim
    num_qubits = 4
    num_parameters = 4
    def depth(self): return 0
    def count_ops(self): return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-sim", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "docs" / "sample_outputs"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if args.reference_sim:
        from tests import reference_sim as ref
        qc.build_feature_map = lambda n, reps=2: _FakeCircuit()
        qc.statevectors = ref.statevectors_for(2)
        suffix = "_reference-sim"
        print("WARNING: using the NumPy reference simulator instead of Qiskit.")

    summary = {"engine": "numpy reference simulator (NOT Qiskit)" if args.reference_sim else "Qiskit",
               "data": "SYNTHETIC demo data", "runs": []}
    for region, scenario in CASES:
        df = dl.generate_demo_data(region, scenario)
        r = run_pipeline(df, region, scenario, True)
        slug = f"{region.lower().replace(' ', '_')}_{dl.scenario_slug(scenario)}{suffix}"
        entry = {"region": region, "scenario": scenario, "status": r["status"]}
        tbl = comparison_table(r["classical"], r["quantum"])
        entry["metrics"] = {row["Model"]: {k: (None if row[k] != row[k] else round(float(row[k]), 4))
                                           for k in ("Accuracy", "Precision", "Recall", "F1")}
                            for _, row in tbl.iterrows()}
        entry["summary"] = neutral_summary(r["classical"], r["quantum"])
        model = QUANTUM_NAME if r["quantum"]["status"] == "ok" else CLASSICAL_NAME
        if r["scene_preds"].get(model) is not None:
            rv.regional_map_fig(r["scene"], r["scene_preds"][model], model).savefig(
                out / f"{slug}_map.png", dpi=110, bbox_inches="tight")
            entry["map_model"] = model
            entry["scene"] = rv.summarize_scene(r["scene"], r["scene_preds"][model])
        viz.comparison_chart_fig(tbl).savefig(out / f"{slug}_comparison.png", dpi=110, bbox_inches="tight")
        viz.feature_space_fig(r["prepared"]).savefig(out / f"{slug}_feature_space.png", dpi=110, bbox_inches="tight")
        if r["quantum"]["status"] == "ok":
            viz.kernel_heatmap_fig(r["quantum"]["K_train"], r["prepared"].y_train).savefig(
                out / f"{slug}_kernel.png", dpi=110, bbox_inches="tight")
        summary["runs"].append(entry)
        print(region, "|", scenario, "|", r["status"])
    (out / f"metrics{suffix}.json").write_text(json.dumps(summary, indent=2))
    print("written to", out)


if __name__ == "__main__":
    main()
