"""Pipeline de punta a punta: datos -> features -> modelo -> evaluación.

Correr con:  python -m src.pipeline

Hace lo mismo que los notebooks 01-03, pero de una sola vez y sin copy-paste. Deja el modelo
serializado, las métricas en JSON y una fila en el registro de experimentos.
"""
import csv
import json
from datetime import datetime

from . import data, features, model
from .config import load_config, path
from .evaluate import business_impact, reconstruct_amount


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()

    # 1) Datos
    df = data.load_raw(cfg)
    df, n_dups = data.clean(df)
    print(f"Datos: {len(df):,} filas tras quitar {n_dups:,} duplicados")

    # 2) Features + splits (con SMOTE solo en train)
    df = features.engineer(df, cfg)
    splits = features.make_splits(df, cfg)

    # 3) Entrenamiento + selección por PR-AUC
    out = model.train_and_select(splits, cfg)
    best = out["best"]
    print(f"Mejor modelo: {best}")

    # 4) Impacto de negocio del mejor modelo (umbral del config)
    proba = out["probas"][best]
    pred = (proba >= cfg["threshold"]).astype(int)
    amount = reconstruct_amount(
        splits["X_test"], splits["scaler"], splits["cols_scaled"], splits["features"]
    )
    impacto = business_impact(out["y_test"], pred, amount)

    # 5) Guardar modelo + métricas + experimento
    model.save_model(out["fitted"][best], best, splits, cfg)
    _write_metrics(out, best, impacto, cfg)
    _log_experiment(out, best, cfg)

    r = out["results"][best]
    print(f"PR-AUC {r['pr_auc']:.4f} | recall {r['recall']:.4f} | "
          f"{impacto['tp']}/{impacto['frauds_total']} fraudes | "
          f"EUR {impacto['losses_avoided_eur']:,.0f} evitados")
    return {"results": out["results"], "best": best, "impact": impacto}


def _write_metrics(out, best, impacto, cfg):
    r = out["results"][best]
    metrics = {
        "best_model": best,
        "comparison": {
            name: {"ROC-AUC": round(m["roc_auc"], 4), "PR-AUC": round(m["pr_auc"], 4),
                   "F1": round(m["f1"], 4), "Precision": round(m["precision"], 4),
                   "Recall": round(m["recall"], 4)}
            for name, m in out["results"].items()
        },
        "best": {
            "roc_auc": round(r["roc_auc"], 4), "pr_auc": round(r["pr_auc"], 4),
            "f1": round(r["f1"], 4), "precision": round(r["precision"], 4),
            "recall": round(r["recall"], 4),
            "tp": impacto["tp"], "fn": impacto["fn"], "fp": impacto["fp"], "tn": impacto["tn"],
            "frauds_total_test": impacto["frauds_total"],
            "losses_avoided_eur": round(impacto["losses_avoided_eur"], 0),
            "losses_missed_eur": round(impacto["losses_missed_eur"], 0),
        },
    }
    with open(path(cfg["paths"]["metrics"]), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def _log_experiment(out, best, cfg):
    """Registro ligero: una fila por corrida para poder comparar después."""
    fp = path(cfg["paths"]["experiments"])
    r = out["results"][best]
    fila = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "best_model": best, "threshold": cfg["threshold"],
        "roc_auc": round(r["roc_auc"], 4), "pr_auc": round(r["pr_auc"], 4),
        "f1": round(r["f1"], 4), "precision": round(r["precision"], 4),
        "recall": round(r["recall"], 4),
    }
    nuevo = not fp.exists()
    with open(fp, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fila.keys()))
        if nuevo:
            w.writeheader()
        w.writerow(fila)


if __name__ == "__main__":
    run()
