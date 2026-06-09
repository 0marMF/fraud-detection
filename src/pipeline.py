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
from .evaluate import (business_impact, classification_metrics, cross_val_prauc,
                       dummy_baseline, reconstruct_amount, threshold_cost_curve)


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()

    # 1) Datos
    df = data.load_raw(cfg)
    df, n_dups = data.clean(df)
    print(f"Datos: {len(df):,} filas tras quitar {n_dups:,} duplicados")

    # 2) Features + splits (SMOTE solo en train; guardamos también el train sin SMOTE)
    df = features.engineer(df, cfg)
    splits = features.make_splits(df, cfg)

    # 3) Entrenamiento + selección por PR-AUC
    out = model.train_and_select(splits, cfg)
    best = out["best"]
    proba = out["probas"][best]
    print(f"Mejor modelo: {best}")

    # 4) Contexto: baseline tonto y robustez por validación cruzada (SMOTE dentro de cada fold)
    baseline = dummy_baseline(splits["X_train_raw"], splits["y_train_raw"],
                              splits["X_test"], out["y_test"])
    cv_mean, cv_std = cross_val_prauc(out["fitted"][best], splits["X_train_raw"],
                                      splits["y_train_raw"], cfg["seed"])
    print(f"PR-AUC CV (5-fold): {cv_mean:.4f} ± {cv_std:.4f} | baseline PR-AUC {baseline['pr_auc']:.4f}")

    # 5) Umbral por coste en euros (FN = monto del fraude; FP = revisión manual)
    amount = reconstruct_amount(splits["X_test"], splits["scaler"],
                                splits["cols_scaled"], splits["features"])
    fp_cost = cfg["costs"]["fp_review_cost"]
    _, _, t_star, cost_star = threshold_cost_curve(out["y_test"], proba, amount, fp_cost)

    impact_05 = business_impact(out["y_test"], (proba >= 0.5).astype(int), amount)
    impact_star = business_impact(out["y_test"], (proba >= t_star).astype(int), amount)
    m_star = classification_metrics(out["y_test"], proba, t_star)
    print(f"Umbral por coste: {t_star:.2f} -> recall {m_star['recall']:.3f}, "
          f"{impact_star['fp']} falsas alarmas, EUR {cost_star:,.0f} de coste total")

    # 6) Guardar modelo (con el umbral elegido) + métricas + experimento
    model.save_model(out["fitted"][best], best, splits, cfg, threshold=t_star)
    _write_metrics(out, best, baseline, (cv_mean, cv_std), impact_05,
                   t_star, m_star, impact_star, cost_star, fp_cost, cfg)
    _log_experiment(out, best, t_star, cfg)
    return {"best": best, "results": out["results"], "operating_point_threshold": t_star}


def _write_metrics(out, best, baseline, cv, impact_05, t_star, m_star,
                   impact_star, cost_star, fp_cost, cfg):
    r = out["results"][best]
    cv_mean, cv_std = cv
    metrics = {
        "best_model": best,
        "comparison": {
            name: {"ROC-AUC": round(m["roc_auc"], 4), "PR-AUC": round(m["pr_auc"], 4),
                   "F1": round(m["f1"], 4), "Precision": round(m["precision"], 4),
                   "Recall": round(m["recall"], 4)}
            for name, m in out["results"].items()
        },
        "baseline_dummy": {"pr_auc": round(baseline["pr_auc"], 4),
                           "recall": round(baseline["recall"], 4)},
        "cv": {"model": best, "metric": "PR-AUC", "n_splits": 5,
               "mean": round(cv_mean, 4), "std": round(cv_std, 4)},
        # Métricas del mejor modelo a umbral 0.5 (referencia, compatibles con el dashboard)
        "best": {
            "roc_auc": round(r["roc_auc"], 4), "pr_auc": round(r["pr_auc"], 4),
            "f1": round(r["f1"], 4), "precision": round(r["precision"], 4),
            "recall": round(r["recall"], 4),
            "tp": impact_05["tp"], "fn": impact_05["fn"], "fp": impact_05["fp"],
            "tn": impact_05["tn"], "frauds_total_test": impact_05["frauds_total"],
            "losses_avoided_eur": round(impact_05["losses_avoided_eur"], 0),
            "losses_missed_eur": round(impact_05["losses_missed_eur"], 0),
        },
        # Punto de operación recomendado: el que minimiza el coste en euros
        "operating_point": {
            "threshold": round(t_star, 3), "fp_review_cost_eur": fp_cost,
            "total_cost_eur": round(cost_star, 0),
            "recall": round(m_star["recall"], 4), "precision": round(m_star["precision"], 4),
            "tp": impact_star["tp"], "fn": impact_star["fn"], "fp": impact_star["fp"],
            "frauds_total_test": impact_star["frauds_total"],
            "losses_avoided_eur": round(impact_star["losses_avoided_eur"], 0),
            "losses_missed_eur": round(impact_star["losses_missed_eur"], 0),
        },
    }
    with open(path(cfg["paths"]["metrics"]), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def _log_experiment(out, best, t_star, cfg):
    """Registro ligero: una fila por corrida para poder comparar después."""
    fp = path(cfg["paths"]["experiments"])
    r = out["results"][best]
    fila = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "best_model": best, "threshold_opt": round(t_star, 3),
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
