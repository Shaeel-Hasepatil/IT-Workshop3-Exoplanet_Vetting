import os, json, sqlite3, datetime
import joblib, numpy as np, pandas as pd, shap, gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "predictions.db")

model = joblib.load(os.path.join(HERE, "vetting_model.joblib"))
cfg = json.load(open(os.path.join(HERE, "config.json")))
all_cols, app_features = cfg["all_cols"], cfg["app_features"]
labels, examples = cfg["labels"], cfg["examples"]

imputer = model.named_steps["imputer"]
explainer = shap.TreeExplainer(model.named_steps["model"])


def shap_class1(X_imputed):
    sv = explainer.shap_values(X_imputed)
    if isinstance(sv, list):
        sv = sv[1]
    elif np.ndim(sv) == 3:
        sv = sv[:, :, 1]
    return sv


def label_from_prob(p):
    return "LIKELY PLANET" if p >= 0.5 else "LIKELY FALSE POSITIVE"


def save_prediction(p):
    try:
        db = sqlite3.connect(DB_PATH)
        db.execute("CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, kepoi_name TEXT, kepid INTEGER, planet_probability REAL, predicted_label TEXT, source TEXT, created_at TEXT)")
        db.execute("INSERT INTO predictions (kepoi_name, kepid, planet_probability, predicted_label, source, created_at) VALUES (?,?,?,?,?,?)",
                   ("USER_INPUT", None, p, label_from_prob(p), "app", datetime.datetime.now().isoformat(timespec="seconds")))
        db.commit()
        db.close()
    except Exception:
        pass


def vet_signal(*values):
    # one row with every feature empty, then fill in what the user typed
    row = pd.DataFrame([[np.nan] * len(all_cols)], columns=all_cols)
    for name, v in zip(app_features, values):
        if v is not None:
            row.at[0, name] = float(v)

    p = float(model.predict_proba(row)[0, 1])

    # top 5 features by SHAP contribution
    row_imp = pd.DataFrame(imputer.transform(row), columns=all_cols)
    contrib = shap_class1(row_imp)[0]
    order = np.argsort(-np.abs(contrib))[:5]
    lines = []
    for i in order:
        direction = "towards PLANET" if contrib[i] > 0 else "towards FALSE POSITIVE"
        shown = "not given (median used)" if pd.isna(row.iloc[0, i]) else "%g" % row.iloc[0, i]
        lines.append("%s = %s: pushes %s" % (all_cols[i], shown, direction))

    save_prediction(p)
    return {"Planet": p, "False positive": 1 - p}, "\n".join(lines)


demo = gr.Interface(
    fn=vet_signal,
    inputs=[gr.Number(label=labels[f]) for f in app_features],
    outputs=[gr.Label(label="Prediction"), gr.Textbox(label="Top 5 reasons", lines=6)],
    examples=examples,
    title="Kepler Signal Vetting Dashboard",
    description="Enter measurements or click an example. Empty fields use the training median.",
)

if __name__ == "__main__":
    demo.launch()
