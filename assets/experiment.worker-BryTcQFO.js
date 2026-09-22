(function(){"use strict";const c="https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";let r=null;function t(a,e){self.postMessage({id:a,type:"status",status:e})}async function p(a){return r||(r=(async()=>{t(a,"Loading Python runtime");const{loadPyodide:e}=await import(c),i=await e();return t(a,"Loading NumPy and scikit-learn"),await i.loadPackage(["numpy","scikit-learn"]),i})().catch(e=>{throw r=null,e})),r}const o={overfitting:`
import json
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

def make_case(seed):
    rng = np.random.default_rng(seed)
    x_train = np.sort(rng.uniform(-3, 3, 30))
    y_train = np.sin(1.7 * x_train) + 0.22 * x_train + rng.normal(0, 0.43, 30)
    x_valid = rng.uniform(-3, 3, 140)
    y_valid = np.sin(1.7 * x_valid) + 0.22 * x_valid + rng.normal(0, 0.43, 140)
    x_curve = np.linspace(-3, 3, 100)

    def fit(max_depth, min_samples_leaf):
        model = DecisionTreeRegressor(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=seed
        )
        model.fit(x_train.reshape(-1, 1), y_train)
        return {
            "trainMse": round(float(mean_squared_error(y_train, model.predict(x_train.reshape(-1, 1)))), 3),
            "validationMse": round(float(mean_squared_error(y_valid, model.predict(x_valid.reshape(-1, 1)))), 3),
            "predictions": [
                {"x": round(float(x), 3), "y": round(float(y), 3)}
                for x, y in zip(x_curve, model.predict(x_curve.reshape(-1, 1)))
            ],
        }

    simple = fit(3, 5)
    flexible = fit(None, 1)
    return {
        "kind": "overfitting",
        "seed": seed,
        "trainingCount": len(x_train),
        "validationCount": len(x_valid),
        "simple": simple,
        "flexible": flexible,
        "observations": [
            {"x": round(float(x), 3), "y": round(float(y), 3)}
            for x, y in zip(x_train, y_train)
        ],
    }

seeds = [17, 28, 42, 57, 99, 123, 211, 303]
valid_cases = []
for seed in seeds:
    candidate = make_case(seed)
    simple = candidate["simple"]
    flexible = candidate["flexible"]
    # A valid variant has a clear training fit and a meaningful validation penalty.
    if (flexible["trainMse"] < 0.015
        and flexible["validationMse"] > flexible["trainMse"] + 0.08
        and flexible["validationMse"] > simple["validationMse"] * 1.1):
        valid_cases.append(candidate)
if not valid_cases:
    raise ValueError("No seed expressed the intended overfitting pattern")
json.dumps(valid_cases)
`,imbalance:`
import json
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def make_case(seed):
    X, y = make_classification(
        n_samples=1100, n_features=8, n_informative=4, n_redundant=1,
        weights=[0.9, 0.1], class_sep=0.85, flip_y=0.02, random_state=seed
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=seed))
    model.fit(X_train, y_train)
    baseline_prediction = np.zeros_like(y_valid)
    classifier_prediction = (model.predict_proba(X_valid)[:, 1] >= 0.2).astype(int)

    def metrics(prediction):
        tn, fp, fn, tp = confusion_matrix(y_valid, prediction, labels=[0, 1]).ravel()
        return {
            "accuracy": round(float(accuracy_score(y_valid, prediction)), 3),
            "precision": round(float(precision_score(y_valid, prediction, zero_division=0)), 3),
            "recall": round(float(recall_score(y_valid, prediction)), 3),
            "truePositive": int(tp), "falsePositive": int(fp),
            "trueNegative": int(tn), "falseNegative": int(fn),
        }

    return {
        "kind": "imbalance", "seed": seed,
        "trainingCount": len(y_train), "validationCount": len(y_valid),
        "positiveCount": int(np.sum(y_valid)),
        "negativeCount": int(len(y_valid) - np.sum(y_valid)),
        "baseline": metrics(baseline_prediction),
        "classifier": metrics(classifier_prediction),
    }

seeds = [17, 28, 42, 57, 99, 123, 211, 303]
valid_cases = []
for seed in seeds:
    candidate = make_case(seed)
    baseline, classifier = candidate["baseline"], candidate["classifier"]
    # An accepted case must make accuracy misleading and reward recall.
    if (baseline["accuracy"] >= 0.85
        and baseline["recall"] == 0
        and classifier["recall"] >= 0.55
        and classifier["precision"] >= 0.15
        and classifier["falseNegative"] < baseline["falseNegative"]):
        valid_cases.append(candidate)
if not valid_cases:
    raise ValueError("No seed expressed the intended class-imbalance pattern")
json.dumps(valid_cases)
`,regularization:`
import json
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def make_case(seed):
    rng = np.random.default_rng(seed)
    x_train = np.sort(rng.uniform(-1, 1, 27))
    y_train = np.sin(3 * x_train) + rng.normal(0, 0.27, 27)
    x_valid = rng.uniform(-1, 1, 160)
    y_valid = np.sin(3 * x_valid) + rng.normal(0, 0.27, 160)
    x_curve = np.linspace(-1, 1, 100)

    def fit(alpha):
        model = make_pipeline(
            PolynomialFeatures(14, include_bias=False),
            StandardScaler(),
            Ridge(alpha=alpha)
        )
        model.fit(x_train.reshape(-1, 1), y_train)
        return {
            "trainMse": round(float(mean_squared_error(y_train, model.predict(x_train.reshape(-1, 1)))), 3),
            "validationMse": round(float(mean_squared_error(y_valid, model.predict(x_valid.reshape(-1, 1)))), 3),
            "predictions": [
                {"x": round(float(x), 3), "y": round(float(y), 3)}
                for x, y in zip(x_curve, model.predict(x_curve.reshape(-1, 1)))
            ],
        }

    return {
        "kind": "regularization", "seed": seed,
        "trainingCount": len(x_train), "validationCount": len(x_valid),
        "low": fit(0.000001), "high": fit(2.0),
        "observations": [
            {"x": round(float(x), 3), "y": round(float(y), 3)}
            for x, y in zip(x_train, y_train)
        ],
    }

seeds = [17, 28, 42, 57, 99, 123, 211, 303]
valid_cases = []
for seed in seeds:
    candidate = make_case(seed)
    low, high = candidate["low"], candidate["high"]
    # The stronger penalty must trade training fit for better holdout error.
    if (high["trainMse"] > low["trainMse"] + 0.01
        and low["validationMse"] > high["validationMse"] * 1.15):
        valid_cases.append(candidate)
if not valid_cases:
    raise ValueError("No seed expressed the intended regularization pattern")
json.dumps(valid_cases)
`},s=new Map;let l=Promise.resolve();async function _(a){const{id:e,kind:i,variant:m}=a.data;try{if(!(i in o))throw new Error("Unknown experiment type");const n=await p(e);s.has(i)||(t(e,"Validating experiment variations"),s.set(i,JSON.parse(await n.runPythonAsync(o[i]))));const d=s.get(i),f=d[Math.max(0,Math.floor(m))%d.length];self.postMessage({id:e,type:"result",result:f})}catch(n){self.postMessage({id:e,type:"error",error:n instanceof Error?n.message:String(n)})}}self.onmessage=a=>{l=l.then(()=>_(a))}})();
