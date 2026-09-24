(function(){"use strict";const c="https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";let r=null;function s(i,e){self.postMessage({id:i,type:"status",status:e})}async function f(i){return r||(r=(async()=>{s(i,"Loading Python runtime");const{loadPyodide:e}=await import(c),a=await e();return s(i,"Loading NumPy and scikit-learn"),await a.loadPackage(["numpy","scikit-learn"]),a})().catch(e=>{throw r=null,e})),r}const o={overfitting:`
import json
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

CONFIGS = [
  ("flexible-overfits", 30, 140, 1.7, .43, (3, 5), (None, 1)),
  ("simple-underfits", 80, 140, 3.4, .16, (1, 10), (5, 3)),
  ("similar-generalization", 80, 140, 1.7, .22, (3, 5), (4, 4)),
  ("holdout-too-small", 34, 24, 1.7, .36, (3, 5), (5, 3)),
]

def make_case(seed, config):
    case_key, train_n, valid_n, frequency, noise, simple_config, flexible_config = config
    rng = np.random.default_rng(seed)
    x_train = np.sort(rng.uniform(-3, 3, train_n))
    y_train = np.sin(frequency * x_train) + .18 * x_train + rng.normal(0, noise, train_n)
    x_valid = rng.uniform(-3, 3, valid_n)
    y_valid = np.sin(frequency * x_valid) + .18 * x_valid + rng.normal(0, noise, valid_n)
    x_curve = np.linspace(-3, 3, 100)
    def fit(values):
        model = DecisionTreeRegressor(max_depth=values[0], min_samples_leaf=values[1], random_state=seed)
        model.fit(x_train.reshape(-1, 1), y_train)
        return {"trainMse": round(float(mean_squared_error(y_train, model.predict(x_train.reshape(-1, 1)))), 3), "validationMse": round(float(mean_squared_error(y_valid, model.predict(x_valid.reshape(-1, 1)))), 3), "predictions": [{"x": round(float(x), 3), "y": round(float(y), 3)} for x, y in zip(x_curve, model.predict(x_curve.reshape(-1, 1)))]}
    return {"kind": "overfitting", "caseKey": case_key, "seed": seed, "trainingCount": len(x_train), "validationCount": len(x_valid), "simple": fit(simple_config), "flexible": fit(flexible_config), "observations": [{"x": round(float(x), 3), "y": round(float(y), 3)} for x, y in zip(x_train, y_train)]}

def accepted(case):
    simple, flexible = case["simple"], case["flexible"]
    if case["caseKey"] == "flexible-overfits": return flexible["trainMse"] < .02 and flexible["validationMse"] > simple["validationMse"] * 1.1
    if case["caseKey"] == "simple-underfits": return flexible["validationMse"] < simple["validationMse"] * .75
    if case["caseKey"] == "similar-generalization": return abs(flexible["validationMse"] - simple["validationMse"]) < .035
    return abs(flexible["validationMse"] - simple["validationMse"]) < .07 and flexible["trainMse"] < simple["trainMse"]

cases = []
for index, config in enumerate(CONFIGS):
    found = next((make_case(seed, config) for seed in range(17 + index, 700) if accepted(make_case(seed, config))), None)
    if found is None: raise ValueError("A model-diagnosis case could not be validated")
    cases.append(found)
json.dumps(cases)
`,imbalance:`
import json
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

CONFIGS = [
  ("recall-first", [.90, .10], .85, .20),
  ("precision-first", [.88, .12], 1.15, .72),
  ("balanced-outcome", [.78, .22], .92, .50),
  ("accuracy-is-usable", [.52, .48], 1.10, .50),
]

def make_case(seed, config):
    case_key, weights, separation, threshold = config
    X, y = make_classification(n_samples=1100, n_features=8, n_informative=4, n_redundant=1, weights=weights, class_sep=separation, flip_y=.02, random_state=seed)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.3, stratify=y, random_state=seed)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=seed))
    model.fit(X_train, y_train)
    baseline_prediction = np.zeros_like(y_valid)
    prediction = (model.predict_proba(X_valid)[:, 1] >= threshold).astype(int)
    def metrics(values):
        tn, fp, fn, tp = confusion_matrix(y_valid, values, labels=[0, 1]).ravel()
        return {"accuracy": round(float(accuracy_score(y_valid, values)), 3), "precision": round(float(precision_score(y_valid, values, zero_division=0)), 3), "recall": round(float(recall_score(y_valid, values)), 3), "truePositive": int(tp), "falsePositive": int(fp), "trueNegative": int(tn), "falseNegative": int(fn)}
    return {"kind": "imbalance", "caseKey": case_key, "seed": seed, "threshold": threshold, "trainingCount": len(y_train), "validationCount": len(y_valid), "positiveCount": int(np.sum(y_valid)), "negativeCount": int(len(y_valid) - np.sum(y_valid)), "baseline": metrics(baseline_prediction), "classifier": metrics(prediction)}

def accepted(case):
    metric = case["classifier"]
    if case["caseKey"] == "recall-first": return metric["recall"] >= .55 and metric["falseNegative"] < case["baseline"]["falseNegative"] and metric["precision"] >= .15
    if case["caseKey"] == "precision-first": return metric["precision"] >= .45 and metric["truePositive"] >= 8
    if case["caseKey"] == "balanced-outcome": return metric["precision"] >= .45 and metric["recall"] >= .45
    return metric["accuracy"] >= .70 and abs(case["positiveCount"] - case["negativeCount"]) < 35

cases = []
for index, config in enumerate(CONFIGS):
    found = next((make_case(seed, config) for seed in range(17 + index, 700) if accepted(make_case(seed, config))), None)
    if found is None: raise ValueError("A metric-choice case could not be validated")
    cases.append(found)
json.dumps(cases)
`,regularization:`
import json
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

CONFIGS = [
  ("penalty-helps", 27, .27, 14, .000001, 2.0),
  ("penalty-hurts", 90, .12, 5, .01, 50.0),
  ("penalty-too-strong", 44, .19, 10, 1.0, 500.0),
  ("difference-is-noise", 70, .18, 7, .2, .45),
]

def make_case(seed, config):
    case_key, train_n, noise, degree, alpha_low, alpha_high = config
    rng = np.random.default_rng(seed)
    x_train = np.sort(rng.uniform(-1, 1, train_n))
    y_train = np.sin(3 * x_train) + rng.normal(0, noise, train_n)
    x_valid = rng.uniform(-1, 1, 160)
    y_valid = np.sin(3 * x_valid) + rng.normal(0, noise, 160)
    x_curve = np.linspace(-1, 1, 100)
    def fit(alpha):
        model = make_pipeline(PolynomialFeatures(degree, include_bias=False), StandardScaler(), Ridge(alpha=alpha))
        model.fit(x_train.reshape(-1, 1), y_train)
        return {"trainMse": round(float(mean_squared_error(y_train, model.predict(x_train.reshape(-1, 1)))), 3), "validationMse": round(float(mean_squared_error(y_valid, model.predict(x_valid.reshape(-1, 1)))), 3), "predictions": [{"x": round(float(x), 3), "y": round(float(y), 3)} for x, y in zip(x_curve, model.predict(x_curve.reshape(-1, 1)))]}
    return {"kind": "regularization", "caseKey": case_key, "seed": seed, "trainingCount": len(x_train), "validationCount": len(x_valid), "low": fit(alpha_low), "high": fit(alpha_high), "observations": [{"x": round(float(x), 3), "y": round(float(y), 3)} for x, y in zip(x_train, y_train)]}

def accepted(case):
    low, high = case["low"], case["high"]
    if case["caseKey"] == "penalty-helps": return high["trainMse"] > low["trainMse"] + .01 and high["validationMse"] < low["validationMse"] * .85
    if case["caseKey"] == "penalty-hurts": return high["trainMse"] > low["trainMse"] + .01 and high["validationMse"] > low["validationMse"] * 1.12
    if case["caseKey"] == "penalty-too-strong": return high["trainMse"] > low["trainMse"] + .05 and high["validationMse"] > low["validationMse"] * 1.3
    return abs(high["validationMse"] - low["validationMse"]) < .025

cases = []
for index, config in enumerate(CONFIGS):
    found = next((make_case(seed, config) for seed in range(17 + index, 700) if accepted(make_case(seed, config))), None)
    if found is None: raise ValueError("A regularization case could not be validated")
    cases.append(found)
json.dumps(cases)
`},t=new Map;let l=Promise.resolve();async function p(i){const{id:e,kind:a,variant:_}=i.data;try{if(!(a in o))throw new Error("Unknown experiment type");const n=await f(e);t.has(a)||(s(e,"Validating distinct session cases"),t.set(a,JSON.parse(await n.runPythonAsync(o[a]))));const d=t.get(a);self.postMessage({id:e,type:"result",result:d[Math.max(0,Math.floor(_))%d.length]})}catch(n){self.postMessage({id:e,type:"error",error:n instanceof Error?n.message:String(n)})}}self.onmessage=i=>{l=l.then(()=>p(i))}})();
