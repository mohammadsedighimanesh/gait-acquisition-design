"""Versioned raw feature reconstruction and fixed-split grouped stacking diagnostic."""
import os
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(key, "1")
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.io import loadmat
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "idea1_2026"
OUT = DATA / "corrected_raw_pipeline_diagnostic_v2"
SOURCE = DATA / "DATA.mat"
LEGACY = DATA / "no_leakage_tabular_modeling" / "no_leakage_trial_level_features.csv"
SPLITS = DATA / "final_100x_repeated_group_validation" / "final_100x_split_records.csv"
SEED = 20261010
SIGNALS = [f"{prefix}_{joint}_{plane}" for prefix in ["R", "M"] for joint in ["ANK", "MT", "MP"] for plane in ["AB", "INV", "DF"]]
ANTH = ["WEIGHT", "HEIGHT", "RF_LENGTH", "LF_LENGTH", "RF_EXT_LENGTH", "LF_EXT_LENGTH", "RF_WIDTH", "LF_WIDTH",
        "RF_HEEL_WIDTH", "LF_HEEL_WIDTH", "RF_MALLEOLUS_HEIGHT", "LF_MALLEOLUS_HEIGHT", "RF_FINGER_SHAPE", "LF_FINGER_SHAPE", "AGE"]
TOES = ["RF_FINGER_SHAPE_first", "LF_FINGER_SHAPE_first"]
SCENARIOS = ["Legacy_matched", "Raw_nonpressure_legacy_pressure", "Corrected_raw", "Corrected_without_pressure"]
BASES = ["Logistic", "ExtraTrees", "RandomForest"]
COUNTS = {"base_classifier_fits": 0, "participant_meta_fits": 0}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def summarize_signal(a):
    # Match pandas trial statistics: sample SD for 100-frame signals.
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max()), "median": float(np.median(a))}


def build_raw(legacy):
    import h5py
    people = np.atleast_1d(loadmat(SOURCE, squeeze_me=True, struct_as_record=False)["PARTICIPANT"])
    assert len(people) == 70
    rows = []
    source_labels = {}
    with h5py.File(OUT / "raw_pressure_dataset_v2.h5", "w") as h:
        pset = h.create_dataset("pressure_raw", shape=(350, 40, 40, 100), dtype="float32",
                               chunks=(1, 40, 40, 100), compression="gzip", compression_opts=1)
        ids, tids, targets = [], [], []
        for pid, person in enumerate(people, 1):
            source_labels[pid] = int(person.FPI_CLASSIFICATION)
            for tid, trial in enumerate(np.atleast_1d(person.TRIAL), 1):
                pressure = np.asarray(trial.PRESS, dtype=float)
                assert pressure.shape == (40, 40, 100) and np.isfinite(pressure).all() and (pressure >= 0).all()
                pset[len(rows)] = pressure.astype(np.float32)
                ids.append(pid); tids.append(tid); targets.append(source_labels[pid])
                record = {"participant_id": pid, "trial_id": tid, "y": source_labels[pid]}
                record.update({f"{name}_first": float(getattr(person, name)) for name in ANTH})
                for name in SIGNALS:
                    a = np.asarray(getattr(trial, name), dtype=float).reshape(-1)
                    assert a.shape == (100,) and np.isfinite(a).all()
                    record.update({f"{name}_{stat}": value for stat, value in summarize_signal(a).items()})
                record.update({f"PMAX_{stat}": value for stat, value in summarize_signal(np.asarray(trial.PMAX)) .items()})
                positive = pressure[pressure > 0]
                total = pressure.sum(axis=(0, 1))
                record.update({"pressure_positive_mean": float(positive.mean()), "pressure_positive_std": float(positive.std()),
                    "pressure_positive_max": float(positive.max()), "pressure_active_fraction": float((pressure > 0).mean()),
                    "pressure_total_mean": float(total.mean()), "pressure_total_std": float(total.std()),
                    "pressure_total_max": float(total.max()), "pressure_peak_frame": float(np.argmax(total)+1),
                    "pressure_temporal_center": float(np.sum(total*np.arange(1,101))/total.sum())})
                rows.append(record)
        h.create_dataset("participant_id", data=np.array(ids, dtype=np.int32))
        h.create_dataset("trial_id", data=np.array(tids, dtype=np.int32))
        h.create_dataset("targets", data=np.array(targets, dtype=np.int32))
        h.attrs["pressure_unit"] = "N/cm^2"
        h.attrs["time_axis"] = "100 normalized stance samples, not elapsed time"
        h.attrs["preprocessing"] = "source pressure, float32 storage; no additional smoothing, normalization or clipping"
        h.attrs["model_scaling"] = "fit any learned transform on current training participants only"
        h.attrs["source_sha256"] = sha(SOURCE)
    raw = pd.DataFrame(rows)
    assert len(raw) == 350 and not raw.duplicated(["participant_id", "trial_id"]).any()
    centered = [c for c in legacy if c.endswith("_subject_centered")]
    for column in centered:
        parent = column.removesuffix("_subject_centered")
        raw[column] = raw[parent]-raw.groupby("participant_id")[parent].transform("mean")
    old_split = legacy.set_index(["participant_id", "trial_id"]).split
    raw["split"] = [old_split.loc[(p,t)] for p,t in zip(raw.participant_id,raw.trial_id)]
    assert set(raw.columns) == set(legacy.columns)
    raw = raw[legacy.columns]
    assert np.array_equal(raw.y, legacy.y) and np.array_equal(raw.participant_id, legacy.participant_id)
    assert np.isfinite(raw.select_dtypes(include="number")).all().all()
    raw.to_csv(OUT / "raw_trial_level_features_v2.csv", index=False)
    return raw


class Cleaner(TransformerMixin, BaseEstimator):
    """Remove constants and duplicate columns using current training rows only."""
    def fit(self, X, y=None):
        a = np.asarray(X)
        keep = []
        buckets = {}
        for j in range(a.shape[1]):
            if np.std(a[:,j]) <= 1e-12:
                continue
            key = hashlib.sha256(np.round(a[:,j],10).tobytes()).hexdigest()
            if any(np.allclose(a[:,j],a[:,k],rtol=1e-10,atol=1e-10) for k in buckets.get(key, [])):
                continue
            keep.append(j)
            buckets.setdefault(key, []).append(j)
        self.keep_ = np.array(keep)
        assert len(keep) > 0
        return self

    def transform(self, X):
        return np.asarray(X)[:,self.keep_]


class AdaptiveSelect(TransformerMixin, BaseEstimator):
    def __init__(self, k=60):
        self.k = k

    def fit(self, X, y):
        with np.errstate(divide="ignore",invalid="ignore"):
            scores, _ = f_classif(X,y)
        order = np.argsort(-np.nan_to_num(scores,nan=-np.inf),kind="stable")
        self.keep_ = np.sort(order[:min(self.k,X.shape[1])])
        return self

    def transform(self, X):
        return X[:,self.keep_]


def pipeline(method, columns, seed):
    numeric = [c for c in columns if c not in TOES]
    prep = ColumnTransformer([
        ("numeric", Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),numeric),
        ("toe_shape",OneHotEncoder(categories=[[1,2,3],[1,2,3]],handle_unknown="ignore",sparse_output=False),TOES)])
    if method == "Logistic":
        clf = LogisticRegression(C=.3,class_weight="balanced",max_iter=3000,tol=1e-7,random_state=seed)
    elif method == "ExtraTrees":
        clf = ExtraTreesClassifier(n_estimators=300,min_samples_leaf=1,max_features=.5,class_weight="balanced",random_state=seed,n_jobs=2)
    else:
        clf = RandomForestClassifier(n_estimators=300,min_samples_leaf=2,max_features="sqrt",class_weight="balanced",random_state=seed,n_jobs=2)
    steps = [("preprocess",prep),("clean",Cleaner())]
    if method == "Logistic":
        steps.append(("selection",AdaptiveSelect(60)))
    steps.append(("clf",clf))
    return Pipeline(steps)


def fit_base(method, X, y, ids, seed):
    model = pipeline(method,list(X.columns),seed)
    count = pd.Series(ids).value_counts()
    weights = np.array([1/count[i] for i in ids])
    model.fit(X,y,clf__sample_weight=weights)
    assert np.array_equal(model.named_steps["clf"].classes_,np.arange(3))
    COUNTS["base_classifier_fits"] += 1
    return model


def participant_table(ids, y, probabilities):
    tmp = pd.DataFrame({"participant_id":ids,"y":y})
    for c in range(3):
        tmp[f"p{c}"] = probabilities[:,c]
    assert tmp.groupby("participant_id").y.nunique().max() == 1
    return tmp.groupby("participant_id",sort=True).agg({"y":"first", "p0":"mean", "p1":"mean", "p2":"mean"})


def metric(y, p):
    return {"balanced_accuracy":balanced_accuracy_score(y,p.argmax(axis=1)),"accuracy":accuracy_score(y,p.argmax(axis=1)),
        "macro_f1":f1_score(y,p.argmax(axis=1),average="macro",labels=[0,1,2]),"log_loss":log_loss(y,p,labels=[0,1,2]),
        "brier":float(np.mean(np.sum((p-np.eye(3)[y])**2,axis=1))),"auc":roc_auc_score(y,p,multi_class="ovr",average="macro")}


def summarize(pred, labels):
    rows=[]
    for keys,g in pred.groupby(["scenario","repeat","split","model"]):
        rows.append(dict(zip(["scenario","repeat","split","model"],keys),**metric(g.y_true.to_numpy(),g[["p0","p1","p2"]].to_numpy())))
    repeated=pd.DataFrame(rows)
    repeated.to_csv(OUT/"participant_metrics_by_repeat.csv",index=False)
    summary=repeated.groupby(["scenario","split","model"]).agg(n_repeats=("repeat","nunique"),
        balanced_accuracy_mean=("balanced_accuracy","mean"),balanced_accuracy_std=("balanced_accuracy","std"),
        accuracy_mean=("accuracy","mean"),macro_f1_mean=("macro_f1","mean"),log_loss_mean=("log_loss","mean"),
        brier_mean=("brier","mean"),auc_mean=("auc","mean")).reset_index()
    summary.to_csv(OUT/"participant_performance_summary.csv",index=False)
    test=pred.loc[pred.split=="test"]
    contrasts=[]
    rng=np.random.default_rng(SEED+7000)
    observed=np.sort(test.participant_id.unique())
    classes=np.array([labels[int(i)] for i in observed])
    draws=np.column_stack([rng.choice(np.flatnonzero(classes==c),size=(5000,sum(classes==c)),replace=True) for c in range(3)])
    paired=test.assign(correct=(test.y_true==test.y_pred).astype(float)).pivot(
        index=["model","repeat","participant_id","y_true"],columns="scenario",values="correct")
    for model in BASES+["SoftVoting","GroupedStacking"]:
        g=paired.loc[model].reset_index()
        for a,b in [("Corrected_raw","Legacy_matched"),("Corrected_raw","Raw_nonpressure_legacy_pressure"),
                    ("Corrected_raw","Corrected_without_pressure")]:
            counts=g.groupby(["repeat","y_true"]).participant_id.transform("size")
            contribution=(g[a]-g[b])/(5*3*counts)
            by_person=pd.Series(contribution.to_numpy(),index=g.participant_id).groupby(level=0).sum().reindex(observed,fill_value=0).to_numpy()
            lo,hi=np.quantile(by_person[draws].sum(axis=1),[.025,.975])
            contrasts.append({"model":model,"contrast":f"{a} - {b}","balanced_accuracy_delta":by_person.sum(),
                "conditional_ci_low":lo,"conditional_ci_high":hi,"observed_test_participants":len(observed),
                "ci_note":"paired participant bootstrap conditional on fitted predictions; test exposure retained; no refit"})
    contrasts=pd.DataFrame(contrasts)
    contrasts.to_csv(OUT/"paired_representation_contrasts.csv",index=False)
    test_summary=summary.loc[summary.split=="test"].copy()
    lines=["# Corrected Raw Pipeline: Bounded Diagnostic", "",
        "Status: COMPLETED. Primary reporting is participant-level, not trial-level.",
        "Five first recorded splits retained: 49 train, 10 validation, 11 test people each.",
        "No hyperparameter or ensemble-weight search, seed search or validation-based model selection.",
        "All scenarios use grouped model training; Legacy_matched is not a literal reproduction of historical buggy stacking.",
        "Toe-shape categories are restored from raw metadata and one-hot encoded consistently in all scenarios.", "",
        "| Representation | Model | Balanced accuracy | Macro F1 | Brier |", "|---|---|---:|---:|---:|"]
    for r in test_summary.itertuples():
        lines.append(f"| {r.scenario} | {r.model} | {r.balanced_accuracy_mean:.4f} | {r.macro_f1_mean:.4f} | {r.brier_mean:.4f} |")
    lines += ["", "## Interpretation Limits", "",
        "This diagnostic separates pressure representation changes while fixing group-disjoint evaluation. It does not isolate every historical preprocessing defect or certify the original 100-repeat results.",
        "Intervals condition on fitted models. Only people appearing in these five test sets contribute; shared training dependence is not fully captured.",
        "The source is repeatedly examined, healthy male, single-cohort data. Improvements are exploratory, not independent clinical or novelty validation.",
        "All five trials are available for subject-centered features and participant probability averaging. Single-trial deployment is outside this protocol.",
        "Raw PMAX is preserved separately from tensor peak; source differences are not silently corrected.",
        "pressure_total_* sums pressure cells; without cell area calibration these values are not ground reaction force in N.",
        "Models for old pressure tensors were not retrained here. The new HDF5 stores physical pressure and leaves any model normalization to current training folds.",
        "Parameters: Logistic C=.3 and top60 inside each fit; ExtraTrees 300 trees, leaf1, max_features=.5; RandomForest 300 trees, leaf2, sqrt; meta Logistic C=.5.",
        "GroupedStacking trains the meta-model on 49 participant-level out-of-fold probability vectors; every base fit excludes all trials of each inner held-out person.",
        "No participants, classes or hard cases were removed."]
    (OUT/"RESULTS_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    print(test_summary.to_string(index=False))
    print(contrasts.loc[contrasts.model.isin(["Logistic","GroupedStacking"])].to_string(index=False))


def selftest():
    rng=np.random.default_rng(5)
    frame=pd.DataFrame({"a":rng.normal(size=45),"a_copy":0.,"constant":1.,TOES[0]:np.arange(45)%3+1,TOES[1]:np.arange(45)%3+1})
    frame["a_copy"]=frame.a
    y=np.arange(45)%3
    model=fit_base("Logistic",frame,y,np.repeat(np.arange(9),5),5)
    assert len(model.named_steps['clean'].keep_)<model.named_steps['preprocess'].transform(frame).shape[1]
    assert np.allclose(model.predict_proba(frame).sum(axis=1),1)
    print("SELFTEST PASSED: training-only constants/duplicates, categorical encoding, finite predictions.")


def prepare():
    if OUT.exists():
        raise FileExistsError(f"Existing diagnostic preserved: {OUT}")
    import h5py
    legacy=pd.read_csv(LEGACY)
    assert not legacy.duplicated(["participant_id","trial_id"]).any()
    OUT.mkdir()
    raw=build_raw(legacy)
    dump("prepared_source_manifest.json",{"status":"RAW_PREPARATION_COMPLETED","source_sha256":sha(SOURCE),
        "legacy_features_sha256":sha(LEGACY),"raw_features_sha256":sha(OUT/"raw_trial_level_features_v2.csv"),
        "raw_h5_sha256":sha(OUT/"raw_pressure_dataset_v2.h5"),"h5py_version":h5py.__version__,
        "preparation_python":sys.version,"raw_rows":len(raw),"actual_model_fits":0})
    print(f"Prepared raw features and pressure: {OUT}")


def run():
    if not (OUT/"prepared_source_manifest.json").exists():
        raise FileNotFoundError("First execute --prepare-only in an existing environment with h5py.")
    if (OUT/"completion.json").exists() or (OUT/"frozen_diagnostic_config.json").exists():
        raise FileExistsError("Existing model experiment preserved.")
    prepared=json.loads((OUT/"prepared_source_manifest.json").read_text())
    for key,path in [('source_sha256',SOURCE),('legacy_features_sha256',LEGACY),
                     ('raw_features_sha256',OUT/'raw_trial_level_features_v2.csv'),('raw_h5_sha256',OUT/'raw_pressure_dataset_v2.h5')]:
        assert sha(path)==prepared[key]
    legacy=pd.read_csv(LEGACY)
    splits=pd.read_csv(SPLITS).iloc[:5].copy()
    assert len(splits)==5
    config={"created_utc_before_modelling":datetime.now(timezone.utc).isoformat(),"source_sha256":sha(SOURCE),
        "legacy_features_sha256":sha(LEGACY),"split_records_sha256":sha(SPLITS),"script_sha256":sha(Path(__file__)),
        "splits":"first five original recorded splits", "n_repeats":5,"seed":SEED,"scenario_order":SCENARIOS,
        "models":BASES+["SoftVoting","GroupedStacking"],"inner_group_folds":3,"fixed_logistic_C":.3,"fixed_meta_C":.5,
        "trees":300,"feature_selector_K":60,"sample_weight":"each participant total base-training weight one",
        "categorical_features":TOES,"toe_categories":[1,2,3],"no_model_selection_on_val_or_test":True,
        "pressure_units":"source N/cm^2, no extra smoothing, normalization or clipping",
        "raw_statistic_std":"signals sample SD; positive pressure cell population SD; source scalar PMAX SD zero",
        "primary_metric":"participant balanced accuracy","bootstrap_draws":5000,"exploratory":True,
        "versions":{"python":sys.version,"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__,"preparation_h5py":prepared['h5py_version']}}
    dump("frozen_diagnostic_config.json",config)
    raw=pd.read_csv(OUT/"raw_trial_level_features_v2.csv")
    features=[c for c in raw if c not in ["participant_id","trial_id","y","split"]]
    assert len(features)==223 and not any("FPI" in c.upper() for c in features)
    pressure_cols=[c for c in features if c.startswith("PMAX_") or c.startswith("pressure_")]
    assert len(pressure_cols)==28
    old=legacy.copy()
    for c in TOES:
        old[c]=raw[c]
    hybrid=raw.copy()
    hybrid[pressure_cols]=legacy[pressure_cols]
    datasets={"Legacy_matched":old,"Raw_nonpressure_legacy_pressure":hybrid,"Corrected_raw":raw,
              "Corrected_without_pressure":raw.drop(columns=pressure_cols)}
    dump("feature_dictionary.json",{"predictors":features,"pressure_columns":pressure_cols,
        "source_units":{"WEIGHT":"kg","HEIGHT":"cm","AGE":"years","foot_dimensions":"mm","R_*":"degrees","M_*":"Nm/kg","pressure":"N/cm^2"},
        "positive_pressure_mean":"mean among raw >0 contact cells", "active_fraction":"fraction of raw cells >0",
        "temporal_coordinates":"1-based normalized stance frame", "pressure_total":"sum of cell pressures; not calibrated force",
        "subject_centered":"base minus own five-trial mean; batch acquisition assumption"})
    splits.to_csv(OUT/"fixed_split_records.csv",index=False)
    predictions,base_oof_rows,inner_rows,preprocessing_rows=[],[],[],[]
    labels=raw.groupby("participant_id").y.first().to_dict()
    for scenario,table in datasets.items():
        columns=[c for c in features if c in table]
        for split in splits.itertuples():
            repeat=int(split.repeat)
            roles={r:np.array(list(map(int,getattr(split,f"{r}_participants").split(',')))) for r in ["train","val","test"]}
            assert all(set(roles[a]).isdisjoint(roles[b]) for a,b in [('train','val'),('train','test'),('val','test')])
            assert set.union(*map(set,roles.values()))==set(range(1,71))
            train=table.loc[table.participant_id.isin(roles['train'])].copy()
            seed=SEED+repeat
            participants=np.sort(roles['train'])
            person_labels=np.array([labels[int(i)] for i in participants])
            group_folds=list(StratifiedKFold(3,shuffle=True,random_state=seed).split(participants,person_labels))
            oof=np.full((len(train),9),np.nan)
            print(f"{scenario}: repeat {repeat+1}/5",flush=True)
            for inner_fold,(it,iv) in enumerate(group_folds):
                fit_ids,hold_ids=participants[it],participants[iv]
                trmask=train.participant_id.isin(fit_ids).to_numpy()
                vamask=train.participant_id.isin(hold_ids).to_numpy()
                assert set(fit_ids).isdisjoint(hold_ids) and set(hold_ids).isdisjoint(roles['test'])
                for role,ids in [("train",fit_ids),("validation",hold_ids)]:
                    inner_rows.extend({"scenario":scenario,"repeat":repeat,"inner_fold":inner_fold,"role":role,"participant_id":int(i)} for i in ids)
                for j,method in enumerate(BASES):
                    model=fit_base(method,train.loc[trmask,columns],train.loc[trmask,'y'].to_numpy(),train.loc[trmask,'participant_id'].to_numpy(),seed)
                    p=model.predict_proba(train.loc[vamask,columns])
                    oof[vamask,3*j:3*j+3]=p
                    for k,(_,row) in enumerate(train.loc[vamask].iterrows()):
                        base_oof_rows.append({"scenario":scenario,"repeat":repeat,"inner_fold":inner_fold,"model":method,
                            "participant_id":int(row.participant_id),"trial_id":int(row.trial_id),"y_true":int(row.y),
                            "p0":p[k,0],"p1":p[k,1],"p2":p[k,2]})
            assert np.isfinite(oof).all()
            meta_x=pd.DataFrame(oof,index=train.participant_id).groupby(level=0).mean().sort_index()
            meta_y=np.array([labels[int(i)] for i in meta_x.index])
            meta=Pipeline([("scaler",StandardScaler()),("clf",LogisticRegression(C=.5,class_weight="balanced",max_iter=3000,tol=1e-7,random_state=seed))])
            meta.fit(meta_x.to_numpy(),meta_y)
            COUNTS['participant_meta_fits']+=1
            models={method:fit_base(method,train[columns],train.y.to_numpy(),train.participant_id.to_numpy(),seed) for method in BASES}
            for method,model in models.items():
                preprocessing_rows.append({"scenario":scenario,"repeat":repeat,"model":method,"input_columns":len(columns),
                    "after_clean_columns":len(model.named_steps['clean'].keep_),
                    "classifier_columns":len(model.named_steps['selection'].keep_) if method=='Logistic' else len(model.named_steps['clean'].keep_)})
            for role in ['val','test']:
                held=table.loc[table.participant_id.isin(roles[role])].copy()
                outputs={method:participant_table(held.participant_id.to_numpy(),held.y.to_numpy(),model.predict_proba(held[columns])) for method,model in models.items()}
                order=outputs['Logistic'].index
                components=[outputs[m].loc[order,['p0','p1','p2']].to_numpy() for m in BASES]
                probs={m:components[j] for j,m in enumerate(BASES)}
                probs['SoftVoting']=np.mean(components,axis=0)
                probs['GroupedStacking']=meta.predict_proba(np.column_stack(components))
                for method,p in probs.items():
                    assert np.isfinite(p).all() and np.allclose(p.sum(axis=1),1)
                    for j,pid in enumerate(order):
                        predictions.append({"scenario":scenario,"repeat":repeat,"split":role,"model":method,"participant_id":int(pid),
                            "y_true":int(labels[int(pid)]),"y_pred":int(p[j].argmax()),"p0":p[j,0],"p1":p[j,1],"p2":p[j,2]})
            pd.DataFrame(predictions).to_csv(OUT/"participant_predictions.csv",index=False)
            dump("progress.json",{"last_completed":[scenario,repeat],**COUNTS})
    pd.DataFrame(base_oof_rows).to_csv(OUT/"grouped_base_oof_predictions.csv",index=False)
    pd.DataFrame(inner_rows).to_csv(OUT/"stacking_inner_split_audit.csv",index=False)
    pd.DataFrame(preprocessing_rows).to_csv(OUT/"training_preprocessing_summary.csv",index=False)
    pred=pd.DataFrame(predictions)
    assert len(pred)==4*5*5*21 and not pred.duplicated(['scenario','repeat','split','model','participant_id']).any()
    summarize(pred,labels)
    assert COUNTS=={'base_classifier_fits':240,'participant_meta_fits':20}
    for name,expected in [('source_sha256',SOURCE),('legacy_features_sha256',LEGACY),('split_records_sha256',SPLITS)]:
        assert sha(expected)==config[name]
    dump("completion.json",{"status":"COMPLETED",**COUNTS,"actual_model_fits":sum(COUNTS.values()),
        "participants":70,"raw_trials":350,"prediction_rows":len(pred),"original_sources_unchanged":True,"environment_changed":False})
    dump("output_manifest.json",{p.name:sha(p) for p in OUT.iterdir() if p.is_file()})
    print(f"Saved: {OUT}")


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        selftest()
    elif args.prepare_only:
        prepare()
    else:
        try:
            run()
        except Exception as exc:
            if OUT.exists() and not (OUT/'completion.json').exists():
                dump('failure.json',{'error':repr(exc),**COUNTS})
            raise
