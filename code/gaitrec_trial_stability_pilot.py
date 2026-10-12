"""Development-only transfer test of recorded-trial decision/explanation stability."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import json
import hashlib
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'idea1_2026/gaitrec_source_vertical_v1'
OUT=ROOT/'idea1_2026/gaitrec_trial_stability_pilot_v1'
SEED=20261030
CLASS_NAMES=['HC','H','K','A','C']
CS=[.1,1.,10.]
SUBSETS=[s for k in range(1,6) for s in itertools.combinations(range(5),k)]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def dump(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def build():
    if OUT.exists(): raise FileExistsError(f'Preserving existing experiment:{OUT}')
    manifest=json.loads((SOURCE/'source_manifest.json').read_text())
    for entry in manifest:
        assert sha(SOURCE/entry['file'])==entry['sha256']
    meta=pd.read_csv(SOURCE/'GRF_metadata.csv')
    assert meta.SESSION_ID.is_unique
    trainpeople=set(meta.loc[meta.TRAIN==1,'SUBJECT_ID']); testpeople=set(meta.loc[meta.TEST==1,'SUBJECT_ID'])
    assert not trainpeople&testpeople
    dev=meta.query('TRAIN==1 and SPEED==2 and SHOD_CONDITION==1 and ORTHOPEDIC_INSOLE==0').copy()
    assert dev.groupby('SUBJECT_ID').CLASS_LABEL.nunique().eq(1).all()
    dev['date']=pd.to_datetime(dev.SESSION_DATE,format='%d.%m.%Y',errors='raise')
    OUT.mkdir(); (OUT/'models').mkdir()
    protocol=dict(purpose='methodological transfer of exhaustive recorded-trial stability audit, not FPI external validation',
        dataset='pinned GaitRec processed vertical GRF only, left/right101stance samples,202predictors',
        official_test='all TEST participants reserved; no test waveform modeling or outcome performance inspection',
        cohort='official TRAIN; self-selected speed2; normal shoes1; no orthopedic insole0; first chronological eligible session with >=5 valid paired trials',
        bag='five lowest trial IDs in first eligible session, label-blind; exhaust all31nonempty subsets, no best-subset selection',
        model='fixed five-trial Mean or Median pooling -> train-only StandardScaler -> PCA12(fullSVD) -> StandardScaler -> balanced multinomial Logistic',
        C_grid=CS,selection='3 inner person folds mean logloss,tie smallerC;3repeats x5person outer folds',
        evaluation='same fixed five-trial-trained model applied to each subset; expected accuracy/BA across subsets, not average probabilities across all trials',
        primary='within-Median expected BA at5minus1; secondary3minus5 with exploratory engineering noninferiority margin3percentage points',
        explanation='fixed full-bag winner versus runner-up linear logit contrast;10groups(left/right x5equal normalized-stance bins), top3group-set stability across all >=3trial subsets',
        stability='finite recorded-subset invariance, not correctness, future-session guarantee or physical clinical phase annotation',
        seed=SEED,conditional_bootstrap='4000class-stratified person draws retaining repeats; no retraining uncertainty',
        source_hashes={v['file']:v['sha256'] for v in manifest},script_sha256=sha(Path(__file__)),
        observed_source_metadata_people=int(meta.SUBJECT_ID.nunique()),official_train_people=len(trainpeople),
        prior_art='aggregation previously studied (Slijepcevic2020); no claim of novel mean/median/PCA/logistic algorithm',
        environment_changed=False)
    dump('protocol_before_fit.json',protocol)
    sideframes=[]
    for side in ['left','right']:
        frames=[]
        for chunk in pd.read_csv(SOURCE/f'GRF_F_V_PRO_{side}.csv',chunksize=8000):
            # Discard official test/other sessions before inspecting waveform values.
            chunk=chunk[chunk.SESSION_ID.isin(dev.SESSION_ID)].copy()
            if len(chunk): frames.append(chunk)
        frame=pd.concat(frames,ignore_index=True)
        assert not frame.duplicated(['SUBJECT_ID','SESSION_ID','TRIAL_ID']).any()
        sideframes.append(frame.set_index(['SUBJECT_ID','SESSION_ID','TRIAL_ID']).sort_index())
    left,right=sideframes
    assert left.index.equals(right.index)
    assert left.shape[1]==right.shape[1]==101
    vectors=np.column_stack([left.to_numpy(float),right.to_numpy(float)])
    finite=np.isfinite(vectors).all(axis=1)
    # Filtered force curves can have small negative excursions; do not delete them.
    valid=finite & (vectors[:,:101].mean(axis=1)>0) & (vectors[:,101:].mean(axis=1)>0)
    keys=left.index.to_frame(index=False)
    keys['row_position']=np.arange(len(keys)); keys['valid']=valid
    bags=[]; people=[]; targets=[]; provenance=[]; excluded=[]
    for pid,g in dev.groupby('SUBJECT_ID'):
        found=False
        for session in g.sort_values(['date','SESSION_ID']).itertuples():
            trials=keys.query('SUBJECT_ID==@pid and SESSION_ID==@session.SESSION_ID and valid').sort_values('TRIAL_ID')
            if len(trials)<5: continue
            chosen=trials.iloc[:5]
            bags.append(vectors[chosen.row_position.to_numpy()]);people.append(int(pid));targets.append(CLASS_NAMES.index(session.CLASS_LABEL))
            provenance.append(dict(person_id=int(pid),session_id=int(session.SESSION_ID),class_name=session.CLASS_LABEL,
                trial_ids=';'.join(map(str,chosen.TRIAL_ID.astype(int))),eligible_trials_in_session=len(trials),
                age=float(session.AGE) if np.isfinite(session.AGE) else None,sex=int(session.SEX) if np.isfinite(session.SEX) else None,
                affected_side=int(session.AFFECTED_SIDE) if np.isfinite(session.AFFECTED_SIDE) else None))
            found=True;break
        if not found: excluded.append(int(pid))
    X=np.asarray(bags); y=np.asarray(targets); people=np.asarray(people)
    assert X.shape==(len(people),5,202) and len(set(people))==len(people) and not set(people)&testpeople
    np.savez_compressed(OUT/'development_bags.npz',X=X,y=y,person_ids=people)
    pd.DataFrame(provenance).to_csv(OUT/'selected_development_sessions.csv',index=False)
    dump('cohort_audit.json',dict(eligible_condition_people=int(dev.SUBJECT_ID.nunique()),selected_people=len(people),
        selected_real_trials=len(people)*5,class_counts={CLASS_NAMES[c]:int(sum(y==c)) for c in range(5)},
        excluded_no_five_valid_paired_trials=excluded,source_matched_trial_rows=len(keys),invalid_matched_trial_rows=int(sum(~valid)),
        negative_values_in_matched_processed_curves=int(np.sum(vectors<0)),
        official_test_people_excluded_from_modeling=True,source_source_ids_left_right_aligned=True,
        scope='only2vertical GRF channels, condition-matched subset; not all GaitRec signals/populations'))
    return X,y,people


def pooled(X,method):
    return X.mean(axis=1) if method=='Mean' else np.median(X,axis=1)


def model(C):
    return Pipeline([('input_scale',StandardScaler()),('pca',PCA(n_components=12,svd_solver='full')),
        ('latent_scale',StandardScaler()),('classifier',LogisticRegression(C=C,class_weight='balanced',tol=1e-7,max_iter=4000,random_state=SEED))])


def expected_metrics(g):
    yy=g.y_true.to_numpy(int); guessed=g.y_pred.to_numpy(int)
    repetitions=g.groupby('person_id').size()
    w=1/g.person_id.map(repetitions).to_numpy(float)
    cm=np.zeros((5,5)); np.add.at(cm,(yy,guessed),w)
    tp=cm.diagonal(); pp=g[[f'p{c}' for c in range(5)]].to_numpy()
    return dict(expected_accuracy=float(tp.sum()/cm.sum()),expected_balanced_accuracy=float(np.mean(tp/cm.sum(axis=1))),
        macro_f1_expected_confusion=float(np.mean(np.divide(2*tp,cm.sum(axis=0)+cm.sum(axis=1),out=np.zeros(5),where=cm.sum(axis=0)+cm.sum(axis=1)>0))),
        expected_log_loss=float(np.average(-np.log(np.maximum(pp[np.arange(len(pp)),yy],np.finfo(float).eps)),weights=w)))


def explanation_audit(estimator,features,subset_predictions):
    inp=estimator.named_steps['input_scale']; pca=estimator.named_steps['pca']; latent=estimator.named_steps['latent_scale']; clf=estimator.named_steps['classifier']
    effective=(clf.coef_/latent.scale_[None])@pca.components_
    z=inp.transform(features)
    logits=estimator.decision_function(features)
    order=np.argsort(logits[-1]); winner=int(order[-1]); runner=int(order[-2])
    contributions=z*(effective[winner]-effective[runner])
    offset=(clf.intercept_[winner]-clf.intercept_[runner])-(latent.mean_/latent.scale_)@(clf.coef_[winner]-clf.coef_[runner])-pca.mean_@(effective[winner]-effective[runner])
    np.testing.assert_allclose(contributions.sum(axis=1)+offset,logits[:,winner]-logits[:,runner],atol=1e-9)
    group=[]
    for side in range(2):
        for positions in np.array_split(np.arange(101),5):
            group.append(contributions[:,side*101+positions].sum(axis=1))
    group=np.asarray(group).T
    ranks=np.argsort(-np.abs(group),axis=1,kind='stable')[:,:3]
    reference=set(ranks[-1]); eligible=np.array([len(s)>=3 for s in SUBSETS])
    stable=bool(np.all(subset_predictions[eligible]==winner))
    changed=bool(any(set(r)!=reference for r in ranks[eligible]))
    jaccard=min(len(set(r)&reference)/len(set(r)|reference) for r in ranks[eligible])
    return dict(full_predicted_class=winner,full_runner_up=runner,decision_stable_delete_up_to2=stable,
        top3_group_set_changes=changed,minimum_top3_jaccard=float(jaccard),
        maximum_logit_additivity_error=float(np.max(np.abs(contributions.sum(axis=1)+offset-(logits[:,winner]-logits[:,runner])))))


def run():
    X,y,people=build()
    assert min(np.bincount(y))>=15
    outer=[]; inner=[]; jobs=[]
    for r in range(3):
        for f,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=SEED+r).split(people,y)):
            batches=list(StratifiedKFold(3,shuffle=True,random_state=SEED+100+r*10+f).split(tr,y[tr]))
            for role,indices in [('train',tr),('test',te)]:
                outer.extend(dict(repeat=r,fold=f,role=role,person_id=int(people[i])) for i in indices)
            for k,(it,iv) in enumerate(batches):
                for role,indices in [('train',tr[it]),('validation',tr[iv])]:
                    inner.extend(dict(repeat=r,fold=f,inner_fold=k,role=role,person_id=int(people[i])) for i in indices)
            jobs.append((r,f,tr,te,batches))
    pd.DataFrame(outer).to_csv(OUT/'outer_person_splits_before_fit.csv',index=False)
    pd.DataFrame(inner).to_csv(OUT/'inner_person_splits_before_fit.csv',index=False)
    records=[]; audit=[]; tuning=[]; fits=0
    for r,f,tr,te,batches in jobs:
        for method in ['Mean','Median']:
            losses=[]
            for C in CS:
                ls=[]
                for it,iv in batches:
                    estimator=model(C).fit(pooled(X[tr[it]],method),y[tr[it]]); fits+=1
                    ls.append(log_loss(y[tr[iv]],estimator.predict_proba(pooled(X[tr[iv]],method)),labels=range(5)))
                losses.append(float(np.mean(ls)))
            chosen=min(range(len(CS)),key=lambda i:(losses[i],CS[i]))
            tuning.extend(dict(repeat=r,fold=f,method=method,C=C,inner_log_loss=losses[i],selected=i==chosen) for i,C in enumerate(CS))
            estimator=model(CS[chosen]).fit(pooled(X[tr],method),y[tr]); fits+=1
            joblib.dump(estimator,OUT/'models'/f'r{r}_f{f}_{method}.joblib')
            for index in te:
                features=np.asarray([pooled(X[index:index+1,list(s),:],method)[0] for s in SUBSETS])
                probabilities=estimator.predict_proba(features); guesses=probabilities.argmax(axis=1)
                records.extend(dict(repeat=r,fold=f,method=method,person_id=int(people[index]),y_true=int(y[index]),
                    subset=';'.join(str(v+1) for v in s),k=len(s),y_pred=int(guess),**{f'p{c}':float(prob[c]) for c in range(5)})
                    for s,prob,guess in zip(SUBSETS,probabilities,guesses))
                audit.append(dict(repeat=r,fold=f,method=method,person_id=int(people[index]),y_true=int(y[index]),
                    full_prediction_correct=bool(guesses[-1]==y[index]),**explanation_audit(estimator,features,guesses)))
        print(f'Repeat{r+1}/3 fold{f+1}/5 | actual fits={fits}',flush=True)
    pred=pd.DataFrame(records); pred.to_csv(OUT/'heldout_subset_predictions.csv',index=False)
    pd.DataFrame(audit).to_csv(OUT/'heldout_decision_explanation_stability.csv',index=False)
    pd.DataFrame(tuning).to_csv(OUT/'inner_tuning.csv',index=False)
    per=pd.DataFrame([dict(method=m,k=k,repeat=r,**expected_metrics(g)) for (m,k,r),g in pred.groupby(['method','k','repeat'])])
    per.to_csv(OUT/'expected_metrics_by_repeat.csv',index=False)
    summary=per.groupby(['method','k']).mean(numeric_only=True).drop(columns='repeat').reset_index()
    summary.to_csv(OUT/'trial_budget_summary.csv',index=False)
    contrasts=[]
    draws=[np.random.default_rng(SEED+9000+c).choice(np.flatnonzero(y==c),size=(4000,int(sum(y==c))),replace=True) for c in range(5)]
    for method in ['Mean','Median']:
        values={}
        for k in range(1,6):
            g=pred.query('method==@method and k==@k').copy()
            g['correct']=(g.y_true==g.y_pred).astype(float)
            values[k]=g.groupby('person_id').correct.mean().reindex(people).to_numpy()
        for candidate,baseline in [(5,1),(3,5)]:
            delta=values[candidate]-values[baseline]
            boot=np.mean([delta[d].mean(axis=1) for d in draws],axis=0)
            point=np.mean([delta[y==c].mean() for c in range(5)])
            lo,hi=np.quantile(boot,[.025,.975])
            contrasts.append(dict(method=method,candidate_k=candidate,baseline_k=baseline,BA_delta=point,
                conditional95_low=lo,conditional95_high=hi))
    pd.DataFrame(contrasts).to_csv(OUT/'paired_trial_budget_contrasts.csv',index=False)
    assert fits==300 and len(pred)==len(people)*3*2*31
    dump('completion.json',dict(status='COMPLETED_DEVELOPMENT_ONLY',actual_model_fits=fits,outer_models=30,
        selected_people=len(people),subset_prediction_rows=len(pred),official_test_used_for_modeling=False,
        FPI_external_validation=False,environment_changed=False))
    print(summary.to_string(index=False))


if __name__=='__main__':
    run()
