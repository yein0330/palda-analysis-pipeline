"""Train-only lexical diagnostic on frozen Validation; no Test or weight output.

This is a diagnostic reference, not a replacement for the shared neural Encoder.
Classes and allowed masks come from the frozen taxonomy, not evaluation labels.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer

from scripts.train_encoder_cpu_baseline import model_text
from scripts.train_encoder_transformer import (
    SENTIMENTS, SPECIAL_THEMES, choose_threshold, evaluate_predictions,
    read_jsonl, runtime_label_map,
)


def binary_scores(x, targets, weights, validation_x, seed, return_model=False):
    if not len(targets) or len(np.unique(targets)) == 1:
        model = {'constant':float(targets[0]) if len(targets) else 0}
    else:
        model = SGDClassifier(loss='log_loss', alpha=1e-5, max_iter=1000, tol=1e-3,
                          class_weight='balanced', random_state=seed, average=True)
        model.fit(x, targets, sample_weight=weights)
    probabilities = predict_binary(model,validation_x)
    return (probabilities,model) if return_model else probabilities


def predict_binary(model, x):
    if isinstance(model,dict):return np.full(x.shape[0],model['constant'])
    return model.predict_proba(x)[:,list(model.classes_).index(1)]


def predict_review_artifact(artifact, rows):
    if not rows:return []
    if any(r['sourceType']!='review' for r in rows):raise ValueError('Review artifact only accepts Review')
    x=artifact['vectorizer'].transform([model_text(r) for r in rows])
    scores={label:predict_binary(model,x) for label,model in artifact['reviewModels'].items()}
    mapping=artifact['mapping'];canonical=mapping['indexToCanonicalTheme']
    result=[]
    for i,row in enumerate(rows):
        allowed={canonical[j] for j in mapping['categoryMasks'][row['category']['level2Id']]}
        result.append({'itemId':row['itemId'],'reviewCandidates':[
            [label,float(values[i])] for label,values in scores.items() if label.split('::')[0] in allowed]})
    return result


def make_vectorizer(feature_mode='char'):
    char = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,5), min_df=2,
                          max_features=40000, sublinear_tf=True, dtype=np.float32)
    if feature_mode == 'char':
        return char
    if feature_mode != 'char-word':
        raise ValueError('Unknown lexical feature mode: '+feature_mode)
    word = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=2,
                          max_features=20000, sublinear_tf=True, dtype=np.float32)
    return Pipeline([('features',FeatureUnion([('char',char),('word',word)])),
                     ('normalize',Normalizer())])


def run(data_dir, output_dir, seed=20260901, save_review_artifact=False, save_primary_artifact=False,
        feature_mode='char'):
    if output_dir.exists():
        raise FileExistsError(output_dir)
    started = time.time()
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    train = read_jsonl(data_dir/'train.jsonl')
    validation = read_jsonl(data_dir/'validation.jsonl')
    mapping = runtime_label_map(json.loads((data_dir/'label_map.json').read_text()), 'canonical')
    canonical = mapping['indexToCanonicalTheme']
    vectorizer = make_vectorizer(feature_mode)
    # Multiple vectorizers must see the same replayable corpus, not a consumed generator.
    x = vectorizer.fit_transform([model_text(r) for r in train])
    vx = vectorizer.transform([model_text(r) for r in validation])
    weights = np.array([r['annotation']['sampleWeight'] for r in train], dtype=np.float32)
    review_mask = np.array([r['sourceType']=='review' for r in train])
    allowed = np.zeros((len(train), len(canonical)), dtype=bool)
    for i,r in enumerate(train):
        allowed[i,mapping['categoryMasks'][r['category']['level2Id']]] = True
    review_targets = [{f"{l['themeId']}::{l['sentiment']}" for l in r['labels']['reviewLabels']} for r in train]
    secondary_targets = [{l['themeId'] for l in r['labels']['qnaSecondaryThemes']} for r in train]
    review_scores, secondary_scores, review_models = {}, {}, {}
    for idx,theme in enumerate(canonical):
        if theme in SPECIAL_THEMES:
            continue
        mask = review_mask & allowed[:,idx]
        for sentiment in SENTIMENTS:
            label = theme+'::'+sentiment
            target = np.array([int(label in labels) for labels in review_targets])[mask]
            review_scores[label],review_models[label] = binary_scores(x[mask],target,weights[mask],vx,seed,return_model=True)
        mask = ~review_mask & allowed[:,idx]
        target = np.array([int(theme in labels) for labels in secondary_targets])[mask]
        secondary_scores[theme] = binary_scores(x[mask],target,weights[mask],vx,seed)
    primary_targets = [r['labels']['qnaPrimaryTheme']['themeId'] for r in train if r['sourceType']=='qna']
    if len(set(primary_targets)) < 2:
        raise ValueError('Lexical diagnostic requires at least two Q&A primary classes')
    primary_model = SGDClassifier(loss='log_loss', alpha=1e-5, max_iter=1000,tol=1e-3,
                                  class_weight='balanced',random_state=seed,average=True)
    primary_model.fit(x[~review_mask],primary_targets,sample_weight=weights[~review_mask])
    primary_scores = primary_model.predict_proba(vx)
    predictions = []
    for i,r in enumerate(validation):
        themes = {canonical[j] for j in mapping['categoryMasks'][r['category']['level2Id']]}
        out = {'itemId':r['itemId'],'sourceType':r['sourceType']}
        if r['sourceType']=='review':
            out.update(reviewTruth=sorted({f"{l['themeId']}::{l['sentiment']}" for l in r['labels']['reviewLabels']}),
                       reviewCandidates=[[l,float(s[i])] for l,s in review_scores.items() if l.split('::')[0] in themes])
        else:
            candidates = [(float(s),label) for label,s in zip(primary_model.classes_,primary_scores[i]) if label in themes]
            if not candidates:
                raise ValueError('No trained primary class allowed for Validation category')
            out.update(primaryTruth=r['labels']['qnaPrimaryTheme']['themeId'],primaryPrediction=str(max(candidates)[1]),
                       secondaryTruth=sorted({l['themeId'] for l in r['labels']['qnaSecondaryThemes']}),
                       secondaryCandidates=[[l,float(s[i])] for l,s in secondary_scores.items() if l in themes],
                       usableTruth=r['usable'],usableProbability=1.0)
        predictions.append(out)
    rt,_ = choose_threshold(predictions,'reviewCandidates','reviewTruth')
    st,_ = choose_threshold(predictions,'secondaryCandidates','secondaryTruth')
    metrics = evaluate_predictions(predictions,rt,st)
    config = {'epoch':0,'indexToCanonicalTheme':canonical,'reviewThreshold':rt,'secondaryThreshold':st,
              'model':feature_mode+'-tfidf+sgd-logistic-category-masked','seed':seed,'featureMode':feature_mode}
    report = {'schemaVersion':'palda-lexical-validation-diagnostic-v1','bestValidation':metrics,
              'elapsedSeconds':time.time()-started,'testSkipped':True,'weightsSaved':save_review_artifact or save_primary_artifact,
              'savedArtifactScope':[name for name,enabled in [('review',save_review_artifact),('qnaPrimary',save_primary_artifact)] if enabled],
              'records':{'train':len(train),'validation':len(validation)},'features':x.shape[1],'featureMode':feature_mode,
              'scriptSha256':script_sha256,
              'hashes':{name:hashlib.sha256((data_dir/name).read_bytes()).hexdigest()
                        for name in ('train.jsonl','validation.jsonl','label_map.json')},
              'warning':'Development diagnostic, not Test. Usable is constant True, not trained. No confidence calibration.'}
    (output_dir/'best').mkdir(parents=True)
    if save_review_artifact:
        joblib.dump({'schemaVersion':'palda-lexical-review-v1','vectorizer':vectorizer,
                    'reviewModels':review_models,'mapping':mapping,'seed':seed,'sourceHashes':report['hashes'],
                    'featureMode':feature_mode},
                    output_dir/'review-model.joblib',compress=3)
    if save_primary_artifact:
        joblib.dump({'schemaVersion':'palda-lexical-primary-v1','vectorizer':vectorizer,
                    'primaryModel':primary_model,'mapping':mapping,'seed':seed,'sourceHashes':report['hashes'],
                    'featureMode':feature_mode},
                    output_dir/'primary-model.joblib',compress=3)
    (output_dir/'best/training_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    (output_dir/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (output_dir/'best-validation-predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in predictions))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20260901)
    p.add_argument('--save-review-artifact',action='store_true')
    p.add_argument('--save-primary-artifact',action='store_true')
    p.add_argument('--feature-mode',choices=('char','char-word'),default='char')
    a=p.parse_args()
    print(json.dumps(run(a.data_dir,a.output_dir,a.seed,a.save_review_artifact,a.save_primary_artifact,a.feature_mode),ensure_ascii=False,indent=2))
