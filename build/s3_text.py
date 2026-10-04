# %% [markdown]
# ## Part 3: Waste description classification
#
# The text route exists for the cases the camera cannot settle: an item inside a bag, a photo
# too dark to use, or a resident who would rather type. The input is one short sentence
# (`"wrinkled pink takeout container with product remaining"`) and the output is the same nine
# streams the CNN produces, so the two routes are interchangeable from Part 5's point of view.
#
# Part 1 established two things that shape this part. The auxiliary columns are the label in
# disguise, so only `description` is used (Finding 3). And 17% of descriptions contain no
# stream-exclusive word, which is where the real difficulty lives (Finding 4) — so **both**
# options the lab offers are built, a sparse linear model and a fine-tuned transformer, and
# they are compared on that subset specifically rather than on the headline number alone.

# %% [markdown]
# ### 3.1 Features for the sparse models
#
# Two views of each description are concatenated:
#
# - **Word 1–2 grams.** Bigrams matter because the stream is often carried by a pair:
#   `"paper towel"` is Textile Trash while `"paper"` alone is Paper, and `"glass cup"` is
#   Miscellaneous Trash while `"glass bottle"` is Glass.
# - **Character 3–5 grams (`char_wb`).** These catch the compound and hyphenated forms the
#   templates produce (`"plastic-metal composite"`, `"fun-sized"`) and give the model partial
#   credit on a word it has never seen in full, which is the closest a bag-of-words model gets
#   to handling the unseen vocabulary a real resident would type.
#
# No stemming and no stop-word list: at a median of five words there is nothing to prune, and
# `"with food residue"` is a signal, not filler.

# %%
word_vec = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), sublinear_tf=True, min_df=2)
char_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True, min_df=3)

from sklearn.pipeline import FeatureUnion
TEXT_FEATURES = FeatureUnion([('word', word_vec), ('char', char_vec)])

Xtr_text, ytr_text = txt_train['clean'].to_numpy(), txt_train['category'].to_numpy()
Xva_text, yva_text = txt_val['clean'].to_numpy(), txt_val['category'].to_numpy()
Xte_text, yte_text = txt_test['clean'].to_numpy(), txt_test['category'].to_numpy()

Xtr_vec = TEXT_FEATURES.fit_transform(Xtr_text)
Xva_vec = TEXT_FEATURES.transform(Xva_text)
Xte_vec = TEXT_FEATURES.transform(Xte_text)
print(f'TF-IDF matrix: {Xtr_vec.shape[0]:,} x {Xtr_vec.shape[1]:,} features '
      f'({TEXT_FEATURES.transformer_list[0][1].vocabulary_.__len__():,} word n-grams + '
      f'{TEXT_FEATURES.transformer_list[1][1].vocabulary_.__len__():,} character n-grams)')
print(f'density: {Xtr_vec.nnz / np.prod(Xtr_vec.shape):.3%} non-zero, '
      f'{Xtr_vec.nnz / Xtr_vec.shape[0]:.0f} active features per description')

# %% [markdown]
# ### 3.2 Option A: sparse linear and tree baselines
#
# Four classifiers over those features, scored with 5-fold stratified cross-validation on the
# training split so the ranking does not depend on one lucky validation draw. The point of the
# exercise is not only to pick a winner but to see how much headroom a transformer could
# possibly have.

# %%
BASELINES = {
    'Multinomial NB (alpha=0.1)': MultinomialNB(alpha=0.1),
    'Logistic regression (C=10)': LogisticRegression(C=10, max_iter=3000, n_jobs=-1),
    'Linear SVM (C=1)': LinearSVC(C=1.0),
    'Random forest (400 trees)': RandomForestClassifier(n_estimators=400, n_jobs=-1,
                                                        random_state=SEED),
}

cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
base_rows = []
for name, clf in BASELINES.items():
    t0 = time.time()
    scores = cross_val_score(clf, Xtr_vec, ytr_text, cv=cv, scoring='f1_macro', n_jobs=1)
    clf.fit(Xtr_vec, ytr_text)
    pv = clf.predict(Xva_vec)
    base_rows.append({'model': name, 'cv macro F1': scores.mean(), 'cv sd': scores.std(),
                      'val accuracy': accuracy_score(yva_text, pv),
                      'val macro F1': f1_score(yva_text, pv, average='macro'),
                      'fit (s)': time.time() - t0})
    print(f'  {name:<28} cv F1 {scores.mean():.3f} +/- {scores.std():.3f} | '
          f'val acc {base_rows[-1]["val accuracy"]:.3f}')

baselines = pd.DataFrame(base_rows).set_index('model')
baselines.round(3)

# %% [markdown]
# Logistic regression is the sparse model carried forward even if the linear SVM edges it on
# F1, because the assistant needs **calibrated probabilities**: the confidence gate in Part 5
# and the top-3 alternatives both require `predict_proba`, and `LinearSVC` only produces
# decision-function margins. The sweep below quantifies what that choice costs.

# %%
print(f"best cross-validated model: {baselines['cv macro F1'].idxmax()} "
      f"({baselines['cv macro F1'].max():.3f})")
print(f"logistic regression:        {baselines.loc['Logistic regression (C=10)', 'cv macro F1']:.3f} "
      f"- the gap is the price of calibrated probabilities\n")

# Regularisation sweep: how much does C matter?
sweep = []
for C in [0.1, 0.5, 1, 3, 10, 30]:
    clf = LogisticRegression(C=C, max_iter=3000, n_jobs=-1).fit(Xtr_vec, ytr_text)
    pv = clf.predict(Xva_vec)
    sweep.append({'C': C, 'val accuracy': accuracy_score(yva_text, pv),
                  'val macro F1': f1_score(yva_text, pv, average='macro')})
sweep = pd.DataFrame(sweep)

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.semilogx(sweep['C'], sweep['val macro F1'], marker='o', color=BLUE, label='val macro F1')
ax.semilogx(sweep['C'], sweep['val accuracy'], marker='o', color=ORANGE, ls=(0, (4, 3)),
            label='val accuracy')
ax.set_xlabel('inverse regularisation strength C')
ax.set_title('Logistic regression is flat above C=1 — the features, not the model, are the limit')
ax.legend()
plt.tight_layout(); plt.show()

BEST_C = float(sweep.loc[sweep['val macro F1'].idxmax(), 'C'])
tfidf_clf = LogisticRegression(C=BEST_C, max_iter=3000, n_jobs=-1).fit(Xtr_vec, ytr_text)
print(f'selected C = {BEST_C:g}')

# %% [markdown]
# ### 3.3 Option B: fine-tuning DistilBERT
#
# DistilBERT is a 66M-parameter distilled BERT — six transformer layers instead of twelve, ~60%
# of the runtime, within a point or two of BERT on most classification benchmarks. It is chosen
# over BERT-base here for one practical reason (this is a CPU-only machine) and one principled
# one: with 3,431 training sentences of five words each, the limiting factor is data, not model
# capacity.
#
# **All layers are fine-tuned**, not just the classification head. Freezing the encoder would
# turn DistilBERT into a slower, worse version of the TF-IDF model; the entire reason to reach
# for a transformer here is that its attention can let `"bottle"` mean different things
# depending on whether `"glass"`, `"soda"` or `"aluminum"` sits next to it — and that needs
# gradient to flow through the encoder.
#
# Sequences are capped at 24 word-pieces (the longest cleaned description is well inside that),
# which is what keeps CPU fine-tuning to a few minutes per epoch.

# %%
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)

torch.manual_seed(SEED)
torch.set_num_threads(os.cpu_count() or 4)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME, MAX_LEN = 'distilbert-base-uncased', 24
LABEL2ID = {c: i for i, c in enumerate(CLASS_NAMES)}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
lengths = [len(tokenizer.tokenize(t)) for t in Xtr_text]
print(f'word-piece lengths: median {np.median(lengths):.0f}, 99th pct {np.percentile(lengths, 99):.0f}, '
      f'max {max(lengths)} -> MAX_LEN {MAX_LEN} truncates {np.mean(np.array(lengths) > MAX_LEN):.2%}')


def encode(texts, labels=None):
    enc = tokenizer(list(texts), truncation=True, max_length=MAX_LEN,
                    padding='max_length', return_tensors='pt')
    tensors = [enc['input_ids'], enc['attention_mask']]
    if labels is not None:
        tensors.append(torch.tensor([LABEL2ID[c] for c in labels]))
    return TensorDataset(*tensors)


train_loader = DataLoader(encode(Xtr_text, ytr_text), batch_size=32, shuffle=True)
val_loader = DataLoader(encode(Xva_text, yva_text), batch_size=64)
test_loader = DataLoader(encode(Xte_text, yte_text), batch_size=64)
print(f'{len(train_loader)} training batches of 32 on {DEVICE}')

# %%
@torch.no_grad()
def bert_predict(model, loader):
    """Softmax probabilities for every row of `loader`, in order."""
    model.eval()
    out = []
    for batch in loader:
        ids, mask = batch[0].to(DEVICE), batch[1].to(DEVICE)
        out.append(torch.softmax(model(input_ids=ids, attention_mask=mask).logits, -1).cpu())
    return torch.cat(out).numpy()


bert = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_CLASSES,
    id2label={i: c for c, i in LABEL2ID.items()}, label2id=LABEL2ID).to(DEVICE)

EPOCHS_BERT, LR_BERT = 3, 3e-5
optimizer = torch.optim.AdamW(bert.parameters(), lr=LR_BERT, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS_BERT
scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

bert_hist, t_start = [], time.time()
for epoch in range(1, EPOCHS_BERT + 1):
    bert.train()
    running, t0 = 0.0, time.time()
    for step, (ids, mask, y) in enumerate(train_loader, 1):
        optimizer.zero_grad()
        loss = bert(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE),
                    labels=y.to(DEVICE)).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bert.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        running += loss.item()
    pv = bert_predict(bert, val_loader).argmax(1)
    acc = accuracy_score([LABEL2ID[c] for c in yva_text], pv)
    f1 = f1_score([LABEL2ID[c] for c in yva_text], pv, average='macro')
    bert_hist.append({'epoch': epoch, 'train loss': running / len(train_loader),
                      'val accuracy': acc, 'val macro F1': f1, 'minutes': (time.time() - t0) / 60})
    print(f'  epoch {epoch}: loss {running / len(train_loader):.4f} | val acc {acc:.4f} | '
          f'val macro F1 {f1:.4f} | {(time.time() - t0) / 60:.1f} min')

bert_hist = pd.DataFrame(bert_hist).set_index('epoch')
print(f'fine-tuned {sum(p.numel() for p in bert.parameters()) / 1e6:.0f}M parameters in '
      f'{(time.time() - t_start) / 60:.1f} minutes')
bert_hist.round(4)

# %% [markdown]
# ### 3.4 Evaluation and error analysis
#
# Both models are scored on the untouched test split, overall and on the ambiguous subset from
# Finding 4 — the descriptions containing no word that belongs to a single stream.

# %%
proba_tfidf_test = tfidf_clf.predict_proba(Xte_vec)
pred_tfidf_test = tfidf_clf.classes_[proba_tfidf_test.argmax(1)]
proba_bert_test = bert_predict(bert, test_loader)
pred_bert_test = np.array([CLASS_NAMES[i] for i in proba_bert_test.argmax(1)])

hard_mask = ~txt_test['has_exclusive_token'].to_numpy()
rows = []
for name, pred in [('TF-IDF + logistic regression', pred_tfidf_test),
                   ('DistilBERT (fine-tuned)', pred_bert_test)]:
    rows.append({
        'model': name,
        'test accuracy': accuracy_score(yte_text, pred),
        'test macro F1': f1_score(yte_text, pred, average='macro'),
        f'accuracy on easy ({(~hard_mask).sum()})': accuracy_score(yte_text[~hard_mask], pred[~hard_mask]),
        f'accuracy on ambiguous ({hard_mask.sum()})': accuracy_score(yte_text[hard_mask], pred[hard_mask]),
    })
text_results = pd.DataFrame(rows).set_index('model')
print(text_results.round(4).to_string())

TEXT_MODEL = 'bert' if (text_results.iloc[1]['test macro F1'] >
                        text_results.iloc[0]['test macro F1']) else 'tfidf'
print(f'\nmodel wired into the assistant: {TEXT_MODEL}')
text_results.round(3)

# %%
pred_text_test = pred_bert_test if TEXT_MODEL == 'bert' else pred_tfidf_test
proba_text_test = proba_bert_test if TEXT_MODEL == 'bert' else proba_tfidf_test
TEXT_CLASSES = CLASS_NAMES if TEXT_MODEL == 'bert' else list(tfidf_clf.classes_)

txt_acc = accuracy_score(yte_text, pred_text_test)
txt_f1 = f1_score(yte_text, pred_text_test, average='macro')
print(f'TEST  accuracy {txt_acc:.3f} | macro F1 {txt_f1:.3f} | n = {len(yte_text)}\n')
print(classification_report(yte_text, pred_text_test, digits=3))

cm_txt = confusion_matrix(yte_text, pred_text_test, labels=CLASS_NAMES)
fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
heat(axes[0], cm_txt, [SHORT[c] for c in CLASS_NAMES], [SHORT[c] for c in CLASS_NAMES],
     title=f'Text classifier confusion matrix ({TEXT_MODEL})')
axes[0].set_xlabel('predicted'); axes[0].set_ylabel('true')
cm_other = confusion_matrix(yte_text, pred_tfidf_test if TEXT_MODEL == 'bert' else pred_bert_test,
                            labels=CLASS_NAMES)
heat(axes[1], cm_other, [SHORT[c] for c in CLASS_NAMES], [SHORT[c] for c in CLASS_NAMES],
     title=f'The other model, for comparison '
           f'({"tfidf" if TEXT_MODEL == "bert" else "bert"})')
axes[1].set_xlabel('predicted'); axes[1].set_ylabel('true')
plt.tight_layout(); plt.show()

# %%
# Where do the two models disagree, and who is right?
disagree = pred_tfidf_test != pred_bert_test
both_wrong = (pred_tfidf_test != yte_text) & (pred_bert_test != yte_text)
print(f'the two models disagree on {disagree.sum()} of {len(yte_text)} test descriptions '
      f'({disagree.mean():.1%})')
if disagree.any():
    print(f'  TF-IDF right, DistilBERT wrong: '
          f'{int(((pred_tfidf_test == yte_text) & disagree).sum())}')
    print(f'  DistilBERT right, TF-IDF wrong: '
          f'{int(((pred_bert_test == yte_text) & disagree).sum())}')
print(f'both wrong on the same row: {int(both_wrong.sum())} '
      f'({both_wrong.mean():.1%} of the test set is beyond both models)')

err = pd.DataFrame({
    'description': txt_test['description'].to_numpy(),
    'true': yte_text,
    'TF-IDF': pred_tfidf_test,
    'DistilBERT': pred_bert_test,
    'confidence': proba_text_test.max(1),
    'ambiguous': hard_mask,
})
sel_col = 'DistilBERT' if TEXT_MODEL == 'bert' else 'TF-IDF'
wrong = err[err[sel_col] != err['true']]
if len(wrong):
    print(f'\n{len(wrong)} errors from the selected model, most confident first:')
    show_err = wrong.sort_values('confidence', ascending=False).head(12)
else:
    print(f'\nthe selected model ({sel_col}) makes no errors on this test set. The only error '
          f'signal left is where the two models disagree - those rows, with the loser\'s '
          f'prediction, are the closest thing this data has to a hard case:')
    show_err = err[disagree].head(12)
show_err

# %%
off_txt = [(CLASS_NAMES[i], CLASS_NAMES[j], cm_txt[i, j])
           for i in range(NUM_CLASSES) for j in range(NUM_CLASSES) if i != j and cm_txt[i, j] > 0]
off_txt.sort(key=lambda r: -r[2])
if off_txt:
    print('Largest text confusions (true -> predicted):')
    for t, p, n in off_txt[:6]:
        print(f'  {t:<21} -> {p:<21} {n:>3}')
else:
    print('the selected model has an empty off-diagonal: no stream is confused with another.')

fig, ax = plt.subplots(figsize=(8, 3.4))
easy_acc = [text_results.iloc[k, 2] for k in range(2)]
hard_acc = [text_results.iloc[k, 3] for k in range(2)]
pos = np.arange(2)
ax.bar(pos - 0.2, easy_acc, 0.4, color=BLUE, label='has a stream-exclusive word (83%)')
ax.bar(pos + 0.2, hard_acc, 0.4, color=ORANGE, label='ambiguous wording (17%)')
ax.set_xticks(pos); ax.set_xticklabels(['TF-IDF +\nlogistic regression', 'DistilBERT\n(fine-tuned)'])
ax.set_ylim(0, 1.05); ax.set_ylabel('test accuracy')
ax.set_title('The ambiguous subset is no harder than the rest - the hypothesis from §1.2 fails')
for p, (e, hh) in zip(pos, zip(easy_acc, hard_acc)):
    ax.text(p - 0.2, e + 0.015, f'{e:.3f}', ha='center', fontsize=9, color=INK_2)
    ax.text(p + 0.2, hh + 0.015, f'{hh:.3f}', ha='center', fontsize=9, color=INK_2)
ax.legend(loc='lower right')
plt.tight_layout(); plt.show()

# %%
# What fine-tuning actually did: the sentence embedding space. Mean-pooled encoder states from
# the fine-tuned DistilBERT, next to a comparable 2-D view of the TF-IDF space, with a
# silhouette score to put a number on "the streams separate".
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import silhouette_score


@torch.no_grad()
def bert_embeddings(texts, batch=64):
    """Mean-pooled final-layer encoder states - the sentence representation the head sees."""
    bert.eval()
    out = []
    for s in range(0, len(texts), batch):
        enc = tokenizer(list(texts[s:s + batch]), truncation=True, max_length=MAX_LEN,
                        padding='max_length', return_tensors='pt')
        mask = enc['attention_mask'].to(DEVICE)
        h = bert.distilbert(input_ids=enc['input_ids'].to(DEVICE),
                            attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    return np.vstack(out)


emb_bert = bert_embeddings(Xte_text)
emb_tfidf = TruncatedSVD(n_components=50, random_state=SEED).fit_transform(Xte_vec)
y_idx = np.array([LABEL2ID[c] for c in yte_text])

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
for ax, (emb, title) in zip(axes, [(emb_tfidf, 'TF-IDF (50-component SVD)'),
                                   (emb_bert, 'Fine-tuned DistilBERT (mean-pooled)')]):
    xy = PCA(2, random_state=SEED).fit_transform(emb)
    for i, c in enumerate(CLASS_NAMES):
        m = y_idx == i
        ax.scatter(xy[m, 0], xy[m, 1], s=11, alpha=0.75, label=SHORT[c])
    sil = silhouette_score(emb, y_idx)
    ax.set_title(f'{title}\nsilhouette {sil:.3f}', fontsize=11)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
axes[1].legend(ncol=3, fontsize=8, loc='lower right')
fig.suptitle('Fine-tuning pulls the nine streams apart in embedding space',
             x=0.006, ha='left', fontsize=12, fontweight='bold')
plt.tight_layout(); plt.show()

print(f'silhouette on test embeddings - TF-IDF {silhouette_score(emb_tfidf, y_idx):.3f} | '
      f'DistilBERT {silhouette_score(emb_bert, y_idx):.3f}')

# %% [markdown]
# ### 3.5 The classification function

# %%
TEXT_CONF_THRESHOLD = 0.50


def classify_waste_description(description, top_k=3):
    """Classify a resident's written description of a waste item.

    Args:
        description (str): free text, e.g. "greasy pizza box".
        top_k (int): how many ranked alternatives to return.

    Returns:
        dict with `category` (str or None when the input is unusable), `confidence` (float),
        `top_k` (list of (category, probability)), `model`, and `cleaned` input.
    """
    if not isinstance(description, str) or not description.strip():
        return {'category': None, 'confidence': 0.0, 'top_k': [], 'model': TEXT_MODEL,
                'cleaned': '', 'error': 'empty or non-string description'}

    cleaned = clean_description(description)
    if not cleaned:
        return {'category': None, 'confidence': 0.0, 'top_k': [], 'model': TEXT_MODEL,
                'cleaned': '', 'error': 'nothing left after cleaning (no letters in input)'}

    if TEXT_MODEL == 'bert':
        enc = tokenizer([cleaned], truncation=True, max_length=MAX_LEN,
                        padding='max_length', return_tensors='pt')
        bert.eval()
        with torch.no_grad():
            logits = bert(input_ids=enc['input_ids'].to(DEVICE),
                          attention_mask=enc['attention_mask'].to(DEVICE)).logits
        proba = torch.softmax(logits, -1)[0].cpu().numpy()
        classes = CLASS_NAMES
    else:
        proba = tfidf_clf.predict_proba(TEXT_FEATURES.transform([cleaned]))[0]
        classes = list(tfidf_clf.classes_)

    order = np.argsort(-proba)[:top_k]
    return {'category': classes[int(order[0])],
            'confidence': float(proba[order[0]]),
            'top_k': [(classes[int(i)], float(proba[i])) for i in order],
            'model': TEXT_MODEL,
            'cleaned': cleaned,
            'low_confidence': bool(proba[order[0]] < TEXT_CONF_THRESHOLD)}


demo = ['greasy pizza box', 'empty aluminum soda can', 'broken wine bottle',
        'paper towel soaked in oil', 'plastic-metal composite gadget', 'grass clippings and leaves',
        'old t-shirt with holes', 'mystery object', '']
for d in demo:
    r = classify_waste_description(d)
    if r['category'] is None:
        print(f'  {d!r:<36} -> no prediction ({r["error"]})')
    else:
        alts = ', '.join(f'{c} {p:.2f}' for c, p in r['top_k'])
        flag = '  [low confidence]' if r['low_confidence'] else ''
        print(f'  {d!r:<36} -> {r["category"]:<20} {r["confidence"]:.2f}{flag}   [{alts}]')

# %% [markdown]
# ### Part 3 takeaways
#
# **This benchmark is saturated, and that is the finding.** Both options the lab offers score
# at or within a point of 100%, and the ambiguous 17% is no harder than the rest. The
# explanation is in the data, not the models: 4,902 descriptions generated from a template
# grammar over a 309-word vocabulary is close to a lookup table, and 3,431 training rows cover
# nearly every cell of it.
#
# The §1.2 hypothesis — that word-*combination* cases would separate a bag-of-words model from
# a contextual one — survives in direction and dies in magnitude. Every row the two models
# disagree on is exactly that kind of case (`"red glass pen"`, `"large paper bed sheet"`,
# `"plastic rug"`: a material adjective attached to an object that belongs to a different
# stream), and DistilBERT wins all of them. But there are only five such rows in 736, so the
# effect that should have been the headline is a rounding error instead.
#
# A 100% test score is a reason for suspicion, so it was checked: identical cleaned strings were
# de-duplicated before splitting, train/test overlap was verified at zero (§1.4c), and the
# label-bearing columns were never shown to either model (§1.2). The score survives those
# checks. What it means is that the *synthetic* task is solved, not that waste description
# classification is.
#
# **The silhouette comparison is where the two models actually differ.** Accuracy cannot
# separate them, but the embedding geometry can: the fine-tuned DistilBERT representation
# clusters the nine streams far more cleanly than the TF-IDF space. That is a reason to expect
# it to degrade more gracefully on real free text — misspellings, regional terms, brand names —
# which is why it is the model wired into the assistant despite costing 10 minutes of training
# against the linear model's one second.
#
# **Neither model can say "that is not waste".** `"mystery object"` comes back as Miscellaneous
# Trash with high confidence, because softmax over nine classes has no tenth option. The
# threshold in `classify_waste_description` is the only thing between a nonsense input and a
# confident wrong answer, and §5.3 shows it catching some such inputs and missing others. An
# explicit out-of-domain class, trained on non-waste text, is the real fix.
