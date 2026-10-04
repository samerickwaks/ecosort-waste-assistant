# %% [markdown]
# ## Part 2: Waste material classification with a CNN
#
# 3,325 training photographs is far too few to train a convolutional network from scratch, so
# the model is built by **transfer learning**: an ImageNet backbone supplies general visual
# features (edges, textures, materials, object parts) and only a small head is learned from
# RealWaste. The part is organised as four decisions, each supported by a measurement rather
# than a preference:
#
# 1. **Which backbone?** (§2.2 — a frozen-feature probe compares two)
# 2. **How wide and how regularised a head?** (§2.3 — a grid on cached features)
# 3. **Frozen features or fine-tuned?** (§2.4–2.5 — two training stages, measured separately)
# 4. **Where does it still fail, and is that acceptable for the product?** (§2.6)

# %% [markdown]
# ### 2.1 Image preprocessing and augmentation
#
# Two transformations are needed, and both belong **inside** the model so that the exported
# artefact is self-contained and inference cannot silently disagree with training:
#
# - **Scaling.** Each ImageNet backbone expects its own input range. MobileNetV2 wants
#   `[-1, 1]`, which `Rescaling(1/127.5, -1)` reproduces exactly; EfficientNet carries its own
#   normalisation inside the graph and wants raw `[0, 255]`. Whichever backbone §2.2 selects,
#   the matching scaling goes in as a layer, so the exported model cannot be fed wrongly.
# - **Augmentation.** Facility photographs are taken looking down at material on a floor, so
#   there is no canonical "up": horizontal *and* vertical flips plus free rotation are all
#   label-preserving. Brightness and contrast jitter stands in for the phone cameras the
#   assistant will really see (§1.1 showed the training images share one lighting setup).
#   Augmentation layers are active during `fit` and inert during `predict`, so no test-time
#   branch is needed.

# %%
augment = tf.keras.Sequential([
    layers.RandomFlip('horizontal_and_vertical', seed=SEED),
    layers.RandomRotation(0.15, fill_mode='reflect', seed=SEED),
    layers.RandomZoom(0.15, fill_mode='reflect', seed=SEED),
    layers.RandomContrast(0.15, seed=SEED),
    layers.RandomBrightness(0.10, value_range=(0, 255), seed=SEED),
], name='augmentation')

sample_img = tf.convert_to_tensor(
    np.asarray(Image.open(img_train.loc[0, 'path']).resize((IMG_WIDTH, IMG_HEIGHT)),
               dtype=np.float32))
fig, axes = plt.subplots(1, 6, figsize=(14, 2.6))
axes[0].imshow(sample_img.numpy().astype('uint8'))
axes[0].set_title(f"original\n{img_train.loc[0, 'label']}", fontsize=9)
for ax in axes[1:]:
    aug = augment(tf.expand_dims(sample_img, 0), training=True)[0]
    ax.imshow(tf.clip_by_value(aug, 0, 255).numpy().astype('uint8'))
    ax.set_title('augmented', fontsize=9)
for ax in axes:
    ax.axis('off')
fig.suptitle('Augmentation: flips, rotation, zoom, brightness and contrast jitter',
             x=0.006, ha='left', fontsize=12, fontweight='bold')
plt.tight_layout(); plt.show()

# Class weights for the 2.9:1 imbalance found in Part 1 (Finding 1).
class_weights = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES),
                                     y=img_train['label_idx'].to_numpy())
CLASS_WEIGHT = {i: float(w) for i, w in enumerate(class_weights)}
print('class weights (inverse frequency, normalised):')
for i, c in enumerate(CLASS_NAMES):
    print(f'  {c:<22} {CLASS_WEIGHT[i]:.2f}')

# %% [markdown]
# ### 2.2 Choosing a backbone: a frozen-feature probe
#
# MobileNetV2 and EfficientNetB0 are both ~5M-parameter ImageNet models with a 1280-dimensional
# pooled embedding, and both are plausible here. Rather than fine-tune both on CPU — hours of
# compute for one decision — each backbone's **frozen** embeddings are extracted once and a
# multinomial logistic regression is fitted on them. That measures the quality of the features
# the backbone already has, which is what transfer learning depends on, and it costs ~2 minutes
# per candidate.

# %%
def extract_features(backbone, preprocess, frame, batch=48):
    """Pooled embeddings for every row of `frame`, in order, with no augmentation."""
    out = []
    for s in range(0, len(frame), batch):
        batch_paths = frame['path'].iloc[s:s + batch]
        arr = np.stack([np.asarray(Image.open(p).resize((IMG_WIDTH, IMG_HEIGHT)),
                                   dtype=np.float32) for p in batch_paths])
        out.append(backbone.predict(preprocess(arr), verbose=0))
    return np.concatenate(out)


CANDIDATES = {
    'MobileNetV2': (tf.keras.applications.MobileNetV2,
                    tf.keras.applications.mobilenet_v2.preprocess_input),
    'EfficientNetB0': (tf.keras.applications.EfficientNetB0,
                       tf.keras.applications.efficientnet.preprocess_input),
}

probe_rows, FEATURES = [], {}
for name, (ctor, prep) in CANDIDATES.items():
    t0 = time.time()
    backbone = ctor(include_top=False, weights='imagenet',
                    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), pooling='avg')
    f_tr = extract_features(backbone, prep, img_train)
    f_va = extract_features(backbone, prep, img_val)
    extract_s = time.time() - t0

    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
    clf.fit(f_tr, img_train['label_idx'])
    pred = clf.predict(f_va)
    probe_rows.append({
        'backbone': name,
        'params (M)': backbone.count_params() / 1e6,
        'embedding': f_tr.shape[1],
        'val accuracy': accuracy_score(img_val['label_idx'], pred),
        'val macro F1': f1_score(img_val['label_idx'], pred, average='macro'),
        'extract (s)': extract_s,
    })
    FEATURES[name] = (f_tr, f_va)
    print(f'{name:<16} probed in {extract_s:.0f}s -> val acc {probe_rows[-1]["val accuracy"]:.3f}')
    release('backbone', 'clf')

probe = pd.DataFrame(probe_rows).set_index('backbone')
probe.round(3)

# %%
BACKBONE_NAME = probe['val macro F1'].idxmax()
print(f'selected backbone: {BACKBONE_NAME}')
F_TRAIN, F_VAL = FEATURES[BACKBONE_NAME]
y_train = img_train['label_idx'].to_numpy()
y_val = img_val['label_idx'].to_numpy()
y_test = img_test['label_idx'].to_numpy()

# %% [markdown]
# ### 2.3 Sizing the classification head on cached features
#
# The head is the only part learned from scratch, and its width, dropout and L2 are exactly the
# knobs the lab asks to tune. Tuning them on the **cached embeddings** costs seconds per
# configuration instead of three minutes per epoch, because the frozen backbone never has to run
# again — and a head trained on frozen features is mathematically the same object as stage-1
# transfer learning without augmentation. Six configurations are compared, from a bare linear
# probe to a two-layer head, on the validation split.

# %%
def build_head(units=(256,), dropout=0.4, l2=1e-4, in_dim=1280, name='head'):
    """Dense head over pooled CNN features. `units=()` gives a plain linear probe."""
    seq = [layers.Input(shape=(in_dim,))]
    for i, u in enumerate(units):
        seq += [layers.Dense(u, use_bias=False, kernel_regularizer=regularizers.l2(l2),
                             name=f'{name}_dense{i}'),
                layers.BatchNormalization(name=f'{name}_bn{i}'),
                layers.Activation('relu', name=f'{name}_relu{i}'),
                layers.Dropout(dropout, seed=SEED, name=f'{name}_drop{i}')]
    seq += [layers.Dense(NUM_CLASSES, activation='softmax',
                         kernel_regularizer=regularizers.l2(l2), name=f'{name}_out')]
    return tf.keras.Sequential(seq, name=name)


HEAD_GRID = [
    {'units': (), 'dropout': 0.0, 'l2': 0.0},
    {'units': (128,), 'dropout': 0.3, 'l2': 1e-4},
    {'units': (256,), 'dropout': 0.4, 'l2': 1e-4},
    {'units': (256,), 'dropout': 0.6, 'l2': 1e-3},
    {'units': (512,), 'dropout': 0.5, 'l2': 1e-4},
    {'units': (512, 128), 'dropout': 0.4, 'l2': 1e-4},
]

grid_rows = []
for cfg in HEAD_GRID:
    tf.keras.utils.set_random_seed(SEED)
    head = build_head(**cfg, in_dim=F_TRAIN.shape[1])
    head.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                 loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    hist = head.fit(F_TRAIN, y_train, validation_data=(F_VAL, y_val),
                    epochs=60, batch_size=64, verbose=0, class_weight=CLASS_WEIGHT,
                    callbacks=[tf.keras.callbacks.EarlyStopping(
                        monitor='val_accuracy', patience=10, restore_best_weights=True)])
    pv = head.predict(F_VAL, verbose=0)
    grid_rows.append({
        'units': str(cfg['units']) if cfg['units'] else 'linear',
        'dropout': cfg['dropout'], 'l2': cfg['l2'],
        'epochs run': len(hist.history['loss']),
        'train acc': hist.history['accuracy'][-1],
        'val acc': accuracy_score(y_val, pv.argmax(1)),
        'val macro F1': f1_score(y_val, pv.argmax(1), average='macro'),
    })
    print(f"  {grid_rows[-1]['units']:<10} drop {cfg['dropout']:.1f} l2 {cfg['l2']:.0e} -> "
          f"val acc {grid_rows[-1]['val acc']:.3f}")
    release('head')

grid = pd.DataFrame(grid_rows)
grid['overfit gap'] = grid['train acc'] - grid['val acc']
grid.sort_values('val macro F1', ascending=False).round(3)

# %%
fig, ax = plt.subplots(figsize=(8.4, 3.4))
g = grid.sort_values('val macro F1')
lbl = [f"{r.units}  d{r.dropout:g}  l2 {r.l2:g}" for r in g.itertuples()]
pos = np.arange(len(g))
ax.barh(pos - 0.19, g['val macro F1'], 0.36, color=BLUE, label='val macro F1')
ax.barh(pos + 0.19, g['overfit gap'], 0.36, color=ORANGE, label='train-val gap')
ax.set_yticks(pos); ax.set_yticklabels(lbl, fontsize=9)
ax.grid(axis='y', visible=False); ax.tick_params(axis='y', length=0)
ax.set_title('Head capacity buys little; regularisation buys a smaller generalisation gap')
ax.legend(loc='lower right')
plt.tight_layout(); plt.show()

BEST_HEAD = HEAD_GRID[int(grid['val macro F1'].idxmax())]
print(f'selected head: {BEST_HEAD}')

# %% [markdown]
# ### 2.4 Stage 1: train the head end-to-end with the backbone frozen
#
# The grid above saw clean, un-augmented features. Stage 1 re-trains the chosen head inside the
# full model, where augmentation is live, so the head learns to tolerate the jitter the real
# pipeline will produce. The backbone stays frozen, which keeps the ImageNet features intact
# while the randomly-initialised head is still producing large, destructive gradients.

# %%
def build_model(head_cfg, trainable_base=False, backbone_name=BACKBONE_NAME):
    """Full classifier: augmentation -> scaling -> backbone -> head, all in one artefact."""
    ctor, _ = CANDIDATES[backbone_name]
    base = ctor(include_top=False, weights='imagenet',
                input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), pooling='avg')
    base.trainable = trainable_base
    scale = (layers.Rescaling(1 / 127.5, offset=-1, name='preprocess')
             if backbone_name == 'MobileNetV2' else layers.Lambda(lambda z: z, name='preprocess'))

    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name='image')
    x = augment(inputs)
    x = scale(x)
    x = base(x, training=False)          # keep BatchNorm in inference mode (standard recipe)
    head = build_head(**head_cfg, in_dim=base.output_shape[-1])
    outputs = head(x)
    return tf.keras.Model(inputs, outputs, name=f'{backbone_name.lower()}_waste'), base, head


tf.keras.utils.set_random_seed(SEED)
model, base_model, head_model = build_model(BEST_HEAD, trainable_base=False)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
              metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2')])
print(f'trainable parameters: {int(sum(np.prod(w.shape) for w in model.trainable_weights)):,} '
      f'of {model.count_params():,}')
model.summary(line_length=96)

# %%
# Label smoothing (0.05) is a second regulariser alongside dropout and L2: with genuinely
# ambiguous photographs (a soiled cardboard tray is defensibly Misc. Trash) forcing the model
# to 1.0 confidence on a single stream is the wrong target.
EPOCHS_S1 = 5
t0 = time.time()
hist1 = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS_S1,
                  class_weight=CLASS_WEIGHT, verbose=2,
                  callbacks=[tf.keras.callbacks.ModelCheckpoint(
                      'cnn_stage1.weights.h5', save_weights_only=True,
                      monitor='val_accuracy', save_best_only=True)])
print(f'stage 1 finished in {(time.time() - t0) / 60:.1f} min')
model.load_weights('cnn_stage1.weights.h5')
s1_eval = model.evaluate(ds_val, verbose=0, return_dict=True)
print({k: round(v, 4) for k, v in s1_eval.items()})

# %% [markdown]
# ### 2.5 Stage 2: fine-tune the top of the backbone
#
# ImageNet's later blocks encode object-level concepts (dog faces, car wheels) that are not what
# separates crushed cardboard from crushed paper; its early blocks encode edges and textures that
# very much are. Fine-tuning therefore unfreezes only the **last 40 layers** and drops the
# learning rate by 100x, so the pre-trained features are nudged rather than overwritten.
# BatchNorm layers stay frozen — with batches of 32 their running statistics are noisy, and
# updating them is the most common way fine-tuning collapses.

# %%
UNFREEZE_LAST = 40
base_model.trainable = True
for lyr in base_model.layers[:-UNFREEZE_LAST]:
    lyr.trainable = False
for lyr in base_model.layers:
    if isinstance(lyr, layers.BatchNormalization):
        lyr.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
              metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2')])
print(f'now trainable: {int(sum(np.prod(w.shape) for w in model.trainable_weights)):,} parameters '
      f'({sum(l.trainable for l in base_model.layers)} of {len(base_model.layers)} backbone layers)')

EPOCHS_S2 = 6
t0 = time.time()
hist2 = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS_S2,
                  class_weight=CLASS_WEIGHT, verbose=2,
                  callbacks=[
                      tf.keras.callbacks.ModelCheckpoint(
                          'cnn_stage2.weights.h5', save_weights_only=True,
                          monitor='val_accuracy', save_best_only=True),
                      tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3,
                                                       restore_best_weights=True),
                      tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                                           patience=2, min_lr=1e-7, verbose=1),
                  ])
print(f'stage 2 finished in {(time.time() - t0) / 60:.1f} min')
model.load_weights('cnn_stage2.weights.h5')

# %%
h = {k: hist1.history[k] + hist2.history[k] for k in ['accuracy', 'val_accuracy', 'loss', 'val_loss']}
ep = np.arange(1, len(h['accuracy']) + 1)
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
for ax, (a, b, title) in zip(axes, [('accuracy', 'val_accuracy', 'Accuracy'),
                                    ('loss', 'val_loss', 'Loss')]):
    ax.plot(ep, h[a], color=MUTED, marker='o', ms=3.5, label='train')
    ax.plot(ep, h[b], color=BLUE, marker='o', ms=3.5, label='validation')
    ax.axvline(EPOCHS_S1 + 0.5, color=ORANGE, lw=1.3, ls=(0, (4, 3)))
    ax.text(EPOCHS_S1 + 0.62, ax.get_ylim()[0] + 0.04 * np.ptp(ax.get_ylim()),
            'fine-tuning starts', color=ORANGE, fontsize=9)
    ax.set_title(title); ax.set_xlabel('epoch'); ax.legend()
fig.suptitle('Two-stage transfer learning: the step at epoch 6 is the backbone unfreezing',
             x=0.006, ha='left', fontsize=12, fontweight='bold')
plt.tight_layout(); plt.show()

print(f"stage 1 best val accuracy: {max(hist1.history['val_accuracy']):.3f}")
print(f"stage 2 best val accuracy: {max(hist2.history['val_accuracy']):.3f}  "
      f"(+{max(hist2.history['val_accuracy']) - max(hist1.history['val_accuracy']):.3f})")

# %% [markdown]
# ### 2.6 Test-set evaluation and error analysis

# %%
proba_test = model.predict(ds_test, verbose=0)
pred_test = proba_test.argmax(1)
conf_test = proba_test.max(1)

cnn_acc = accuracy_score(y_test, pred_test)
cnn_f1 = f1_score(y_test, pred_test, average='macro')
cnn_top2 = float(np.mean([y in np.argsort(-p)[:2] for y, p in zip(y_test, proba_test)]))
print(f'TEST  accuracy {cnn_acc:.3f} | macro F1 {cnn_f1:.3f} | top-2 accuracy {cnn_top2:.3f} '
      f'| n = {len(y_test)}')
print(f'majority-class baseline {max(np.bincount(y_test)) / len(y_test):.3f}\n')
print(classification_report(y_test, pred_test, target_names=CLASS_NAMES, digits=3))

# %%
cm = confusion_matrix(y_test, pred_test)
cm_norm = cm / cm.sum(1, keepdims=True)
fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), gridspec_kw={'width_ratios': [1, 1]})
heat(axes[0], cm, [SHORT[c] for c in CLASS_NAMES], [SHORT[c] for c in CLASS_NAMES],
     title='Confusion matrix (counts)', fmt='{:.0f}')
axes[0].set_xlabel('predicted'); axes[0].set_ylabel('true')
heat(axes[1], cm_norm * 100, [SHORT[c] for c in CLASS_NAMES], [SHORT[c] for c in CLASS_NAMES],
     title='Row-normalised (% of each true stream)', fmt='{:.0f}', vmax=100)
axes[1].set_xlabel('predicted'); axes[1].set_ylabel('true')
plt.tight_layout(); plt.show()

off = [(CLASS_NAMES[i], CLASS_NAMES[j], cm[i, j], cm_norm[i, j])
       for i in range(NUM_CLASSES) for j in range(NUM_CLASSES) if i != j and cm[i, j] > 0]
off.sort(key=lambda r: -r[2])
print('Largest confusions (true -> predicted):')
for t, p, n, frac in off[:8]:
    print(f'  {t:<21} -> {p:<21} {n:>3} images ({frac:.0%} of {t})')

# %%
per_class = pd.DataFrame({
    'recall': cm.diagonal() / cm.sum(1),
    'precision': np.divide(cm.diagonal(), cm.sum(0), out=np.zeros(NUM_CLASSES), where=cm.sum(0) > 0),
    'test images': cm.sum(1),
    'train images': [int((img_train['label'] == c).sum()) for c in CLASS_NAMES],
    'edge density': [char.loc[c, 'edge_density'] for c in CLASS_NAMES],
}, index=CLASS_NAMES)

fig, axes = plt.subplots(1, 2, figsize=(14, 3.8))
s = per_class['recall'].sort_values()
barh(axes[0], [SHORT[c] for c in s.index], s.values, title='Per-stream recall on the test set',
     fmt='{:.2f}')
axes[0].axvline(cnn_acc, color=ORANGE, lw=1.3, ls=(0, (4, 3)))
axes[0].text(cnn_acc + 0.01, 8.4, 'overall', color=ORANGE, fontsize=9)
axes[1].scatter(per_class['train images'], per_class['recall'], s=46, color=BLUE, zorder=3)
for c, r in per_class.iterrows():
    axes[1].annotate(SHORT[c], (r['train images'], r['recall']), fontsize=8, color=INK_2,
                     xytext=(4, 4), textcoords='offset points')
axes[1].set_xlabel('training images'); axes[1].set_ylabel('test recall')
axes[1].set_title('Recall vs training volume')
plt.tight_layout(); plt.show()

print(f"correlation between training volume and recall: "
      f"{per_class['train images'].corr(per_class['recall']):.2f}   "
      f"(if this is not positive, imbalance is not what limits the weak streams)")
print(f"correlation between edge density and recall:    "
      f"{per_class['edge density'].corr(per_class['recall']):.2f}")
per_class.round(3)

# %%
# What the mistakes look like. Confident errors are the product risk: the assistant will state
# a wrong bin without hedging, which is worse than saying "I am not sure".
err_idx = np.where(pred_test != y_test)[0]
order = err_idx[np.argsort(-conf_test[err_idx])]
fig, axes = plt.subplots(2, 6, figsize=(15, 5.4))
for ax, k in zip(axes.ravel(), order[:12]):
    ax.imshow(Image.open(img_test.loc[k, 'path']).resize((180, 180)))
    ax.set_title(f'true {SHORT[CLASS_NAMES[y_test[k]]]}\npred {SHORT[CLASS_NAMES[pred_test[k]]]} '
                 f'({conf_test[k]:.0%})', fontsize=8.5)
    ax.axis('off')
fig.suptitle('The twelve most confident mistakes', x=0.006, ha='left', fontsize=12,
             fontweight='bold')
plt.tight_layout(); plt.show()

print(f'mean confidence when right: {conf_test[pred_test == y_test].mean():.3f}')
print(f'mean confidence when wrong: {conf_test[pred_test != y_test].mean():.3f}')

# %%
# Is the confidence score usable as a gate? If accuracy rises sharply with confidence, the
# assistant can abstain on low-confidence photos instead of guessing (used in Part 5).
bands = [(0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 1.01)]
rows = []
for lo, hi in bands:
    m = (conf_test >= lo) & (conf_test < hi)
    rows.append({'confidence': f'{lo:.1f}-{hi:.1f}' if hi <= 1 else f'>={lo:.1f}',
                 'images': int(m.sum()), 'share': m.mean(),
                 'accuracy': accuracy_score(y_test[m], pred_test[m]) if m.sum() else np.nan})
calib = pd.DataFrame(rows)
print(calib.round(3).to_string(index=False))

CNN_CONF_THRESHOLD = 0.60
gate = conf_test >= CNN_CONF_THRESHOLD
gated_acc = accuracy_score(y_test[gate], pred_test[gate]) if gate.any() else float('nan')
print(f'\nabstaining below {CNN_CONF_THRESHOLD:.2f}: answers {gate.mean():.0%} of photos '
      f'at {gated_acc:.3f} accuracy, defers {1 - gate.mean():.0%} to the text route')

# %%
cnn_summary = pd.DataFrame([
    {'stage': 'majority class', 'val accuracy': np.nan,
     'test accuracy': max(np.bincount(y_test)) / len(y_test), 'test macro F1': np.nan},
    {'stage': f'{BACKBONE_NAME} frozen features + logistic regression',
     'val accuracy': probe.loc[BACKBONE_NAME, 'val accuracy'],
     'test accuracy': np.nan, 'test macro F1': np.nan},
    {'stage': 'stage 1: frozen backbone + tuned head',
     'val accuracy': max(hist1.history['val_accuracy']), 'test accuracy': np.nan,
     'test macro F1': np.nan},
    {'stage': f'stage 2: fine-tuned top {UNFREEZE_LAST} layers',
     'val accuracy': max(hist2.history['val_accuracy']), 'test accuracy': cnn_acc,
     'test macro F1': cnn_f1},
]).set_index('stage')
cnn_summary.round(3)

# %% [markdown]
# One row of that table deserves an explanation, because it looks like a regression: the
# frozen-feature probe scores **higher** on validation than stage 1 does. They are not
# measuring the same thing. The probe fits a convex solver to run convergence on clean,
# un-augmented embeddings — the easiest version of the problem, and an optimistic number. Stage
# 1 trains the same head by SGD for five epochs on *augmented* inputs, which is harder and is
# the point: those five epochs are what make the head robust enough for the backbone to be
# unfrozen underneath it without the gradients tearing the ImageNet features apart. Stage 2
# then passes both.

# %% [markdown]
# ### Part 2 takeaways
#
# Three things in the evaluation above are worth more than the headline accuracy.
#
# **The errors are structured, not random.** They concentrate among the *container* streams —
# plastic, metal and glass — which is exactly right: a crushed drink container photographed on
# a tipping floor shows its shape, not its material, and the label depends on the material.
# Paper against cardboard is the same story in fibre. Miscellaneous Trash is the hardest stream
# by construction rather than by appearance: it is defined by *not* being one of the other
# eight, so there is no visual property for a convolutional network to converge on.
#
# **Training volume does not predict recall** — the correlation printed above is negative. The
# two largest streams, Plastic and Metal, are among the weakest, and the smallest (Textile
# Trash) is not. That rules out imbalance as the binding constraint and points at
# confusability instead, which is why the response is class weighting plus macro F1 rather than
# a larger resampling effort.
#
# **Confidence is informative**, and the calibration table is the evidence: accuracy climbs
# steeply with the softmax maximum. That is what makes the gate in Part 5 defensible rather
# than decorative — the assistant can answer the large, reliable majority of photographs and
# ask for a written description on the rest, instead of guessing uniformly.
#
# For the product, *which* errors happen matters as much as how many. A paper/cardboard mix-up
# is nearly harmless because both go to the fibre stream; predicting Plastic for miscellaneous
# trash puts a non-recyclable item into a recycling load and contaminates it. An
# error-cost-weighted loss would be the principled next step, and it needs the facility's
# contamination costs, which EcoSort does not have yet.
