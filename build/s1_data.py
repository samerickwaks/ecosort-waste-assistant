# %% [markdown]
# ## Part 1: Dataset exploration and preparation
#
# Three sources feed the assistant, and each one is explored before it is used:
#
# | File | Content | Used by |
# |---|---|---|
# | `RealWaste/` | 4,752 photographs of real waste, foldered into 9 streams | Part 2 (CNN) |
# | `waste_descriptions.csv` | 5,000 resident-style descriptions with a stream label | Part 3 (text classifier) |
# | `waste_policy_documents.json` | 14 Metro City policy documents | Part 4 (RAG) |
#
# The job of this part is not to produce pretty tables; it is to find the things that will
# break the models later. Four of them turn up, and each one changes a decision downstream.

# %%
# -------------------------
# Paths and global config
# -------------------------
DATA_DIR = pathlib.Path('RealWaste')
DESCRIPTIONS_CSV = pathlib.Path('waste_descriptions.csv')
POLICY_JSON = pathlib.Path('waste_policy_documents.json')

IMG_HEIGHT = IMG_WIDTH = 224          # MobileNetV2's native ImageNet resolution
BATCH_SIZE = 32
SPLIT = (0.70, 0.15, 0.15)            # train / validation / test, used for every modality

CLASS_NAMES = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())
NUM_CLASSES = len(CLASS_NAMES)
SHORT = {c: (c.replace('Miscellaneous', 'Misc.').replace(' Organics', ' Org.')
             .replace(' Trash', ' Tr.')) for c in CLASS_NAMES}   # compact axis labels

print(f'{NUM_CLASSES} waste streams: {CLASS_NAMES}')
for p in (DATA_DIR, DESCRIPTIONS_CSV, POLICY_JSON):
    print(f'  {"found   " if p.exists() else "MISSING "} {p}')

# %% [markdown]
# ### 1.1 The RealWaste image set
#
# RealWaste is a *field* dataset: the photographs were taken of material arriving at the
# Whyte's Gully recovery facility in Wollongong, NSW. That provenance matters — the images are
# of crushed, soiled, partly-shredded items on a tipping floor, not clean product shots, which
# is both why the dataset is useful for EcoSort and why accuracy here will be well below what
# a catalogue-image benchmark would suggest.

# %%
# Inventory: one row per image file, label taken from the parent folder.
images = pd.DataFrame(
    [(f, f.parent.name, f.name) for f in sorted(DATA_DIR.glob('*/*.jpg'))],
    columns=['path', 'label', 'filename'])
images['label_idx'] = images['label'].map({c: i for i, c in enumerate(CLASS_NAMES)})

counts = images['label'].value_counts()
imbalance = counts.max() / counts.min()
print(f'{len(images):,} images across {NUM_CLASSES} streams')
print(f'largest stream {counts.idxmax()} ({counts.max()}), smallest {counts.idxmin()} ({counts.min()}) '
      f'-> imbalance ratio {imbalance:.1f}:1')
print(f'a majority-class-only baseline would score {counts.max() / len(images):.1%} accuracy')
counts.to_frame('images').assign(share=lambda d: (d['images'] / len(images)).map('{:.1%}'.format))

# %%
fig, ax = plt.subplots(figsize=(7.2, 3.6))
barh(ax, [SHORT[c] for c in counts.index], counts.values,
     title='RealWaste is imbalanced: Plastic has 2.9x the images of Textile Trash',
     xlabel='images')
ax.axvline(len(images) / NUM_CLASSES, color=ORANGE, lw=1.4, ls=(0, (4, 3)))
ax.text(len(images) / NUM_CLASSES + 8, NUM_CLASSES - 0.6, 'even split (528)',
        color=ORANGE, fontsize=9, va='center')
plt.tight_layout(); plt.show()

# %% [markdown]
# **Why this matters.** A 2.9:1 imbalance is mild enough that oversampling would mostly add
# redundant gradient steps, but large enough that an unweighted model will quietly trade
# Textile Trash recall for Plastic recall. Part 2 therefore uses **class weights** (inverse
# frequency) rather than resampling, and reports **macro F1** alongside accuracy so the small
# streams cannot be hidden by the large ones.

# %%
# Single pass over every file: header info plus a greyscale thumbnail reused for three
# diagnostics (exposure, edge density, near-duplicate detection). `draft()` lets libjpeg
# decode straight to a reduced size, which keeps the pass to well under a minute.
t0 = time.time()
rows, thumbs = [], []
for f in images['path']:
    im = Image.open(f)
    w, h, mode = im.size[0], im.size[1], im.mode
    im.draft('L', (130, 130))
    g128 = np.asarray(im.convert('L').resize((128, 128)), dtype=np.float32)
    edge = float(np.abs(np.diff(g128, axis=0)).mean() + np.abs(np.diff(g128, axis=1)).mean())
    rows.append((w, h, mode, float(g128.mean()), float(g128.std()), edge))
    thumbs.append(np.asarray(Image.fromarray(g128.astype('uint8')).resize((32, 32)),
                             dtype=np.float32).ravel())

images[['width', 'height', 'mode', 'brightness', 'contrast', 'edge_density']] = pd.DataFrame(rows, index=images.index)
THUMBS = np.stack(thumbs)
print(f'scanned {len(images):,} images in {time.time() - t0:.0f}s')
print(f"resolutions: {images.groupby(['width', 'height']).size().to_dict()}")
print(f"colour modes: {images['mode'].value_counts().to_dict()}")
print(f"file size: {images['path'].map(lambda p: p.stat().st_size).mean() / 1024:.0f} kB mean")

# %% [markdown]
# Every image is **524x524 RGB** — no aspect-ratio or colour-mode handling is needed, and the
# resize to 224x224 is a clean 2.3x downscale with no distortion. That is unusually tidy, and
# it is also a *limitation*: a resident's phone photo will not be square, will not be framed
# the same way, and will not be lit by a facility floodlight. Section 5.4 returns to this.

# %%
# What the model will actually see: two examples per stream, after the 224x224 resize.
fig, axes = plt.subplots(2, NUM_CLASSES, figsize=(15, 3.9))
rng = np.random.default_rng(SEED)
for j, cls in enumerate(CLASS_NAMES):
    picks = rng.choice(images.index[images['label'] == cls], 2, replace=False)
    for i, idx in enumerate(picks):
        axes[i, j].imshow(Image.open(images.at[idx, 'path']).resize((IMG_WIDTH, IMG_HEIGHT)))
        axes[i, j].axis('off')
    axes[0, j].set_title(SHORT[cls], fontsize=9, loc='center', pad=4)
fig.suptitle('RealWaste at model resolution: damaged, soiled, overlapping, variable background',
             x=0.009, ha='left', fontsize=12, fontweight='bold')
plt.tight_layout(); plt.show()

# %%
# Exposure, contrast and edge density per stream — a quick check for shortcut features
# (if one class were systematically darker or busier, the CNN could learn that instead
# of the material).
char = images.groupby('label')[['brightness', 'contrast', 'edge_density']].mean()
fig, axes = plt.subplots(1, 3, figsize=(14, 3.4))
for ax, (col, title, color) in zip(axes, [
        ('brightness', 'Mean brightness (0-255)', BLUE),
        ('contrast', 'Contrast (pixel std.)', AQUA),
        ('edge_density', 'Edge density (mean |gradient|)', ORANGE)]):
    s = char[col].sort_values()
    barh(ax, [SHORT[c] for c in s.index], s.values, color=color, title=title, fmt='{:.1f}')
fig.suptitle('Low-level image statistics are nearly flat across streams — except texture',
             x=0.006, ha='left', fontsize=12, fontweight='bold')
plt.tight_layout(); plt.show()

print(f"brightness spread across streams: {char['brightness'].max() - char['brightness'].min():.1f} grey levels")
print(f"edge density: Vegetation {char.loc['Vegetation', 'edge_density']:.1f} vs "
      f"Glass {char.loc['Glass', 'edge_density']:.1f} "
      f"({char.loc['Vegetation', 'edge_density'] / char.loc['Glass', 'edge_density']:.1f}x)")

# %% [markdown]
# Brightness and contrast are essentially constant across streams (a 9-grey-level spread on a
# 0-255 scale), so there is no exposure shortcut for the model to latch onto. **Edge density is
# not flat**: Vegetation and Food Organics are more than twice as textured as Glass and
# Cardboard. That is a genuine signal (leaves and food scraps really are high-frequency), and it
# predicts which streams will be easy — the confusion matrix in §2.6 confirms it.

# %%
# Near-duplicate audit. A facility photographs the same item more than once; if two frames of
# one bottle land on opposite sides of the split, the test score is inflated. Pairwise mean
# absolute difference on the 32x32 thumbnails, within class only.
t0 = time.time()
dup_pairs = []
for cls in CLASS_NAMES:
    idx = images.index[images['label'] == cls].to_numpy()
    X = THUMBS[idx]
    for s in range(0, len(idx), 250):                        # chunked to bound memory
        D = np.abs(X[s:s + 250, None, :] - X[None, :, :]).mean(-1)
        for a in range(D.shape[0]):
            for b in np.where(D[a] < 6.0)[0]:
                if idx[s + a] < idx[b]:
                    dup_pairs.append((idx[s + a], idx[b]))

# Union-find over the near-duplicate pairs -> clusters of frames showing the same item
parent = {i: i for i in images.index}
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i
for a, b in dup_pairs:
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb
images['item_group'] = [find(i) for i in images.index]

n_clustered = int((images.groupby('item_group')['item_group'].transform('size') > 1).sum())
print(f'{len(dup_pairs)} near-duplicate pairs -> {n_clustered} images ({n_clustered / len(images):.1%}) '
      f'belong to a multi-frame group  [{time.time() - t0:.0f}s]')
print(images[images.groupby('item_group')['item_group'].transform('size') > 1]
      .groupby('label').size().to_frame('near-duplicate images'))

# %%
# The worst offender, shown: Glass bottles re-photographed on the same patch of floor.
biggest = images.groupby('item_group').size().sort_values(ascending=False)
grp = images[images['item_group'] == biggest.index[0]]
fig, axes = plt.subplots(1, min(5, len(grp)), figsize=(2.3 * min(5, len(grp)), 2.6))
for ax, (_, r) in zip(np.atleast_1d(axes), grp.iterrows()):
    ax.imshow(Image.open(r['path']).resize((160, 160)))
    ax.set_title(r['filename'], fontsize=8)
    ax.axis('off')
fig.suptitle(f'One item, {len(grp)} frames — these must not straddle the train/test boundary',
             x=0.006, ha='left', fontsize=11, fontweight='bold')
plt.tight_layout(); plt.show()

# %% [markdown]
# **Finding 1 (images).** 1.5% of the set is near-duplicate frames, and they are concentrated in
# Glass (12% of that stream). Splitting at random would leak those items across the boundary.
# The split built in §1.4 is therefore **grouped** — every frame of an item goes to the same
# side — as well as stratified.

# %% [markdown]
# ### 1.2 The resident descriptions
#
# `waste_descriptions.csv` simulates what a resident types into the app. Five columns arrive
# with it, and the first job is to work out which of them a model is actually allowed to see.

# %%
desc = pd.read_csv(DESCRIPTIONS_CSV)
print(f'{len(desc):,} rows x {desc.shape[1]} columns: {list(desc.columns)}')
print(f'missing values:\n{desc.isna().sum().to_string()}')
desc.head(6)

# %%
# Leakage audit: how many distinct values does each auxiliary column take, and does any single
# value ever span more than one category?
audit = []
for col in ['description', 'disposal_instruction', 'common_confusion', 'material_composition']:
    s = desc[[col, 'category']].dropna()
    spans = s.groupby(col)['category'].nunique()
    audit.append({'column': col,
                  'distinct values': s[col].nunique(),
                  'values spanning >1 category': int((spans > 1).sum()),
                  'category determined by this column': f'{(spans == 1).mean():.0%} of values'})
audit = pd.DataFrame(audit)
print(audit.to_string(index=False))
print()
print('material_composition values (one per stream - this column *is* the label):')
print(desc.groupby('category')['material_composition'].first().to_string())

# %% [markdown]
# **Finding 2 (text): three of the five columns are the label in disguise.**
# `material_composition` takes exactly 9 values, one per stream — it is a renamed target.
# `common_confusion` likewise takes 9 values, and 43 of the 44 `disposal_instruction` strings
# map to a single stream. A classifier trained on any of them would score ~100% and learn
# nothing about language.
#
# **Decision:** the text classifier in Part 3 is trained on the `description` column *only*.
# The other columns are not wasted — `disposal_instruction` becomes reference material for the
# RAG corpus in §1.4, where quoting the label-correlated text is the whole point.

# %%
desc['n_words'] = desc['description'].str.split().str.len()
desc['n_chars'] = desc['description'].str.len()
vocab = Counter(w for d in desc['description'] for w in re.findall(r"[a-z']+", d.lower()))

cat_counts = desc['category'].value_counts()
print(f"descriptions: {desc['n_words'].min()}-{desc['n_words'].max()} words "
      f"(median {desc['n_words'].median():.0f}), {desc['n_chars'].mean():.0f} characters on average")
print(f'vocabulary: {len(vocab)} distinct lowercase word types in {sum(vocab.values()):,} tokens')
print(f'duplicate description strings: {desc["description"].duplicated().sum()} '
      f'({desc["description"].nunique():,} unique)')
print(f'conflicting labels on an identical string: '
      f'{int((desc.groupby("description")["category"].nunique() > 1).sum())}')
print(f'class balance: {cat_counts.max()}-{cat_counts.min()} per stream '
      f'(imbalance {cat_counts.max() / cat_counts.min():.2f}:1 — effectively balanced)')

fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
axes[0].hist(desc['n_words'], bins=range(1, 12), color=BLUE, rwidth=0.86, align='left')
axes[0].set_title('Descriptions are short: 5 words at the median')
axes[0].set_xlabel('words'); axes[0].set_ylabel('descriptions')
axes[0].set_xticks(range(1, 11))
top = pd.Series(dict(vocab.most_common(14))).sort_values()
barh(axes[1], top.index, top.values, color=AQUA, title='Most frequent tokens', xlabel='occurrences')
plt.tight_layout(); plt.show()

# %% [markdown]
# A 309-word vocabulary over 5,000 documents is tiny, and the shape of the data explains why:
# the descriptions are **templated** — a condition adjective, an optional size/brand modifier, a
# head noun, and an optional state clause (`"wrinkled pink takeout container with product
# remaining"`). That has two consequences. It makes the task easy enough that a linear model on
# character and word n-grams is a serious contender, and it means the only hard cases are the
# ones where the *head noun itself* is ambiguous.

# %%
# Which tokens are shared across streams, and how badly?
tok2cat = defaultdict(Counter)
for d, c in zip(desc['description'], desc['category']):
    for w in set(re.findall(r"[a-z']+", d.lower())):
        tok2cat[w][c] += 1

# A token is "informative but ambiguous" when a couple of streams account for nearly all of
# its uses, yet no single stream dominates it: 'bottle' is one of three streams, which is
# useful and insufficient at the same time. Generic modifiers ('with', 'large') spread across
# all nine and are filtered out by the top-3 coverage test.
rows_amb = []
for w, cc in tok2cat.items():
    n = sum(cc.values())
    if n < 60:
        continue
    shares = np.array([v for _, v in cc.most_common()]) / n
    if shares[0] < 0.90 and shares[:3].sum() > 0.80:
        rows_amb.append((w, n, shares[0], len(cc), dict(cc.most_common(3))))
rows_amb.sort(key=lambda r: -r[1])
print("Informative-but-ambiguous tokens (>=60 uses; top 3 streams cover >80%, none over 90%):")
for w, n, top1, k, top3 in rows_amb[:10]:
    print(f"  {w:<12} {n:4d} uses  majority stream {top1:4.0%}  spans {k} streams   {top3}")

exclusive = {w for w, cc in tok2cat.items() if len(cc) == 1}
desc['has_exclusive_token'] = desc['description'].map(
    lambda d: any(w in exclusive for w in re.findall(r"[a-z']+", d.lower())))
hard = desc[~desc['has_exclusive_token']]
print(f'\n{len(hard)} descriptions ({len(hard) / len(desc):.0%}) contain no stream-exclusive word at all;'
      f' these are the rows Part 3 has to reason about rather than look up.')
print(hard['category'].value_counts().to_frame('ambiguous descriptions').T.to_string())

# %%
fig, ax = plt.subplots(figsize=(8.2, 3.5))
show_tokens = ['bottle', 'paper', 'container', 'bag']
order = [c for c in CLASS_NAMES]
w = 0.8 / len(show_tokens)
for k, tok in enumerate(show_tokens):
    vals = [tok2cat[tok].get(c, 0) for c in order]
    ax.bar(np.arange(len(order)) + (k - (len(show_tokens) - 1) / 2) * w, vals, w,
           label=f'"{tok}"')
ax.set_xticks(range(len(order)))
ax.set_xticklabels([SHORT[c] for c in order], rotation=30, ha='right')
ax.set_ylabel('descriptions')
ax.set_title('Head nouns do not determine the stream: "bottle" is glass, plastic *and* metal')
ax.legend(ncol=4)
plt.tight_layout(); plt.show()

# %% [markdown]
# **Finding 3 (text).** `"bottle"` appears in Glass, Plastic *and* Metal descriptions;
# `"paper"` appears in Paper but also in Cardboard and Textile Trash (`"paper towel"`);
# `"cup"` is split almost evenly three ways. 17% of rows contain no stream-exclusive word at
# all, so their label depends on how words combine rather than on which words are present.
#
# The obvious hypothesis is that this 17% is where a bag-of-words model and a contextual
# transformer part company. Part 3 therefore trains both and scores them **on that subset
# specifically** rather than only on the headline accuracy — and §3.4 reports that the
# hypothesis is wrong, which is more interesting than if it had held.

# %% [markdown]
# ### 1.3 The Metro City policy documents

# %%
with open(POLICY_JSON, encoding='utf-8') as fh:
    policies = json.load(fh)

pol = pd.DataFrame(policies)
pol['n_chars'] = pol['document_text'].str.len()
pol['n_categories'] = pol['categories_covered'].map(len)
print(f'{len(pol)} policy documents, {pol["n_chars"].sum():,} characters total '
      f'({pol["n_chars"].min()}-{pol["n_chars"].max()} each)')
print(f'jurisdictions: {pol["jurisdiction"].unique().tolist()} | '
      f'effective dates {pol["effective_date"].min()} to {pol["effective_date"].max()}')
pol[['policy_id', 'policy_type', 'n_categories', 'effective_date', 'n_chars']]

# %%
print(pol.loc[1, 'policy_type'], '\n' + '-' * 60)
print(pol.loc[1, 'document_text'])

# %%
# Every document follows the same heading structure, which is what makes section-level
# chunking possible in §1.4. Two heading styles occur: topic headings in the nine dedicated
# documents ("Acceptable Items:", "Preparation Instructions:") and per-stream headings inside
# the five cross-cutting ones ("GLASS GUIDELINES:").
SECTION_RE = re.compile(r'^([A-Z][A-Za-z /-]+):\s*$', re.M)   # the hyphen matters:
#                                                               "Non-Acceptable Items:" is a
#                                                               heading, not part of the list
#                                                               above it
sections = Counter(m.group(1) for d in pol['document_text'] for m in SECTION_RE.finditer(d))
print('Section headings, and how many of the 14 documents use each:')
for s, n in sections.most_common():
    print(f'  {s:<32} {n:>2}/14')

cover = np.zeros((len(pol), NUM_CLASSES))
for i, cats in enumerate(pol['categories_covered']):
    for c in cats:
        cover[i, CLASS_NAMES.index(c)] = 1
per_cat = cover.sum(0).astype(int)

fig, ax = plt.subplots(figsize=(9.5, 5))
heat(ax, cover, [SHORT[c] for c in CLASS_NAMES],
     [f'{r.policy_id}. {r.policy_type}' for r in pol.itertuples()],
     title='Policy coverage: one dedicated document per stream, plus five cross-cutting ones',
     fmt='{:.0f}', vmax=1.4)
plt.tight_layout(); plt.show()
print('documents covering each stream:',
      ', '.join(f'{SHORT[c]} {n}' for c, n in zip(CLASS_NAMES, per_cat)))
print(f'thinnest coverage: {CLASS_NAMES[int(per_cat.argmin())]} ({per_cat.min()} document) - '
      f'retrieval for that stream has only one place to go.')

# %% [markdown]
# Documents 1-9 are one-per-stream guidelines; documents 10-14 are cross-cutting (municipal,
# residential, commercial, multi-unit, community). Every stream is covered by its dedicated
# document plus between 0 and 3 general ones, so **retrieval has a ground truth**: for the query
# *"How do I recycle Glass?"*, the relevant set is exactly the documents whose
# `categories_covered` contains Glass. §4.2 uses that to score the retriever with recall@k and
# MRR instead of eyeballing it.

# %% [markdown]
# ### 1.4 Data pipelines and splits
#
# #### 1.4a The provided Keras pipeline, and why the modelling does not use it

# %%
# ---- Provided starter cell, run as given ----
import pathlib
data_dir = pathlib.Path('RealWaste')

num_classes = len([item for item in data_dir.glob('*') if item.is_dir()])
print(f"Number of classes: {num_classes}")
class_names = sorted([item.name for item in data_dir.glob('*') if item.is_dir()])
print(f"Class names: {class_names}")
image_count = len(list(data_dir.glob('*/*.jpg'))) + len(list(data_dir.glob('*/*.png')))
print(f"Total images found: {image_count}")

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="training", seed=42,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE,
    label_mode='categorical', shuffle=True)
validation_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="validation", seed=42,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE,
    label_mode='categorical', shuffle=True)

val_batches = tf.data.experimental.cardinality(validation_ds)
test_dataset = validation_ds.take(val_batches // 2)
validation_ds = validation_ds.skip(val_batches // 2)
print(f"Number of training batches: {tf.data.experimental.cardinality(train_ds)}")
print(f"Number of validation batches: {tf.data.experimental.cardinality(validation_ds)}")
print(f"Number of test batches: {tf.data.experimental.cardinality(test_dataset)}")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
# ---- end provided cell ----
# (The three `.cache()` calls in the original are dropped: caching 3,800 decoded 224x224x3
#  float32 images needs ~2.3 GB of RAM, which this machine does not have spare.)

# %% [markdown]
# The provided pipeline is a correct 80/20 split and it runs, but three properties make it
# unsuitable for the evaluation this lab is graded on:
#
# 1. **The test set is carved out of the validation set.** `validation_ds.take(n//2)` is taken
#    *after* the generator has been consumed to build it, and both halves come from the same
#    20% partition — so the "test" set is a sibling of the data used for early stopping, not an
#    untouched holdout.
# 2. **No grouping.** Random assignment puts near-duplicate frames of the same bottle
#    (Finding 1) on both sides of the boundary.
# 3. **Batch-level splitting.** Slicing by batch cannot guarantee stratification; with 9
#    uneven classes the small streams drift.
#
# The modelling below therefore uses the file-level split built next. The provided datasets stay
# defined so the cell above remains runnable and comparable.

# %% [markdown]
# #### 1.4b A grouped, stratified 70/15/15 image split

# %%
def grouped_stratified_split(df, group_col, label_col, fractions=SPLIT, seed=SEED):
    """Assign whole groups to train/val/test while holding each label's proportions.

    Groups (here: all frames of one physical item) are never broken apart. Within a label,
    groups are shuffled and then greedily placed into whichever split is furthest below its
    target count, which keeps the stratification tight even though group sizes vary.
    """
    rng = np.random.default_rng(seed)
    assignment = pd.Series(index=df.index, dtype=object)
    for label, part in df.groupby(label_col):
        groups = list(part.groupby(group_col).groups.values())
        rng.shuffle(groups)
        targets = np.array(fractions) * len(part)
        filled = np.zeros(3)
        for g in groups:
            j = int(np.argmax(targets - filled))     # most under-filled split wins
            assignment[list(g)] = ['train', 'val', 'test'][j]
            filled[j] += len(g)
    return assignment


images['split'] = grouped_stratified_split(images, 'item_group', 'label')

split_tab = (pd.crosstab(images['label'], images['split'])[['train', 'val', 'test']])
split_tab.loc['TOTAL'] = split_tab.sum()
split_share = split_tab.div(split_tab.sum(axis=1), axis=0)
print(split_tab.to_string())
print('\nper-stream share of each split (target 70 / 15 / 15):')
print(split_share.applymap('{:.1%}'.format).to_string())

leak = images.groupby('item_group')['split'].nunique().gt(1).sum()
print(f'\nitem groups split across more than one partition: {leak}  (must be 0)')

# %%
# tf.data pipelines built from explicit file lists, so the split above is exactly what trains.
def make_image_ds(frame, shuffle=False, batch_size=BATCH_SIZE):
    """Decode -> resize -> one-hot. No scaling here: the backbone's own preprocess_input
    layer handles that inside the model, so the saved model is self-contained."""
    paths = tf.constant([str(p) for p in frame['path']])
    labels = tf.keras.utils.to_categorical(frame['label_idx'], NUM_CLASSES)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(frame), seed=SEED, reshuffle_each_iteration=True)

    def load(path, label):
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method='bilinear')
        return img, label

    return (ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(batch_size)
              .prefetch(tf.data.AUTOTUNE))


img_train = images[images['split'] == 'train'].reset_index(drop=True)
img_val = images[images['split'] == 'val'].reset_index(drop=True)
img_test = images[images['split'] == 'test'].reset_index(drop=True)

ds_train = make_image_ds(img_train, shuffle=True)
ds_val = make_image_ds(img_val)
ds_test = make_image_ds(img_test)

xb, yb = next(iter(ds_train))
print(f'train {len(img_train)} | val {len(img_val)} | test {len(img_test)}')
print(f'batch: images {xb.shape} {xb.dtype} range [{float(tf.reduce_min(xb)):.0f}, '
      f'{float(tf.reduce_max(xb)):.0f}] | labels {yb.shape} one-hot')

# %% [markdown]
# #### 1.4c Text cleaning and split
#
# The cleaning is deliberately light. Descriptions are short, templated and already lowercase-ish;
# aggressive stemming or stop-word removal would destroy the few function words that carry
# meaning (`"with food residue"`, `"partially filled"`). What is worth doing is normalising case
# and punctuation, collapsing the brand placeholders (`Brand A`, `Premium`, `Supermarket`) that
# are pure noise, and **de-duplicating** so the 61 repeated strings cannot straddle the split.

# %%
BRAND_RE = re.compile(r'\b(brand [a-z]|premium|budget|designer|eco-friendly|supermarket|'
                      r'generic|store|value|luxury)\b', re.I)


def clean_description(text):
    """Normalise a resident description without destroying its content words."""
    t = str(text).lower()
    t = BRAND_RE.sub(' ', t)                     # brand placeholders carry no stream signal
    t = re.sub(r'[^a-z\s-]', ' ', t)             # keep hyphens: "plastic-metal composite"
    t = re.sub(r'\s+', ' ', t).strip()
    return t


desc['clean'] = desc['description'].map(clean_description)
print('cleaning examples (rows where normalisation actually changes something):')
changed = desc[desc['description'].str.lower() != desc['clean']]
for s in changed['description'].sample(5, random_state=3):
    print(f'  {s!r:<58} ->  {clean_description(s)!r}')
print(f'{len(changed) / len(desc):.0%} of descriptions carry a brand placeholder or punctuation '
      f'that cleaning removes')

text_df = (desc.drop_duplicates(subset='clean')
               .loc[:, ['clean', 'description', 'category', 'has_exclusive_token']]
               .reset_index(drop=True))
print(f'\n{len(desc):,} rows -> {len(text_df):,} after de-duplicating cleaned strings')

txt_train, txt_tmp = train_test_split(text_df, test_size=SPLIT[1] + SPLIT[2],
                                      stratify=text_df['category'], random_state=SEED)
rel = SPLIT[2] / (SPLIT[1] + SPLIT[2])
txt_val, txt_test = train_test_split(txt_tmp, test_size=rel,
                                     stratify=txt_tmp['category'], random_state=SEED)
for name, part in [('train', txt_train), ('val', txt_val), ('test', txt_test)]:
    print(f'  {name:<5} {len(part):>5} rows | '
          f'class share {part["category"].value_counts(normalize=True).min():.3f}-'
          f'{part["category"].value_counts(normalize=True).max():.3f}')
print(f'overlap between train and test descriptions: '
      f'{len(set(txt_train["clean"]) & set(txt_test["clean"]))}')

# %% [markdown]
# #### 1.4d The retrieval corpus
#
# Two kinds of document go into the index. The 14 policies are split **by section** — a
# resident asking how to prepare an item wants the *Preparation Instructions* block, not the
# whole 700-character document, and section-level chunks keep the retrieved context short
# enough for FLAN-T5's prompt. Alongside them, each stream contributes one **disposal card**
# built from the `disposal_instruction` and `common_confusion` columns of the description file,
# which is where the advice on confusable items actually lives.

# %%
def split_sections(text):
    """Split a policy document into (heading, body) sections on its 'Heading:' lines.

    The bare title line is not emitted as a chunk of its own - three words retrieve nothing
    useful - but it is prepended to every section so each chunk names its own source.
    """
    marks = [(m.start(), m.group(1)) for m in SECTION_RE.finditer(text)]
    if not marks:
        return [('Document', text.strip())]
    out = []
    for i, (pos, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        body = text[text.index('\n', pos) + 1:end].strip()
        if body:
            out.append((name, body))
    return out


def section_categories(heading, doc_categories):
    """A 'GLASS GUIDELINES:' section inside a cross-cutting document is about Glass only."""
    for c in CLASS_NAMES:
        if heading.upper().startswith(c.upper()):
            return [c]
    return list(doc_categories)


# Each chunk is stored twice over, for two different consumers:
#   `text` - document title + section heading + content. This is what gets embedded, because
#            the heading words ("Preparation Instructions", "Non-Acceptable Items") are
#            exactly what makes a chunk findable by a question about preparation or exclusions.
#   `body` - content only. This is what goes into a generation prompt, because a model shown
#            the heading tends to answer with the heading.
def flatten(text):
    return re.sub(r'\s+', ' ', text.replace('\n- ', '; ').replace('\n', ' ')).strip().lstrip('- ')


chunks = []
for p in policies:
    title = p['document_text'].split('\n')[0].strip()
    for name, body in split_sections(p['document_text']):
        chunks.append({'chunk_id': f"P{p['policy_id']}-{name.replace(' ', '')[:18]}",
                       'source': f"Policy {p['policy_id']}: {p['policy_type']}",
                       'section': name,
                       'categories': section_categories(name, p['categories_covered']),
                       'effective_date': p['effective_date'],
                       'text': f"{title}. {name}: {body}",
                       'body': flatten(body)})

for cat, grp in desc.groupby('category'):
    tips = sorted(grp['disposal_instruction'].dropna().unique())
    conf = sorted(grp['common_confusion'].dropna().unique())
    card = ' '.join(tips) + (' Common confusion: ' + ' '.join(conf) if conf else '')
    chunks.append({'chunk_id': f'D-{cat.replace(" ", "")}', 'source': f'{cat} disposal card',
                   'section': 'Disposal Instructions', 'categories': [cat],
                   'effective_date': None,
                   'text': f'{cat} disposal instructions: {card}',
                   'body': flatten(card)})

corpus = pd.DataFrame(chunks)
corpus['n_words'] = corpus['text'].str.split().str.len()
chunks_per_cat = pd.Series({c: int(corpus['categories'].map(lambda cs: c in cs).sum())
                            for c in CLASS_NAMES})
print(f'{len(corpus)} retrievable chunks from {len(pol)} policies + {NUM_CLASSES} disposal cards')
print(f'chunk length: {corpus["n_words"].min()}-{corpus["n_words"].max()} words '
      f'(median {corpus["n_words"].median():.0f}) - short enough that the top 3 fit in a '
      f'FLAN-T5 prompt together')
print(f'\nrelevant chunks per stream (the retrieval ground truth used in 4.2):')
print(chunks_per_cat.to_string())
corpus.head(4)[['chunk_id', 'source', 'section', 'categories', 'n_words']]

# %% [markdown]
# ### Part 1 takeaways — what the rest of the notebook has to handle
#
# | # | Finding | Where it bites | Response |
# |---|---|---|---|
# | 1 | 2.9:1 class imbalance in the images | Textile Trash recall | class weights + macro F1 (§2.3, §2.6) |
# | 2 | Near-duplicate frames, 12% of Glass | inflated test accuracy | grouped split (§1.4b) |
# | 3 | Three columns of `waste_descriptions.csv` are the label | a ~100% text model that learned nothing | train on `description` only (§1.2) |
# | 4 | 17% of descriptions have no stream-exclusive word | expected to be the hard cases | evaluated separately in §3.4 (and turns out not to be) |
#
# Two limitations are worth stating now because no modelling choice can fix them. The images are
# all 524x524 facility photographs under one lighting setup, so measured accuracy is an
# **upper bound** on performance against resident phone photos. And the descriptions are
# synthetic and templated — a 309-word vocabulary is perhaps a tenth of what real free text
# would bring, so the text classifier's headline score should be read as a ceiling too.
