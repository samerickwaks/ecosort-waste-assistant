# %% [markdown]
# ## Part 5: The integrated waste management assistant
#
# Three models become one product here. The integration is not just plumbing — it is where the
# weaknesses measured in Parts 2–4 get handled:
#
# | Measured weakness | Where it was found | How the assistant handles it |
# |---|---|---|
# | Photos of Misc. Trash are unreliable | §2.6 confusion matrix | confidence gate; ask for a description instead of guessing |
# | Confidence correlates with correctness | §2.6 calibration table | the gate threshold is set from that table, not invented |
# | The text model saturates its benchmark, so its confidence is untested on real wording | §3.4 | top-3 alternatives are returned, not just the argmax |
# | The generator can drift off policy | §4.3 | grounding check with an extractive fallback |
# | Neither classifier can say "not waste" | §3.5 | low-confidence status, surfaced to the user |

# %% [markdown]
# ### 5.1 Architecture
#
# One entry point, two input routes, one answer format. Routing is by input type; both routes
# produce the same `(category, confidence, alternatives)` triple, which is the contract that
# lets the downstream RAG stage stay ignorant of where the classification came from. When both
# a photo and a description arrive, their probability vectors are fused rather than one being
# discarded.

# %%
fig, ax = plt.subplots(figsize=(13, 4.6))
ax.set_xlim(0, 13); ax.set_ylim(0, 4.6); ax.axis('off')


def box(x, y, w, h, title, sub='', fc='#eef4fd', ec=BLUE):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.3,
                               joinstyle='round', zorder=2))
    ax.text(x + w / 2, y + h - 0.33, title, ha='center', va='top', fontsize=9.5,
            fontweight='bold', color=INK, zorder=3)
    if sub:
        ax.text(x + w / 2, y + h - 0.72, sub, ha='center', va='top', fontsize=8,
                color=INK_2, zorder=3, linespacing=1.45)


def arrow(x1, y1, x2, y2, label='', color=AXIS):
    ax.annotate('', (x2, y2), (x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=1.2, shrinkA=2, shrinkB=2))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, ha='center', fontsize=7.6, color=INK_2)


box(0.1, 2.55, 2.0, 1.3, 'Photo', 'JPEG / PNG\nor file path', fc='#fdf3ec', ec=ORANGE)
box(0.1, 0.7, 2.0, 1.3, 'Description', 'free text', fc='#fdf3ec', ec=ORANGE)
box(2.6, 1.6, 2.1, 1.65, 'Router', 'input type,\nvalidation,\nerror capture')
box(5.2, 2.55, 2.5, 1.3, 'CNN', f'{BACKBONE_NAME}\nfine-tuned (§2)')
box(5.2, 0.7, 2.5, 1.3, 'Text classifier',
    ('DistilBERT' if TEXT_MODEL == 'bert' else 'TF-IDF + logistic') + '\nfine-tuned (§3)')
box(8.2, 1.6, 2.1, 1.65, 'Confidence\ngate', 'fuse routes,\nabstain below\nthreshold')
box(10.8, 1.6, 2.1, 1.65, 'RAG', 'retrieve policy,\ngenerate, verify\ngrounding (§4)')
box(8.2, 0.05, 4.7, 0.95, 'Feedback store', 'corrections logged, replayed as overrides, '
    'reported as accuracy', fc='#f2f1ea', ec=MUTED)

arrow(2.1, 3.2, 2.6, 2.85); arrow(2.1, 1.35, 2.6, 2.0)
arrow(4.7, 2.85, 5.2, 3.2, 'image'); arrow(4.7, 2.0, 5.2, 1.35, 'text')
arrow(7.7, 3.2, 8.2, 2.85); arrow(7.7, 1.35, 8.2, 2.0)
arrow(10.3, 2.42, 10.8, 2.42, 'category')
arrow(10.9, 1.6, 10.0, 1.0, color=MUTED)
ax.text(12.9, 3.6, 'answer + citations', ha='right', fontsize=9, color=BLUE, fontweight='bold')
ax.annotate('', (12.9, 3.4), (11.85, 3.25), arrowprops=dict(arrowstyle='-|>', color=BLUE, lw=1.4))
ax.set_title('EcoSort assistant: one entry point, two routes, one grounded answer',
             loc='left', fontsize=12, fontweight='bold')
plt.tight_layout(); plt.show()

# %% [markdown]
# ### 5.2 Implementation

# %%
class WasteAssistant:
    """End-to-end waste advisor: photo or description in, grounded instructions out.

    Every public method returns a dictionary with the same shape, including on failure, so a
    caller never has to distinguish between an exception and an answer.
    """

    def __init__(self, cnn, image_threshold=CNN_CONF_THRESHOLD,
                 text_threshold=TEXT_CONF_THRESHOLD, fusion_weight=0.6):
        self.cnn = cnn
        self.image_threshold = image_threshold
        self.text_threshold = text_threshold
        self.fusion_weight = fusion_weight          # weight on the image route when both arrive
        self.feedback = []                          # list of logged corrections
        self.overrides = {}                         # cleaned description -> corrected stream
        self._n = 0

    # ---------- input handling ----------
    @staticmethod
    def _load_image(source):
        """Accept a path, a PIL image or an array; return a 224x224x3 float32 batch.

        Anything a phone might send - greyscale, RGBA, portrait, tiny - is normalised here
        rather than being allowed to reach the model as a shape error.
        """
        if isinstance(source, (str, pathlib.Path)):
            path = pathlib.Path(source)
            if not path.exists():
                raise FileNotFoundError(f'no such image: {path}')
            img = Image.open(path)
        elif isinstance(source, Image.Image):
            img = source
        elif isinstance(source, np.ndarray):
            arr = source
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, -1)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype('uint8')
            img = Image.fromarray(arr[..., :3])
        else:
            raise TypeError(f'unsupported image input: {type(source).__name__}')
        img = img.convert('RGB').resize((IMG_WIDTH, IMG_HEIGHT))
        return np.asarray(img, dtype=np.float32)[None, ...]

    # ---------- routes ----------
    def classify_image(self, source):
        batch = self._load_image(source)
        proba = self.cnn.predict(batch, verbose=0)[0]
        return proba

    def classify_text(self, description):
        result = classify_waste_description(description, top_k=NUM_CLASSES)
        if result['category'] is None:
            raise ValueError(result['error'])
        proba = np.zeros(NUM_CLASSES)
        for cat, p in result['top_k']:
            proba[CLASS_NAMES.index(cat)] = p
        return proba, result['cleaned']

    # ---------- main entry point ----------
    def advise(self, image=None, text=None, generate=True):
        """Classify an item and return grounded disposal instructions.

        Args:
            image: path, PIL image or array. Optional.
            text (str): resident's description. Optional.
            generate (bool): set False to skip the language model and return the
                classification only (used for bulk evaluation).

        Returns:
            dict with status, category, confidence, alternatives, instructions, citations,
            warnings, route and latency_s. `status` is 'ok', 'uncertain' or 'error'.
        """
        t0 = time.time()
        self._n += 1
        out = {'request_id': f'req-{self._n:04d}', 'status': 'ok', 'route': None,
               'category': None, 'confidence': 0.0, 'alternatives': [], 'instructions': None,
               'citations': [], 'warnings': [], 'latency_s': 0.0, '_cleaned_text': None}

        if image is None and (text is None or not str(text).strip()):
            out.update(status='error', warnings=['no input: provide an image, a description, '
                                                 'or both'], latency_s=time.time() - t0)
            return out

        probas, routes = [], []
        if image is not None:
            try:
                probas.append((self.classify_image(image), self.fusion_weight))
                routes.append('image')
            except Exception as exc:
                out['warnings'].append(f'image route failed ({type(exc).__name__}: {exc})')
        if text is not None and str(text).strip():
            try:
                p, cleaned = self.classify_text(text)
                out['_cleaned_text'] = cleaned
                # a confirmed correction for this exact wording wins outright
                if cleaned in self.overrides:
                    p = np.zeros(NUM_CLASSES)
                    p[CLASS_NAMES.index(self.overrides[cleaned])] = 1.0
                    out['warnings'].append('applied a correction recorded from earlier feedback')
                probas.append((p, 1 - self.fusion_weight if routes else 1.0))
                routes.append('text')
            except Exception as exc:
                out['warnings'].append(f'text route failed ({type(exc).__name__}: {exc})')

        if not probas:
            out.update(status='error', latency_s=time.time() - t0)
            return out

        weights = np.array([w for _, w in probas])
        fused = np.average([p for p, _ in probas], axis=0, weights=weights / weights.sum())
        order = np.argsort(-fused)
        out['route'] = '+'.join(routes)
        out['category'] = CLASS_NAMES[int(order[0])]
        out['confidence'] = float(fused[order[0]])
        out['alternatives'] = [(CLASS_NAMES[int(i)], float(fused[i])) for i in order[:3]]

        threshold = (self.image_threshold if routes == ['image']
                     else self.text_threshold if routes == ['text']
                     else min(self.image_threshold, self.text_threshold))
        if out['confidence'] < threshold:
            out['status'] = 'uncertain'
            out['warnings'].append(
                f'confidence {out["confidence"]:.2f} is below the {threshold:.2f} threshold for '
                f'the {out["route"]} route' +
                (' - a written description would help' if routes == ['image'] else
                 ' - a photo would help' if routes == ['text'] else ''))

        if generate:
            instructions, citations = generate_recycling_instructions(out['category'])
            if out['status'] == 'uncertain':
                instructions = (f'Best guess: {out["category"]}. Please confirm before acting.\n'
                                + instructions)
            out['instructions'] = instructions
            out['citations'] = citations

        out['latency_s'] = time.time() - t0
        return out

    # ---------- feedback loop ----------
    def record_feedback(self, result, correct_category, note=''):
        """Log a resident or operator correction and learn the obvious lesson from it."""
        if correct_category not in CLASS_NAMES:
            return {'accepted': False, 'reason': f'unknown stream {correct_category!r}'}
        entry = {'request_id': result.get('request_id'), 'route': result.get('route'),
                 'predicted': result.get('category'), 'confidence': result.get('confidence'),
                 'correct': correct_category, 'agreed': result.get('category') == correct_category,
                 'note': note, 'cleaned_text': result.get('_cleaned_text')}
        self.feedback.append(entry)
        if entry['cleaned_text'] and not entry['agreed']:
            self.overrides[entry['cleaned_text']] = correct_category
        return {'accepted': True, 'logged': len(self.feedback),
                'overrides': len(self.overrides)}

    def feedback_report(self):
        if not self.feedback:
            return pd.DataFrame()
        fb = pd.DataFrame(self.feedback)
        by_route = fb.groupby('route').agg(answers=('agreed', 'size'),
                                           accuracy=('agreed', 'mean'),
                                           mean_confidence=('confidence', 'mean'))
        print(f'{len(fb)} corrections logged | live accuracy {fb["agreed"].mean():.1%} | '
              f'{len(self.overrides)} wording overrides active')
        wrong = fb[~fb['agreed']]
        if len(wrong):
            print('\nmost frequent real-world confusions:')
            for (p, c), n in wrong.groupby(['predicted', 'correct']).size().sort_values(
                    ascending=False).head(5).items():
                print(f'  predicted {p:<20} actually {c:<20} x{n}')
        return by_route.round(3)


def waste_management_assistant(input_data, input_type='image'):
    """Lab-specified entry point. Thin wrapper over `WasteAssistant.advise`.

    Args:
        input_data: an image path/array, or a text description.
        input_type (str): 'image', 'text', or 'auto' to infer from the input.

    Returns:
        dict: waste category, confidence, recycling instructions and policy citations.
    """
    if input_type == 'auto':
        input_type = 'text' if isinstance(input_data, str) and not pathlib.Path(
            str(input_data)).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'} else 'image'
    if input_type == 'image':
        return assistant.advise(image=input_data)
    if input_type == 'text':
        return assistant.advise(text=input_data)
    return {'status': 'error', 'warnings': [f'unknown input_type {input_type!r}; '
                                            f"expected 'image', 'text' or 'auto'"]}


assistant = WasteAssistant(model)
print('assistant ready:', assistant.cnn.name,
      f'| image gate {assistant.image_threshold:.2f} | text gate {assistant.text_threshold:.2f}')


def show(result, width=92):
    """Readable rendering of an assistant response."""
    print(f'[{result["request_id"]}] status={result["status"]} route={result["route"]} '
          f'({result["latency_s"]:.2f}s)')
    if result['category']:
        alts = ', '.join(f'{c} {p:.2f}' for c, p in result['alternatives'])
        print(f'  category : {result["category"]} ({result["confidence"]:.2f})   [{alts}]')
    for w in result['warnings']:
        print(f'  warning  : {w}')
    if result['instructions']:
        print('  instructions:')
        wrap(result['instructions'], width=width, indent='    ')
        print('  sources:')
        for c in result['citations']:
            print(f'    - {c["source"]} / {c["section"]}'
                  + (f' (effective {c["effective_date"]})' if c['effective_date'] else ''))

# %% [markdown]
# ### 5.3 Testing the integrated system
#
# #### A single photograph, end to end

# %%
demo_row = img_test[img_test['label'] == 'Glass'].iloc[3]
print(f'input: {demo_row["path"]}  (true stream: {demo_row["label"]})\n')
fig, ax = plt.subplots(figsize=(2.6, 2.6))
ax.imshow(Image.open(demo_row['path']).resize((224, 224))); ax.axis('off')
ax.set_title('resident photo', fontsize=9)
plt.show()

show(waste_management_assistant(str(demo_row['path']), input_type='image'))

# %% [markdown]
# #### A written description, end to end

# %%
show(waste_management_assistant('a greasy cardboard pizza box with cheese stuck to it',
                                input_type='text'))

# %% [markdown]
# #### Both at once: probability fusion

# %%
amb = img_test[img_test['label'] == 'Miscellaneous Trash'].iloc[1]
print(f'photo of {amb["label"]} + the resident typing a description of the same item\n')
img_only = assistant.advise(image=str(amb['path']), generate=False)
txt_only = assistant.advise(text='broken household item made of mixed materials', generate=False)
both = assistant.advise(image=str(amb['path']),
                        text='broken household item made of mixed materials', generate=False)
for label, r in [('image only', img_only), ('text only', txt_only), ('fused', both)]:
    print(f'  {label:<12} {r["category"]:<22} {r["confidence"]:.2f}  '
          f'[{", ".join(f"{c} {p:.2f}" for c, p in r["alternatives"])}]')

# %% [markdown]
# #### Bulk evaluation on the held-out test sets
#
# Classification is run over a sample from each test split with generation switched off (the
# language model is the slow part and was already evaluated in §4.5); a handful of full
# round-trips are timed separately so the quoted latency is the real one.

# %%
N_EVAL = 60
rng = np.random.default_rng(SEED)
img_sample = img_test.iloc[rng.choice(len(img_test), N_EVAL, replace=False)]
txt_sample = txt_test.iloc[rng.choice(len(txt_test), N_EVAL, replace=False)]

img_rows = []
for r in img_sample.itertuples():
    res = assistant.advise(image=str(r.path), generate=False)
    img_rows.append({'true': r.label, 'pred': res['category'], 'conf': res['confidence'],
                     'status': res['status'], 'latency': res['latency_s']})
txt_rows = []
for r in txt_sample.itertuples():
    res = assistant.advise(text=r.clean, generate=False)
    txt_rows.append({'true': r.category, 'pred': res['category'], 'conf': res['confidence'],
                     'status': res['status'], 'latency': res['latency_s']})

e2e = pd.DataFrame([
    {'route': 'image', **{
        'n': len(img_rows),
        'accuracy': np.mean([r['true'] == r['pred'] for r in img_rows]),
        'answered (status ok)': np.mean([r['status'] == 'ok' for r in img_rows]),
        'accuracy when answered': np.mean([r['true'] == r['pred'] for r in img_rows
                                           if r['status'] == 'ok']),
        'classify latency (s)': np.mean([r['latency'] for r in img_rows])}},
    {'route': 'text', **{
        'n': len(txt_rows),
        'accuracy': np.mean([r['true'] == r['pred'] for r in txt_rows]),
        'answered (status ok)': np.mean([r['status'] == 'ok' for r in txt_rows]),
        'accuracy when answered': np.mean([r['true'] == r['pred'] for r in txt_rows
                                           if r['status'] == 'ok']),
        'classify latency (s)': np.mean([r['latency'] for r in txt_rows])}},
]).set_index('route')
print(e2e.round(3).to_string())

t0 = time.time()
for r in img_sample.head(3).itertuples():
    assistant.advise(image=str(r.path))
print(f'\nfull round trip including retrieval and generation: {(time.time() - t0) / 3:.1f}s per request')

# %% [markdown]
# #### Edge cases
#
# Robustness is a property of the integration, not of the models. Every one of these inputs is
# something a deployed endpoint will eventually receive.

# %%
tiny = np.random.default_rng(0).integers(0, 255, (12, 7), dtype=np.uint8)        # greyscale, tiny
portrait = Image.open(img_test.iloc[0]['path']).resize((300, 700)).convert('RGBA')  # wrong shape
corrupt = pathlib.Path('build/_corrupt.jpg')
corrupt.parent.mkdir(exist_ok=True)
corrupt.write_bytes(b'not actually a jpeg')

EDGE_CASES = [
    ('missing file', dict(image='RealWaste/does_not_exist.jpg')),
    ('corrupt file', dict(image=str(corrupt))),
    ('greyscale 12x7 array', dict(image=tiny)),
    ('portrait RGBA image', dict(image=portrait)),
    ('wrong type (int)', dict(image=42)),
    ('empty string', dict(text='')),
    ('punctuation only', dict(text='!!! ???')),
    ('no input at all', dict()),
    ('non-waste sentence', dict(text='what is the capital of Australia')),
    ('very long text', dict(text='plastic ' * 400)),
    ('valid image + unusable text', dict(image=str(img_test.iloc[0]['path']), text='12345')),
]

edge_rows = []
for name, kwargs in EDGE_CASES:
    try:
        r = assistant.advise(generate=False, **kwargs)
        edge_rows.append({'case': name, 'status': r['status'],
                          'category': r['category'] or '-',
                          'confidence': round(r['confidence'], 2),
                          'warning': (r['warnings'][0][:66] + '...') if r['warnings'] else ''})
    except Exception as exc:                                   # must never happen
        edge_rows.append({'case': name, 'status': 'UNCAUGHT EXCEPTION',
                          'category': '-', 'confidence': 0,
                          'warning': f'{type(exc).__name__}: {exc}'})
edges = pd.DataFrame(edge_rows)
print(edges.to_string(index=False))
print(f'\nuncaught exceptions: {(edges["status"] == "UNCAUGHT EXCEPTION").sum()} of {len(edges)}')
corrupt.unlink(missing_ok=True)

# %% [markdown]
# #### The feedback loop
#
# A deployed assistant is wrong a measurable fraction of the time, and the cheapest source of
# labelled data is the resident who just saw the answer. Corrections are logged, aggregated into
# a live accuracy report, and - for text inputs - replayed as exact-wording overrides, so the
# same mistake is not repeated on the same phrasing while a retrain is pending.

# %%
assistant_fb = WasteAssistant(model)
for r in txt_sample.head(25).itertuples():
    res = assistant_fb.advise(text=r.clean, generate=False)
    assistant_fb.record_feedback(res, r.category)
for r in img_sample.head(15).itertuples():
    res = assistant_fb.advise(image=str(r.path), generate=False)
    assistant_fb.record_feedback(res, r.label)

report = assistant_fb.feedback_report()
print()
print(report.to_string())

# %%
# The override in action: a wording the model gets wrong, corrected once, then re-asked.
# A fresh assistant is used to find the failing case - `assistant_fb` has already learned
# overrides from the corrections logged above, so it would report no errors left.
probe = WasteAssistant(model)
wrong_rows = [(r.clean, r.category) for r in txt_test.head(80).itertuples()
              if probe.advise(text=r.clean, generate=False)['category'] != r.category]

if wrong_rows:
    wording, truth = wrong_rows[0]
    note = f'a real error: {len(wrong_rows)} of the first 80 test descriptions are misread'
else:
    # The text classifier makes no errors on this synthetic test set (§3.4), so there is no
    # real failure to repair. The mechanism is shown instead on a deliberately under-specified
    # wording, where the resident knows something the description does not say: this scrap of
    # fabric is laminated, so it belongs in Miscellaneous Trash rather than Textile Trash.
    wording, truth = 'small piece of coated fabric', 'Miscellaneous Trash'
    note = ('no real errors on this test set, so the mechanism is shown on an '
            'under-specified wording the resident can disambiguate')

print(f'({note})\n')
fresh = WasteAssistant(model)
before = fresh.advise(text=wording, generate=False)
print(f'description : {wording!r}')
print(f'before      : {before["category"]} ({before["confidence"]:.2f})   '
      f'resident says: {truth}')
print(f'feedback    : {fresh.record_feedback(before, truth, note="resident correction")}')
after = fresh.advise(text=wording, generate=False)
print(f'after       : {after["category"]} ({after["confidence"]:.2f})   {after["warnings"]}')
print('\nThe override is a patch, not a fix: it only matches this exact wording. Its real value '
      'is that the correction is now a labelled example for the next retrain.')

# %% [markdown]
# ### 5.4 What this system is, and what it is not
#
# **What was built.** A single endpoint that accepts a photograph, a sentence, or both; routes
# each to a model trained and evaluated on a held-out split; fuses them when both are present;
# refuses to answer confidently when it should not; and returns disposal instructions grounded
# in - and citing - the Metro City policy documents, falling back to quoted policy when the
# generator drifts. Every component is measured rather than asserted, and the integration layer
# is tested against the malformed inputs a real endpoint receives.
#
# **What it is not.** Four limitations are structural, not bugs:
#
# 1. **Domain shift on photographs.** Every training image is a 524x524 facility photograph
#    under one lighting setup (§1.1). Resident phone photos will be framed, lit and cropped
#    differently, and the honest expectation is a meaningful drop from the test accuracy in
#    §2.6. Augmentation mitigates it; only real user photos will fix it.
# 2. **Synthetic text.** A 309-word templated vocabulary (§1.2) is not how people write, and
#    the text classifier scoring 100% (§3.4) is a statement about the benchmark, not about the
#    model. Its true accuracy on resident free text is unknown and certainly lower; the gap
#    between DistilBERT and the linear baseline, invisible here, is where it should show up.
# 3. **A small generator.** FLAN-T5-base produces correct but plain instructions, and the
#    grounding guard fires rather than being decorative. A larger model would write better
#    prose; the architecture, not the model, is what keeps it factual.
# 4. **Nine streams, closed world.** Neither classifier can say "this is not waste" or "this is
#    hazardous"; it can only say it is unsure. Batteries, e-waste and chemicals are named in the
#    Miscellaneous Trash policy but have no class of their own, which for a safety-relevant
#    category is the single most important gap to close next.
#
# **Where the next unit of effort goes.** In order: collect several hundred resident photos and
# fine-tune on them, because that is where the largest real-world gap is; add an explicit
# hazardous-waste class; and route the feedback store into a weekly retrain so the corrections
# logged above become training data instead of patches.
