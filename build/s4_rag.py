# %% [markdown]
# ## Part 4: Recycling instruction generation with RAG
#
# Knowing an item is Glass is only half an answer. The resident still needs to know that caps
# come off, that the jar should be rinsed, that labels can stay, and that Metro City will not
# take window glass or Pyrex in the same bin. That information lives in the policy documents,
# it changes when the council changes it, and it must not be invented — which is the case for
# **retrieval-augmented generation** rather than a model asked to recall recycling rules from
# its pre-training.
#
# The pipeline is: *query* → **retrieve** the handful of policy chunks that answer it →
# **generate** an instruction conditioned on those chunks → **verify** that what came out is
# supported by what went in. Each stage is measured: retrieval with recall@k and MRR against
# the known policy-to-stream mapping (§4.2), generation with a grounding score and ROUGE-L
# against the retrieved text (§4.3), and the whole thing end-to-end in §4.5.

# %% [markdown]
# ### 4.1 Embedding the corpus and building the index
#
# The policy chunks from §1.4d are embedded with **all-MiniLM-L6-v2**: a 6-layer, 22M-parameter
# sentence encoder producing 384-dimensional vectors, trained with a contrastive objective on
# over a billion sentence pairs. It is the right tool for this specific job because the task is
# *asymmetric* short-query retrieval — a nine-word question against a thirty-word policy
# paragraph — which is exactly what it was tuned for, and because at 80 MB it loads in seconds
# next to the two models already in memory.
#
# Embeddings are L2-normalised, so cosine similarity is a dot product and the whole index is one
# small matrix multiply. A vector database (FAISS, Chroma) would add a dependency and an
# approximation to a search that scans every row exactly, in microseconds; it becomes the right
# answer somewhere around 10^5 chunks, not at this scale.

# %%
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
t0 = time.time()
embedder = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)
CORPUS_EMB = embedder.encode(corpus['text'].tolist(), normalize_embeddings=True,
                             batch_size=32, show_progress_bar=False)
print(f'{EMBED_MODEL_NAME} loaded and {len(corpus)} chunks embedded in {time.time() - t0:.1f}s')
print(f'index: {CORPUS_EMB.shape[0]} x {CORPUS_EMB.shape[1]} float32 '
      f'({CORPUS_EMB.nbytes / 1024:.0f} kB), L2-normalised: '
      f'{np.allclose(np.linalg.norm(CORPUS_EMB, axis=1), 1.0)}')

# A sparse index over the same chunks, as the baseline the dense retriever has to beat.
SPARSE_VEC = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), sublinear_tf=True,
                             stop_words='english')
CORPUS_SPARSE = SPARSE_VEC.fit_transform(corpus['text'])


def retrieve(query, k=3, category=None, category_boost=0.15, require_category=False,
             sections=None, section_boost=0.12, avoid_sections=None, method='dense'):
    """Return the top-k chunks for a query.

    Three pieces of metadata shape the ranking, all available at serving time:

    - `category`: the waste stream, which a classifier has already produced. Chunks tagged
      with that stream get a similarity bonus - soft filtering, so a genuinely relevant
      general-rules chunk can still surface.
    - `sections`: the policy sections that answer this *kind* of question. "Which items must
      NOT go in?" is answered by Non-Acceptable Items; without the hint the retriever
      cheerfully returns the near-identical Acceptable Items block instead.
    - `avoid_sections`: sections that must not reach the generator at all (see below).
    """
    if method == 'dense':
        q = embedder.encode([query], normalize_embeddings=True)[0]
        sims = CORPUS_EMB @ q
    else:
        q = SPARSE_VEC.transform([query])
        sims = (CORPUS_SPARSE @ q.T).toarray().ravel()

    sims = sims.astype(float).copy()
    if category is not None:
        on_stream = corpus['categories'].map(lambda cs: category in cs).to_numpy()
        if require_category:
            # Miscellaneous Trash has no "Collection Method" section of its own, and without
            # this filter the nearest one by similarity is Metal's - which would quote a
            # resident the wrong stream's rules under the right stream's heading.
            sims[~on_stream] = -np.inf
        else:
            sims += category_boost * on_stream
    if sections:
        want = {s.lower() for s in sections}
        sims += section_boost * corpus['section'].str.lower().isin(want).to_numpy()
    if avoid_sections:
        # a hard guardrail rather than a soft penalty: when the resident asks what is banned,
        # the list of accepted items must not be in the context at all. It is near-identical
        # wording with the opposite meaning, and a small model cannot be relied on to tell
        # them apart.
        block = {s.lower() for s in avoid_sections}
        sims[corpus['section'].str.lower().isin(block).to_numpy()] = -np.inf

    top = np.argsort(-sims)[:k]
    return corpus.iloc[top].assign(score=sims[top]).reset_index(drop=True)


def format_context(chunks):
    """Policy extracts as prompt context: content only, one bullet per chunk.

    This deliberately uses `body` rather than `text`. Earlier versions passed the full chunk
    (document title, section heading, content) and prefixed each line with its source, and the
    generator answered by echoing that scaffolding - "Prepare: Preparation Instructions." The
    headings stay in the embedded `text`, where they help retrieval, and out of the prompt,
    where they only invite copying.
    """
    return '\n'.join(f'- {r.body}' for r in chunks.itertuples())


demo_q = 'How should I recycle Glass?'
print(f'\nquery: {demo_q!r}\n')
for r in retrieve(demo_q, k=3).itertuples():
    print(f'  [{r.score:.3f}] {r.source} / {r.section}')
    wrap(r.text[:240] + ('...' if len(r.text) > 240 else ''), width=92, indent='        ')

# %% [markdown]
# ### 4.2 Does the retriever actually retrieve?
#
# This is measurable without any human judgement. Every chunk carries the streams its document
# covers, so for the query *"How do I recycle Glass?"* the relevant set is exactly the chunks
# tagged Glass. Three retrievers are compared over 36 queries — four phrasings per stream, from
# formal to the way a resident would actually type — using:
#
# - **recall@k**: is at least one relevant chunk in the top k?
# - **precision@3**: how much of the context handed to the generator is on-topic?
# - **MRR**: how high up the first relevant chunk lands.

# %%
QUERY_TEMPLATES = [
    'How to recycle {c}?',
    'What bin does {c} go in and how do I prepare it?',
    'Metro City rules for disposing of {c}',
    'I have some {c} at home - what do I do with it?',
]
eval_queries = [(t.format(c=cat.lower()), cat) for cat in CLASS_NAMES for t in QUERY_TEMPLATES]
print(f'{len(eval_queries)} evaluation queries ({len(QUERY_TEMPLATES)} phrasings x {NUM_CLASSES} streams)')


def retrieval_scores(method, use_category, k_list=(1, 3, 5)):
    hits = {k: [] for k in k_list}
    precision3, rr = [], []
    for q, cat in eval_queries:
        got = retrieve(q, k=max(k_list), category=cat if use_category else None, method=method)
        rel = got['categories'].map(lambda cs: cat in cs).to_numpy()
        for k in k_list:
            hits[k].append(bool(rel[:k].any()))
        precision3.append(rel[:3].mean())
        first = np.where(rel)[0]
        rr.append(1 / (first[0] + 1) if len(first) else 0.0)
    out = {f'recall@{k}': float(np.mean(hits[k])) for k in k_list}
    out['precision@3'] = float(np.mean(precision3))
    out['MRR'] = float(np.mean(rr))
    return out


retr_results = pd.DataFrame([
    {'retriever': 'TF-IDF (sparse), query only', **retrieval_scores('sparse', False)},
    {'retriever': 'MiniLM (dense), query only', **retrieval_scores('dense', False)},
    {'retriever': 'MiniLM + stream metadata boost', **retrieval_scores('dense', True)},
]).set_index('retriever')
print(retr_results.round(3).to_string())

fig, ax = plt.subplots(figsize=(8.6, 3.4))
pos = np.arange(len(retr_results))
for i, (col, color) in enumerate(zip(['recall@1', 'recall@3', 'MRR'], [BLUE, AQUA, ORANGE])):
    ax.barh(pos + (i - 1) * 0.26, retr_results[col], 0.25, color=color, label=col)
ax.set_yticks(pos); ax.set_yticklabels(retr_results.index, fontsize=9)
ax.invert_yaxis(); ax.grid(axis='y', visible=False); ax.tick_params(axis='y', length=0)
ax.set_xlim(0, 1.05)
ax.set_title('Retrieval quality over 36 queries')
ax.legend(ncol=3, loc='lower right')
plt.tight_layout(); plt.show()

RETRIEVER = 'dense'
print(f'production retriever: MiniLM dense + metadata boost (the stream is known at serving '
      f'time because a classifier produced it)')

# %%
# Where the sparse retriever fails and the dense one does not: vocabulary mismatch.
probe_q = 'I have some food organics at home - what do I do with it?'
for method in ['sparse', 'dense']:
    got = retrieve(probe_q, k=3, method=method)
    rel = got['categories'].map(lambda cs: 'Food Organics' in cs).to_numpy()
    print(f'\n{method:>6}: ' + ' | '.join(
        f'{"OK " if r else "off"} {s}/{sec}' for r, s, sec in zip(rel, got['source'], got['section'])))

# %% [markdown]
# ### 4.3 Generation: FLAN-T5 conditioned on retrieved policy
#
# **FLAN-T5-base** (250M parameters) is the generator. It is an instruction-tuned
# encoder-decoder, which suits this task better than a same-sized decoder-only model for two
# reasons: the encoder reads the retrieved policy in full before a single token is produced,
# and instruction tuning means it follows "use only the context below" without few-shot
# examples. It is also small enough to run on CPU in a few seconds per answer.
#
# Expectations should be set honestly: a 250M-parameter model writes plain, occasionally clumsy
# prose and will happily copy a whole paragraph if allowed. That is acceptable here — and in
# fact *desirable*, because the product requirement is that the instruction reflects **Metro
# City's policy**, not the model's general knowledge of recycling. The metrics below are chosen
# to measure precisely that.

# %%
from transformers import AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

GEN_MODEL_NAME = 'google/flan-t5-base'
t0 = time.time()
gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
generator = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME).to(DEVICE).eval()
print(f'{GEN_MODEL_NAME}: {sum(p.numel() for p in generator.parameters()) / 1e6:.0f}M parameters, '
      f'loaded in {time.time() - t0:.0f}s')

PROMPT = (
    "You are Metro City's waste advisor. Using only the policy extracts below, write clear "
    "disposal instructions for a resident.\n"
    "Say which bin or collection the item goes in, how to prepare it, and one thing that must "
    "not be put in with it.\n\n"
    "Policy extracts:\n{context}\n\n"
    "Question: {question}\n"
    "Instructions:")


def build_prompt(question, chunks):
    return PROMPT.format(context=format_context(chunks), question=question)


def content_words(text):
    return [w for w in re.findall(r"[a-z]+", str(text).lower())
            if w not in ENGLISH_STOP_WORDS and len(w) > 2]


def grounding_score(generated, context):
    """Share of the answer's content words that appear in the retrieved context.

    1.0 means every substantive word is traceable to a policy extract; a low score means the
    model is writing from its pre-training instead of from Metro City's rules.
    """
    gen, ctx = content_words(generated), set(content_words(context))
    return float(np.mean([w in ctx for w in gen])) if gen else 0.0


def rouge_l(generated, reference):
    """ROUGE-L F1 via longest common subsequence over content words."""
    a, b = content_words(generated), content_words(reference)
    if not a or not b:
        return 0.0
    dp = np.zeros((len(a) + 1, len(b) + 1), dtype=np.int32)
    for i, x in enumerate(a, 1):
        for j, y in enumerate(b, 1):
            dp[i, j] = dp[i - 1, j - 1] + 1 if x == y else max(dp[i - 1, j], dp[i, j - 1])
    lcs = int(dp[-1, -1])
    p, r = lcs / len(a), lcs / len(b)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def distinct_2(text):
    """Lexical variety: share of unique bigrams. Low values mean the model is looping."""
    toks = re.findall(r"[a-z]+", text.lower())
    bg = list(zip(toks, toks[1:]))
    return len(set(bg)) / len(bg) if bg else 0.0


def generate_with(model, tokenizer, prompt, max_new_tokens=110, **gen_kwargs):
    """Run any of the seq2seq models in this part on a prompt.

    Small seq2seq models sometimes carry on past the answer and re-emit the prompt's own
    scaffolding ("... compostable bags. Question: Which bin ..."), so the decode is cut at the
    first structural marker it produces.
    """
    enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, **gen_kwargs)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return re.split(r'\s*(?:Question|Policy extracts|Answer)\s*:', text)[0].strip()


def generate(prompt, **kw):
    """Shorthand for the default generator (FLAN-T5-base)."""
    return generate_with(generator, gen_tokenizer, prompt, **kw)

# %% [markdown]
# #### Decoding strategy
#
# The sampling parameters are not cosmetic here. Temperature and nucleus sampling buy variety
# at the cost of *invention*, and invention is the one thing a policy assistant may not do. Four
# strategies are run over three streams and scored on grounding (how much of the answer is
# traceable to the retrieved policy), ROUGE-L against that policy, length, and distinct-2.

# %%
STRATEGIES = {
    'greedy': {},
    'beam search (4)': {'num_beams': 4, 'no_repeat_ngram_size': 3, 'early_stopping': True},
    'nucleus (p=0.9, T=0.7)': {'do_sample': True, 'top_p': 0.9, 'temperature': 0.7, 'top_k': 0},
    'nucleus (p=0.95, T=1.2)': {'do_sample': True, 'top_p': 0.95, 'temperature': 1.2, 'top_k': 0},
}
SAMPLE_STREAMS = ['Glass', 'Food Organics', 'Miscellaneous Trash']

torch.manual_seed(SEED)
dec_rows, dec_examples = [], {}
for sname, kwargs in STRATEGIES.items():
    scores = []
    for cat in SAMPLE_STREAMS:
        q = f'How do I recycle {cat.lower()}?'
        ctx = retrieve(q, k=3, category=cat)
        prompt = build_prompt(q, ctx)
        ctx_text = ' '.join(ctx['text'])
        t0 = time.time()
        ans = generate(prompt, **kwargs)
        scores.append((grounding_score(ans, ctx_text), rouge_l(ans, ctx_text),
                       len(ans.split()), distinct_2(ans), time.time() - t0))
        dec_examples.setdefault(sname, {})[cat] = ans
    g, r, l, d, s = np.array(scores).mean(0)
    dec_rows.append({'strategy': sname, 'grounding': g, 'ROUGE-L vs context': r,
                     'words': l, 'distinct-2': d, 'seconds': s})
    print(f'  {sname:<24} grounding {g:.3f} | ROUGE-L {r:.3f} | {l:.0f} words | {s:.1f}s')

decoding = pd.DataFrame(dec_rows).set_index('strategy')
decoding.round(3)

# %%
fig, ax = plt.subplots(figsize=(8.6, 3.3))
pos = np.arange(len(decoding))
ax.barh(pos - 0.2, decoding['grounding'], 0.38, color=BLUE, label='grounding (higher = safer)')
ax.barh(pos + 0.2, decoding['distinct-2'], 0.38, color=ORANGE, label='distinct-2 (variety)')
ax.set_yticks(pos); ax.set_yticklabels(decoding.index, fontsize=9)
ax.invert_yaxis(); ax.grid(axis='y', visible=False); ax.tick_params(axis='y', length=0)
ax.set_xlim(0, 1.05)
ax.set_title('Sampling buys variety by inventing words that are not in the policy')
ax.legend(loc='lower right')
plt.tight_layout(); plt.show()

print('Same question, three decoders:\n')
for sname in ['greedy', 'beam search (4)', 'nucleus (p=0.95, T=1.2)']:
    print(f'[{sname}]')
    wrap(dec_examples[sname]['Glass'], width=92, indent='   ')
    print()

DECODER = max(decoding.index, key=lambda s: decoding.loc[s, 'grounding'] * 0.7
              + decoding.loc[s, 'ROUGE-L vs context'] * 0.3)
DECODE_KWARGS = STRATEGIES[DECODER]
print(f'selected decoder: {DECODER}')

# %% [markdown]
# The sampling experiment settles the decoder, and it also exposes a second problem: asked one
# broad question, FLAN-T5 returns **a single clause** — grounded, and inadequate as disposal
# advice. Raising `min_new_tokens` only pads it; the model has answered the question it was
# asked.
#
# So ask better questions. A resident needs three specific things, and asking for them
# separately buys three things a single prompt cannot:
#
# 1. **Narrower retrieval.** Each field queries the index on its own, so the preparation advice
#    is conditioned on preparation policy rather than on whatever one vague query surfaced.
# 2. **Per-field verification.** Grounding and section attribution can be checked field by
#    field instead of over a blur of text — which is what makes the "right section" metric in
#    §4.4 possible at all.
# 3. **A fixed output shape**, so the assistant's answer is the same three lines every time.
#
# What it does *not* fix on its own is the terseness: zero-shot, the three fields are three
# short clauses. §4.4 is where that gets solved.

# %%
# Each field states the question it asks and the policy sections that answer it. The section
# list is used twice: as a retrieval hint now, and as the source of training targets in §4.4.
FIELD_SPEC = [
    ('Bin', 'Which bin or collection service does {c} go in?',
     ['Collection Method'], []),
    ('Prepare', 'How should {c} be prepared or cleaned before it is put out?',
     ['Preparation Instructions'], []),
    ('Do not include', 'Which items must NOT be put in with {c}?',
     ['Non-Acceptable Items', 'Items That Cannot Be Recycled'], ['Acceptable Items']),
]
# Two wordings were tried. A one-shot version, with a worked example on an invented stream,
# was *worse*: FLAN-T5 copied the example's hedge and answered "Not specified in Metro City
# policy" even where the extract plainly answered the question. The zero-shot instruction below
# keeps the escape clause (Miscellaneous Trash genuinely has no Collection Method section) but
# does not demonstrate using it.
FIELD_INSTRUCTION = (
    "Answer the resident's question in one full sentence, using only the Metro City policy "
    "extracts provided, and reusing the policy's own wording. If the extracts do not answer "
    "the question, reply 'Not specified in Metro City policy.'\n\n"
    "Policy extracts:\n{context}\n"
    "Question: {question}\nAnswer:")


def generate_structured(category, model=None, tokenizer=None, k=2, **kw):
    """Compose a three-field instruction, retrieving separately for each field.

    Returns (text, chunks_used). Each field is generated from its own retrieval, so the
    preparation advice is conditioned on preparation policy rather than on whatever the
    broad query happened to surface.
    """
    model = generator if model is None else model
    tokenizer = gen_tokenizer if tokenizer is None else tokenizer
    lines, used, answers = [f'{category} - Metro City disposal instructions'], [], []
    for field, template, sections, avoid in FIELD_SPEC:
        q = template.format(c=category.lower())
        # retrieve-then-filter: pull a slightly wider net, then keep only the chunks that
        # belong to the sections this field is about. Without the filter the second-ranked
        # chunk is usually off-topic (a Benefits paragraph, or the accepted-items list from a
        # cross-cutting document) and the generator answers from it.
        wide = retrieve(q, k=k + 2, category=category, require_category=True,
                        sections=sections, avoid_sections=avoid)
        on_section = wide[wide['section'].str.lower().isin({s.lower() for s in sections})]
        chunks = (on_section if len(on_section) else wide).head(k).reset_index(drop=True)
        used.append(chunks)
        prompt = FIELD_INSTRUCTION.format(context=format_context(chunks), question=q)
        answer = generate_with(model, tokenizer, prompt, max_new_tokens=70,
                               **(kw or DECODE_KWARGS)).strip()
        answers.append((field, answer))
        lines.append(f'{field}: {answer}')
    used = pd.concat(used).drop_duplicates(subset='chunk_id').reset_index(drop=True)
    return '\n'.join(lines), used, answers


def fields_body(fields):
    """The model's own words only - grounding is scored on these, not on the fixed scaffolding."""
    return ' '.join(a for _, a in fields)


t0 = time.time()
struct_text, struct_chunks, struct_fields = generate_structured('Glass')
struct_body = fields_body(struct_fields)
print(f'structured answer in {time.time() - t0:.1f}s '
      f'({len(struct_text.split())} words, {len(struct_chunks)} chunks cited)\n')
wrap(struct_text, width=92, indent='  ')

single_ctx = retrieve('How do I recycle glass?', k=3, category='Glass')
single_text = generate(build_prompt('How do I recycle glass?', single_ctx), **DECODE_KWARGS)
print(f'\n{"":<24}{"generated words":>17}{"grounding":>12}{"ROUGE-L vs policy":>20}')
for label, body, ctx in [('single broad prompt', single_text, ' '.join(single_ctx['text'])),
                         ('structured 3-field', struct_body, ' '.join(struct_chunks['text']))]:
    print(f'  {label:<22}{len(body.split()):>17}{grounding_score(body, ctx):>12.3f}'
          f'{rouge_l(body, " ".join(single_ctx["text"])):>20.3f}')

# %% [markdown]
# ### 4.4 Fine-tuning the generator on grounded instruction pairs
#
# Zero-shot FLAN-T5 answers each field, but it answers in its own voice and at its own length —
# sometimes a fragment, sometimes a sentence lifted whole from the context. The product needs a
# consistent register: short, imperative, and phrased the way Metro City phrases it.
#
# That supervision needs no annotation, because the policy documents **already contain the
# targets**. Each field maps to a section: *Bin* to Collection Method, *Prepare* to Preparation
# Instructions, *Do not include* to Non-Acceptable Items. Pairing each field prompt (in two
# phrasings) with its section body gives a small supervised set of exactly the task the
# production path runs.
#
# **FLAN-T5-small** (80M) is the student rather than base: the point is to demonstrate and
# measure the procedure on CPU in minutes, and a smaller model makes the comparison against
# zero-shot base more informative. **Two streams are held out entirely** (Paper and Textile
# Trash), so the evaluation reports generalisation to a stream the model never saw a target
# for — not the memorisation of nine answers.
#
# Three things are measured, and the second is the one that matters most for a policy
# assistant:
#
# - **grounding** — share of the answer's content words that appear in the retrieved policy;
# - **right section** — whether each field's answer is closer to the policy section that
#   *answers* it than to the section most likely to be confused with it (for "what must not go
#   in", that decoy is the near-identically-worded list of items that *may* go in). A high
#   grounding score alone does not catch this failure: both lists are in the policy, so an
#   answer drawn from the wrong one is perfectly "grounded" and exactly backwards;
# - **ROUGE-L** against the reference instruction assembled from policy.

# %%
HELD_OUT = ['Paper', 'Textile Trash']
# the target for each field is the body of the policy section that field asks about
FIELD_SOURCE_SECTIONS = {field: sections for field, _, sections, _ in FIELD_SPEC}
FIELD_AVOID = {field: avoid for field, _, _, avoid in FIELD_SPEC}
# ...and the section a confused model is most likely to answer from instead. "Acceptable
# Items" and "Non-Acceptable Items" are lexically almost identical, so an answer drawn from
# the wrong one is the dangerous failure here: it tells a resident to bin the one thing the
# policy forbids. §4.4 scores how often each model lands on the right side of that pair.
DECOY_SECTIONS = {'Bin': ['Benefits'],
                  'Prepare': ['Acceptable Items'],
                  'Do not include': ['Acceptable Items']}


def section_text(cat, names):
    """The body of the first matching section in the stream's own policy document."""
    mine = corpus[corpus['categories'].map(lambda cs: cs == [cat])]
    for name in names:
        hit = mine[mine['section'].str.lower() == name.lower()]
        if len(hit):
            body = hit.iloc[0]['text'].split(': ', 1)[-1].replace('\n- ', '; ').replace('\n', ' ')
            body = re.sub(r'\s+', ' ', body).strip()
            return re.sub(r'^-\s*', '', body)          # drop the first list bullet
    return ''


def reference_instruction(cat):
    """The gold three-field instruction for a stream, assembled straight from policy."""
    lines = [f'{cat} - Metro City disposal instructions']
    for field, names in FIELD_SOURCE_SECTIONS.items():
        lines.append(f'{field}: {section_text(cat, names) or "Not specified in Metro City policy."}')
    return '\n'.join(lines)


PARAPHRASES = {
    'Bin': ['Which bin or collection service does {c} go in?', 'Where do I put {c}?'],
    'Prepare': ['How should {c} be prepared or cleaned before it is put out?',
                'What do I need to do to {c} before collection?'],
    'Do not include': ['Which items must NOT be put in with {c}?',
                       'What is not accepted in the {c} stream?'],
}

train_pairs = []
for cat in CLASS_NAMES:
    if cat in HELD_OUT:
        continue
    for field, names in FIELD_SOURCE_SECTIONS.items():
        target = section_text(cat, names)
        if not target:
            continue
        for q_tmpl in PARAPHRASES[field]:
            q = q_tmpl.format(c=cat.lower())
            ctx = retrieve(q, k=2, category=cat, sections=names,
                           avoid_sections=FIELD_AVOID[field])
            prompt = FIELD_INSTRUCTION.format(context=format_context(ctx), question=q)
            train_pairs.append({'input': prompt, 'target': target,
                                'category': cat, 'field': field})

pairs_df = pd.DataFrame(train_pairs)
print(f'{len(pairs_df)} training pairs over {pairs_df["category"].nunique()} streams x '
      f'{pairs_df["field"].nunique()} fields ({", ".join(HELD_OUT)} held out entirely)')
print(f'target length: {pairs_df["target"].str.split().str.len().mean():.0f} words on average')
print('\nreference instruction for Glass (the shape the model is being taught):')
wrap(reference_instruction('Glass'), width=92, indent='   ')

# %%
FT_MODEL_NAME = 'google/flan-t5-small'
ft_tokenizer = AutoTokenizer.from_pretrained(FT_MODEL_NAME)
ft_model = AutoModelForSeq2SeqLM.from_pretrained(FT_MODEL_NAME).to(DEVICE)

enc_in = ft_tokenizer(pairs_df['input'].tolist(), truncation=True, max_length=448,
                      padding='max_length', return_tensors='pt')
enc_out = ft_tokenizer(pairs_df['target'].tolist(), truncation=True, max_length=128,
                       padding='max_length', return_tensors='pt')
labels = enc_out['input_ids'].clone()
labels[labels == ft_tokenizer.pad_token_id] = -100      # pad tokens must not contribute loss

ft_loader = DataLoader(TensorDataset(enc_in['input_ids'], enc_in['attention_mask'], labels),
                       batch_size=4, shuffle=True)
ft_optim = torch.optim.AdamW(ft_model.parameters(), lr=3e-4, weight_decay=0.01)
EPOCHS_FT = 3
ft_sched = get_linear_schedule_with_warmup(ft_optim, 10, len(ft_loader) * EPOCHS_FT)

torch.manual_seed(SEED)
ft_model.train()
t0 = time.time()
for epoch in range(1, EPOCHS_FT + 1):
    running = 0.0
    for ids, mask, lab in ft_loader:
        ft_optim.zero_grad()
        loss = ft_model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE),
                        labels=lab.to(DEVICE)).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ft_model.parameters(), 1.0)
        ft_optim.step()
        ft_sched.step()
        running += loss.item()
    print(f'  epoch {epoch}: loss {running / len(ft_loader):.4f}  [{(time.time() - t0) / 60:.1f} min]')
ft_model.eval()
print(f'fine-tuned {FT_MODEL_NAME} in {(time.time() - t0) / 60:.1f} minutes')

# %%
base_small = AutoModelForSeq2SeqLM.from_pretrained(FT_MODEL_NAME).to(DEVICE).eval()
MODELS = [('FLAN-T5-small zero-shot', base_small, ft_tokenizer),
          ('FLAN-T5-small fine-tuned', ft_model, ft_tokenizer),
          ('FLAN-T5-base zero-shot', generator, gen_tokenizer)]

comparison = []
for cat in CLASS_NAMES:
    ref = reference_instruction(cat)
    for label, mdl, tk in MODELS:
        t0 = time.time()
        ans, used, fields = generate_structured(cat, model=mdl, tokenizer=tk)
        body = fields_body(fields)
        attrib = np.mean([rouge_l(a, section_text(cat, FIELD_SOURCE_SECTIONS[f])) >
                          rouge_l(a, section_text(cat, DECOY_SECTIONS[f]))
                          for f, a in fields])
        comparison.append({'model': label, 'category': cat, 'held out': cat in HELD_OUT,
                           'grounding': grounding_score(body, ' '.join(used['text'])),
                           'right section': attrib,
                           'ROUGE-L vs policy': rouge_l(ans, ref),
                           'generated words': len(body.split()), 'seconds': time.time() - t0,
                           'answer': ans})
    print(f'  {cat} done')
comp = pd.DataFrame(comparison)

summary_gen = comp.groupby('model')[['grounding', 'right section', 'ROUGE-L vs policy',
                                     'generated words', 'seconds']].mean()
held = comp[comp['held out']].groupby('model')[['grounding', 'ROUGE-L vs policy']].mean()
held.columns = [f'{c} (held-out streams)' for c in held.columns]
gen_results = summary_gen.join(held).round(3)
print(gen_results.to_string())

# %%
fig, ax = plt.subplots(figsize=(9.2, 3.4))
pos = np.arange(len(gen_results))
for i, (col, color, lbl) in enumerate([
        ('grounding', BLUE, 'grounding (words traceable to policy)'),
        ('right section', AQUA, 'answered from the right policy section'),
        ('ROUGE-L vs policy', ORANGE, 'ROUGE-L vs reference instruction')]):
    ax.barh(pos + (i - 1) * 0.26, gen_results[col], 0.25, color=color, label=lbl)
ax.set_yticks(pos); ax.set_yticklabels(gen_results.index, fontsize=9)
ax.invert_yaxis(); ax.grid(axis='y', visible=False); ax.tick_params(axis='y', length=0)
ax.set_xlim(0, 1.05)
ax.set_title('Three generators on the same structured task')
ax.legend(loc='lower right', fontsize=8)
plt.tight_layout(); plt.show()

print('Glass, three generators:\n')
for label in gen_results.index:
    row = comp[(comp['model'] == label) & (comp['category'] == 'Glass')].iloc[0]
    print(f'[{label}]  grounding {row["grounding"]:.2f} | right section '
          f'{row["right section"]:.2f}')
    wrap(row['answer'], width=92, indent='   ')
    print()

# The generator carried forward is the one that is both grounded and attributed: a high
# grounding score on its own only says the words came from *somewhere* in the context.
GEN_CHOICE = (gen_results['grounding'] * 0.4 + gen_results['right section'] * 0.4
              + gen_results['ROUGE-L vs policy'] * 0.2).idxmax()
print(f'generator wired into the assistant: {GEN_CHOICE}')
print(f'  grounding {gen_results.loc[GEN_CHOICE, "grounding"]:.3f} | '
      f'right section {gen_results.loc[GEN_CHOICE, "right section"]:.3f} | '
      f'{gen_results.loc[GEN_CHOICE, "seconds"]:.1f}s per answer')

# %% [markdown]
# ### 4.5 The instruction-generation function
#
# Two safeguards wrap the model, because a plausible-sounding wrong instruction is worse than
# no instruction:
#
# 1. **Grounding check.** The answer's content words are compared against the retrieved policy.
#    Below `MIN_GROUNDING` the generated text is discarded and an **extractive fallback** —
#    the relevant policy sentences, quoted verbatim — is returned instead. The system degrades
#    to a correct quotation rather than to a confident invention.
# 2. **Citations, always.** Every answer returns the policy documents, sections and effective
#    dates it was built from, so a resident (or a council officer) can check it.

# %%
MIN_GROUNDING = 0.55


def extractive_fallback(chunks, category):
    """Quote the policy directly - used when the generated text is not sufficiently grounded."""
    order = {'Collection Method': 0, 'Preparation Instructions': 1, 'Acceptable Items': 2}
    rows = chunks.assign(rank=chunks['section'].map(lambda s: order.get(s, 3))).sort_values('rank')
    lines = [f"{category}: follow Metro City's {rows.iloc[0]['source']}."]
    for r in rows.head(3).itertuples():
        body = r.text.split(': ', 1)[-1].replace('\n- ', '; ').replace('\n', ' ')
        body = re.sub(r'\s+', ' ', body).strip()
        lines.append(f'- {r.section}: {body}')
    return '\n'.join(lines)


def generate_recycling_instructions(waste_category, question=None, k=3, verbose=False):
    """Generate grounded recycling instructions for a waste stream.

    Args:
        waste_category (str): one of the nine Metro City streams.
        question (str): optional resident wording; defaults to "How do I recycle {category}?".
        k (int): number of policy chunks to retrieve.

    Returns:
        (str, list): the instruction text, and the policy documents it is grounded in - each a
        dict with source, section, effective date, similarity and the quoted text.
    """
    if not isinstance(waste_category, str) or waste_category not in CLASS_NAMES:
        return (f'Unknown waste stream {waste_category!r}. Metro City collects: '
                f'{", ".join(CLASS_NAMES)}.', [])

    model, tk = ((ft_model, ft_tokenizer) if 'fine-tuned' in GEN_CHOICE
                 else (generator, gen_tokenizer) if 'base' in GEN_CHOICE
                 else (base_small, ft_tokenizer))

    if question:
        # a specific resident question: answer that, rather than the three standard fields
        chunks = retrieve(question, k=k, category=waste_category)
        answer = generate_with(model, tk,
                               FIELD_INSTRUCTION.format(context=format_context(chunks),
                                                        question=question),
                               max_new_tokens=70, **DECODE_KWARGS)
        body = answer
    else:
        answer, chunks, fields = generate_structured(waste_category, model=model,
                                                     tokenizer=tk, k=k)
        body = fields_body(fields)

    ctx_text = ' '.join(chunks['text'])
    score = grounding_score(body, ctx_text)

    if score < MIN_GROUNDING or len(answer.split()) < 8:
        answer = extractive_fallback(chunks, waste_category)
        score = grounding_score(answer, ctx_text)
        if verbose:
            print(f'  [grounding {score:.2f} below {MIN_GROUNDING}: fell back to quoted policy]')

    citations = [{'source': r.source, 'section': r.section, 'effective_date': r.effective_date,
                  'similarity': round(float(r.score), 3), 'text': r.text}
                 for r in chunks.itertuples()]
    return answer, citations


for cat in ['Glass', 'Food Organics', 'Miscellaneous Trash']:
    text, cites = generate_recycling_instructions(cat, verbose=True)
    print(f'=== {cat} ===')
    wrap(text, width=92, indent='  ')
    print('  sources: ' + '; '.join(f'{c["source"]} / {c["section"]}' for c in cites))
    print()

# %%
# End-to-end check over all nine streams: is every answer grounded, cited, and on-topic?
rag_rows = []
for cat in CLASS_NAMES:
    t0 = time.time()
    text, cites = generate_recycling_instructions(cat)
    ctx = ' '.join(c['text'] for c in cites)
    rag_rows.append({'stream': cat, 'words': len(text.split()),
                     'grounding': grounding_score(text, ctx),
                     'ROUGE-L vs policy': rouge_l(text, reference_instruction(cat)),
                     'citations on stream': float(np.mean(
                         [cat in corpus.loc[corpus['text'] == c['text'], 'categories'].iloc[0]
                          for c in cites])),
                     'seconds': time.time() - t0})
rag_eval = pd.DataFrame(rag_rows).set_index('stream')
print(rag_eval.round(3).to_string())
print(f'\nmean grounding {rag_eval["grounding"].mean():.3f} | '
      f'mean on-stream citations {rag_eval["citations on stream"].mean():.1%} | '
      f'mean latency {rag_eval["seconds"].mean():.1f}s')

# %% [markdown]
# ### Part 4 takeaways
#
# **Retrieval is the reliable half**, for a boring reason: a few dozen well-structured chunks
# over a nine-way topic space is an easy search problem, and the stream label coming out of
# Parts 2 and 3 makes it easier still. Most of the engineering here went not into the search
# but into *which* chunk reaches the generator — the stream filter, the section hint, and the
# hard block on the accepted-items list when the question is about exclusions. Each of those
# was added in response to a specific wrong answer, and the "right section" column is what
# shows them working.
#
# **Generation is the weak half, exactly as the lab predicts.** Zero-shot FLAN-T5-base answers
# in fragments ("Bin: Bins"), and when the sampling temperature is raised it invents collection
# schemes Metro City does not operate. Fine-tuning the 80M model on policy sections fixes both
# problems at once and beats the three-times-larger zero-shot model on every metric here — and
# it holds up on the two streams held out of training, which says it learned the behaviour
# rather than the nine answers.
#
# It is worth being precise about what the fine-tuned model learned: **to quote, not to
# write.** Its ROUGE-L against policy is near 1.0 because it reproduces the relevant section
# almost verbatim. For this product that is the desired behaviour — a resident acting on a
# disposal instruction needs Metro City's rule, not a paraphrase of it — but those numbers
# should not be read as fluency. A larger generator would write more naturally; it would also
# need more guarding, not less.
#
# **The design does not depend on the generator being right.** Grounding is measured rather
# than assumed, the answer is replaced by quoted policy whenever the measurement fails, and
# citations with effective dates are attached either way. The worst case is a verbatim
# quotation of the correct document, not a fluent fabrication — the right failure mode for
# something a resident will act on.
