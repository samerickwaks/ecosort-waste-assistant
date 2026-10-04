# ---- DEV stub replacing Parts 1-3 cheaply so Parts 4-5 can be iterated on ----
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

DEVICE = 'cpu'
DATA_DIR = pathlib.Path('RealWaste')
IMG_HEIGHT = IMG_WIDTH = 224
CLASS_NAMES = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())
NUM_CLASSES = len(CLASS_NAMES)
SHORT = {c: c.replace('Miscellaneous', 'Misc.').replace(' Organics', ' Org.').replace(' Trash', ' Tr.')
         for c in CLASS_NAMES}
BACKBONE_NAME, TEXT_MODEL = 'EfficientNetB0', 'bert'
CNN_CONF_THRESHOLD, TEXT_CONF_THRESHOLD = 0.60, 0.50

desc = pd.read_csv('waste_descriptions.csv')
with open('waste_policy_documents.json', encoding='utf-8') as fh:
    policies = json.load(fh)
pol = pd.DataFrame(policies)
SECTION_RE = re.compile(r'^([A-Z][A-Za-z /-]+):\s*$', re.M)


def flatten(text):
    out = text.replace('\n- ', '; ').replace('\n', ' ')
    return re.sub(r'\s+', ' ', out).strip().lstrip('- ')


def split_sections(text):
    marks = [(m.start(), m.group(1)) for m in SECTION_RE.finditer(text)]
    out = []
    for i, (pos, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        body = text[text.index('\n', pos) + 1:end].strip()
        if body:
            out.append((name, body))
    return out


def section_categories(heading, doc_categories):
    for c in CLASS_NAMES:
        if heading.upper().startswith(c.upper()):
            return [c]
    return list(doc_categories)


chunks = []
for p in policies:
    title = p['document_text'].split('\n')[0].strip()
    for name, body in split_sections(p['document_text']):
        chunks.append({'chunk_id': f"P{p['policy_id']}-{name.replace(' ', '')[:18]}",
                       'source': f"Policy {p['policy_id']}: {p['policy_type']}",
                       'section': name, 'categories': section_categories(name, p['categories_covered']),
                       'effective_date': p['effective_date'],
                       'text': f'{title}. {name}: {body}', 'body': flatten(body)})
for cat, grp in desc.groupby('category'):
    tips = sorted(grp['disposal_instruction'].dropna().unique())
    conf = sorted(grp['common_confusion'].dropna().unique())
    chunks.append({'chunk_id': f'D-{cat.replace(" ", "")}', 'source': f'{cat} disposal card',
                   'section': 'Disposal Instructions', 'categories': [cat], 'effective_date': None,
                   'text': f'{cat} disposal instructions: ' + ' '.join(tips)
                           + (' Common confusion: ' + ' '.join(conf) if conf else ''),
                   'body': flatten(' '.join(tips)
                                   + (' Common confusion: ' + ' '.join(conf) if conf else ''))})
corpus = pd.DataFrame(chunks)
corpus['n_words'] = corpus['text'].str.split().str.len()

BRAND_RE = re.compile(r'\b(brand [a-z]|premium|budget|designer|eco-friendly|supermarket|'
                      r'generic|store|value|luxury)\b', re.I)


def clean_description(text):
    t = BRAND_RE.sub(' ', str(text).lower())
    t = re.sub(r'[^a-z\s-]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


desc['clean'] = desc['description'].map(clean_description)
txt_test = desc.head(40).assign(has_exclusive_token=True)[
    ['clean', 'description', 'category', 'has_exclusive_token']].reset_index(drop=True)
images = pd.DataFrame([(f, f.parent.name, f.name) for f in sorted(DATA_DIR.glob('*/*.jpg'))],
                      columns=['path', 'label', 'filename'])
images['label_idx'] = images['label'].map({c: i for i, c in enumerate(CLASS_NAMES)})
img_test = images.groupby('label', group_keys=False).head(6).reset_index(drop=True)

tf.keras.utils.set_random_seed(SEED)
model = tf.keras.Sequential([
    layers.Input((IMG_HEIGHT, IMG_WIDTH, 3)), layers.Rescaling(1 / 127.5, offset=-1),
    layers.Conv2D(8, 3, strides=4, activation='relu'), layers.GlobalAveragePooling2D(),
    layers.Dense(NUM_CLASSES, activation='softmax')], name='stub_cnn')


def classify_waste_description(description, top_k=3):
    if not isinstance(description, str) or not description.strip():
        return {'category': None, 'confidence': 0.0, 'top_k': [], 'model': 'stub',
                'cleaned': '', 'error': 'empty or non-string description'}
    cleaned = clean_description(description)
    if not cleaned:
        return {'category': None, 'confidence': 0.0, 'top_k': [], 'model': 'stub',
                'cleaned': '', 'error': 'nothing left after cleaning'}
    rng = np.random.default_rng(abs(hash(cleaned)) % 2 ** 31)
    proba = rng.dirichlet(np.ones(NUM_CLASSES) * 0.4)
    order = np.argsort(-proba)[:top_k]
    return {'category': CLASS_NAMES[int(order[0])], 'confidence': float(proba[order[0]]),
            'top_k': [(CLASS_NAMES[int(i)], float(proba[i])) for i in order], 'model': 'stub',
            'cleaned': cleaned, 'low_confidence': bool(proba[order[0]] < TEXT_CONF_THRESHOLD)}


print(f'DEV stub: {len(corpus)} chunks, {len(img_test)} test images, {len(txt_test)} test texts')
