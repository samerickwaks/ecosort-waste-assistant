# EcoSort Waste Management Assistant

Moringa School - Module 8 Summative Lab: CNNs, transformers and retrieval-augmented generation.

An end-to-end waste-sorting assistant that takes a photo of an item **or** a resident's
written description, classifies the material, and generates grounded disposal instructions
from Metro City's recycling policy documents.

## What's in here

| Path | What it is |
| --- | --- |
| [waste_management_summative.ipynb](waste_management_summative.ipynb) | The deliverable notebook — all five parts, executed with outputs |
| [build/](build/) | Percent-format sources the notebook is generated from (see below) |
| [waste_descriptions.csv](waste_descriptions.csv) | Resident waste descriptions with category labels |
| [waste_policy_documents.json](waste_policy_documents.json) | Metro City policy corpus used as the RAG knowledge base |
| [DATASET.md](DATASET.md) | How to obtain the RealWaste image set (not committed — 668 MB) |

## The pipeline

**Part 1 — Data.** Explores three sources: the RealWaste image set (9 classes, 4,752 images),
the resident descriptions, and the policy corpus. Builds the splits and `tf.data` pipelines.

**Part 2 — Image classification.** Compares MobileNetV2 and EfficientNetB0 with a frozen-feature
probe, sizes the classification head on cached features, then trains in two stages: 5 epochs with
the backbone frozen, then 6 epochs fine-tuning the top of the backbone.

**Part 3 — Text classification.** Sparse linear and tree baselines on TF-IDF features versus a
fine-tuned DistilBERT (3 epochs, LR 3e-5). The better of the two is selected for the assistant.

**Part 4 — RAG.** Embeds the policy corpus with `all-MiniLM-L6-v2`, checks the retriever actually
retrieves, then conditions FLAN-T5 on the retrieved passage. Includes a fine-tuning pass on
grounded instruction pairs.

**Part 5 — Integration.** Wires the three components into one assistant that routes image or text
input to the right classifier and returns retrieved-and-grounded disposal instructions.

## Running it

The notebook is generated from the sources in [build/](build/) — **edit those, not the `.ipynb`**:

```bash
python build/make_notebook.py      # assemble s0..s5 into the notebook
python build/execute_notebook.py   # run it end to end
```

Before the first run:

1. Download the RealWaste images into `./RealWaste/` — see [DATASET.md](DATASET.md).
2. Install the dependencies:

```bash
pip install tensorflow torch transformers sentence-transformers \
            scikit-learn pandas numpy matplotlib pillow
```

Trained weights (`cnn_stage1.weights.h5`, `cnn_stage2.weights.h5`) are not committed — Part 2
regenerates them. Seeded with `SEED = 42` throughout, though CNN training is not bit-reproducible
across machines.

## Attribution

The RealWaste dataset is CC BY-NC-SA 4.0 and is **not redistributed** in this repository.
If you use it, cite [RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification
Using Deep Learning](https://www.mdpi.com/2078-2489/14/12/633).
