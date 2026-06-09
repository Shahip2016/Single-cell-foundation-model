# 🧬 The Layman's Guide to CellFM

> *Everything you need to understand this project, even if you've never done biology or AI before.*

---

## Part 1: The Biology — What Are Cells and Genes?

### Your Body Is Made of Trillions of Tiny Cells

Your body contains roughly **37 trillion cells**. Each cell is like a tiny, self-contained city:

- 🏭 **Factories** (ribosomes) that build proteins
- 📚 **A library** (nucleus) containing the complete instruction manual (DNA)
- 🔋 **Power plants** (mitochondria) that generate energy
- 🚛 **Transportation networks** that move things around

### Genes Are the Instructions

Inside every cell's library (nucleus), there's a complete copy of your **DNA** — about 20,000 instruction pages called **genes**. Each gene contains instructions for building one specific protein (a molecular machine).

**The key insight**: Even though *every cell has the same DNA*, different cell types "read" different pages. A brain cell reads the "make neurotransmitters" pages intensely, while a blood cell reads the "carry oxygen" pages instead.

### Gene Expression = How Loudly Each Gene Is Being Read

**Gene expression** is a number that tells us how actively a cell is reading each gene:
- `Gene_BRCA1 = 0` → This gene is silent (not being read)
- `Gene_TP53 = 150` → This gene is moderately active
- `Gene_HBB = 5000` → This gene is very active (common in red blood cells)

### Single-Cell RNA Sequencing (scRNA-seq)

Traditional biology measured gene expression for millions of cells mixed together (like averaging the temperature of an entire building). **scRNA-seq** measures each cell individually (like putting a thermometer in every room).

This produces a **matrix** where:
- Each **row** is one cell
- Each **column** is one gene
- Each **value** is the expression level

```
          Gene1  Gene2  Gene3  Gene4  ...  Gene20000
Cell_1  [  0.0    3.2    0.0    1.5   ...    0.0   ]
Cell_2  [  2.1    0.0    5.7    0.0   ...    0.3   ]
Cell_3  [  0.0    0.0    0.0    8.4   ...    0.0   ]
```

**Notice**: Most values are 0! This is called **sparsity** — a typical cell only actively reads ~2,000-4,000 of its 20,000 genes.

---

## Part 2: The AI — What Is a Foundation Model?

### Foundation Models Are Like Universal Students

A **foundation model** is an AI trained on massive amounts of data to learn general patterns. Think of it as a student who:

1. **Pre-training** = Reads millions of books (expensive, done once)
2. **Fine-tuning** = Takes a specialized course for a specific job (cheap, done many times)

**Famous examples**:
| Domain | Foundation Model | Pre-trained On |
|--------|-----------------|----------------|
| Language | GPT-4, Claude | Trillions of words |
| Images | DALL-E, Stable Diffusion | Billions of images |
| **Cells** | **CellFM** | **100 million cells** |

### How Does CellFM Learn?

CellFM uses **masked pretraining** — the same idea as fill-in-the-blank:

```
Original cell:     Gene1=3.2  Gene2=0.0  Gene3=5.7  Gene4=1.5
Masked version:    Gene1=3.2  Gene2=???  Gene3=5.7  Gene4=???

Task: "Predict the masked values!"
```

By doing this billions of times across 100 million cells, CellFM learns:
- Which genes tend to be active together
- What makes a brain cell different from a blood cell
- The "grammar" of gene expression

---

## Part 3: The Architecture — How CellFM Works

### Step 1: Embedding (Translating Numbers to Vectors)

Raw gene expression values (like `3.2`) are just numbers. The **embedding module** converts them into rich vector representations that the AI can process.

Think of it like translating a word into its full meaning:
- Word "bank" → `[financial institution, river edge, to rely on, ...]`
- Gene expression `3.2 for TP53` → `[512-dimensional vector capturing the meaning]`

### Step 2: RetNet Backbone (The Brain)

**RetNet** (Retentive Network) is the core "thinking" architecture. It's similar to the Transformer used in ChatGPT, but with a clever trick:

| Feature | Transformer | RetNet |
|---------|------------|--------|
| How it processes | Looks at all pairs of elements (expensive) | Uses "retention" with exponential decay (cheaper) |
| Complexity | O(n²) — quadratic | O(n) — linear |
| Memory | Needs to store all pair interactions | Summarizes history efficiently |

**Why it matters**: A cell has ~2,000+ active genes. With a Transformer, that's 2000² = 4,000,000 pair comparisons. RetNet makes this much more efficient.

### Step 3: SGLU (Smart Filter)

After the retention layer processes information, the **SGLU** (Simple Gated Linear Unit with SiLU activation) acts as a smart filter:

```
Input → Two parallel paths:
  Path 1: Learn "what's important" (gate)
  Path 2: Learn "the actual content"
  
Output = Path1 × Path2  (only important content gets through)
```

### Step 4: LoRA Fine-tuning (Efficient Adaptation)

Training the full 800M parameter model from scratch is like rebuilding a house for every new tenant. **LoRA** (Low-Rank Adaptation) is like adding removable furniture:

```
Original weight matrix W (huge, frozen):
    ┌─────────────────┐
    │  800M parameters │  ← DON'T touch these
    │  (pre-trained)   │
    └─────────────────┘

LoRA adapter (small, trainable):
    ┌──────┐   ┌──────┐
    │ A    │ × │ B    │  ← Only train THESE
    │ (small)  │ (small)  │     (~1% of parameters)
    └──────┘   └──────┘
    
Final output = W·x + A·B·x
```

---

## Part 4: Key Concepts Glossary

| Term | Plain English |
|------|--------------|
| **scRNA-seq** | Technology to measure gene activity in individual cells |
| **Gene expression** | How actively a cell is reading a particular gene |
| **Embedding** | Converting raw numbers into meaningful vector representations |
| **Attention/Retention** | The AI's ability to figure out which genes relate to each other |
| **Masking** | Hiding some genes and training the AI to predict them |
| **Foundation model** | An AI trained on massive data that can be adapted for many tasks |
| **Fine-tuning** | Adapting a pre-trained model for a specific task |
| **LoRA** | A trick to fine-tune efficiently by only training small adapter matrices |
| **Batch effects** | Technical noise from different lab setups that obscures real biology |
| **h5ad/AnnData** | The standard file format for single-cell data |

---

## Part 5: Running the Code

### What You Need
- **Python 3.9+** — The programming language
- **PyTorch** — The deep learning framework
- **Scanpy** — The single-cell analysis toolkit
- A **GPU** with at least 8GB memory (for the 80M model) or 32GB (for 800M)

### First Steps
1. Follow the installation in the README
2. Start with the tutorial notebooks in `notebooks/`
3. Try cell annotation on a small dataset first

---

*Questions? Open an issue on GitHub or check the tutorial notebooks!*
