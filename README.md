# Logic Tensor Networks: presentation and hands-on case study

This repository presents **Logic Tensor Networks (LTN)**, a neuro-symbolic
framework that combines deep learning with first-order logical reasoning:
it fixes the *form* of a predicate through a logical formula, while the
*content* it computes is learned by gradient descent.

The presentation and the notebooks are written in French.

The repository has two parts:

1. **A theoretical presentation** (`presentation/`): how a logical symbol
   (constant, predicate, function, variable) is represented as a tensor
   (*grounding*), how logical connectives and quantifiers are turned into
   differentiable operations (stable product configuration, diagonal and
   guarded quantification), and how the satisfaction of a knowledge base
   becomes a loss function for training a neural network. See the
   [tutorials](code/tutorials).
2. **A hands-on case study** (`code/`), built on the
   [LTNtorch](https://github.com/logictensornetworks/LTNtorch) library:
   semi-supervised recognition of handwritten digits, where an LTN learns
   to recognize individual digits without ever receiving their label, only
   the sum of two digits. The results are compared with a purely supervised
   baseline. See the [examples](code/examples), in particular
   [the case study](code/examples/4-semi-supervised_pattern_recognition.ipynb).

## Repository structure

```
logic-tensor-networks/
├── code/
│   ├── venv/                          Python virtual environment (not versioned)
│   ├── requirements.txt               Python dependencies
│   ├── diagrams/                      script drawing the connective and quantifier diagrams
│   ├── tutorials/                     my notes on the LTNtorch tutorials
│   │   ├── 1-grounding_non_logical_symbols.ipynb
│   │   ├── 2-grounding_connectives.ipynb
│   │   ├── 2b-operators-and-gradients.ipynb
│   │   └── 3-knowledgebase-and-learning.ipynb
│   └── examples/                      official LTNtorch examples
│       ├── 1-binary_classification.ipynb
│       ├── 2-multi_class_single_label_classification.ipynb
│       ├── 3-multi_class_multi_label_classification.ipynb
│       ├── 4-semi-supervised_pattern_recognition.ipynb   <- case study of the presentation
│       ├── 5-regression.ipynb
│       ├── 6-clustering.ipynb
│       ├── 7-learning_embeddings_with_LTN.ipynb
│       ├── datasets/
│       └── images/
│
└── presentation/
    ├── main.tex           presentation source (Beamer)
    ├── main.pdf           compiled presentation
    └── images/            figures used in the slides
```

### Tutorials (`code/tutorials/`)

My notes on the LTNtorch tutorials: my own explanation of each step, to be
read in order. Each one introduces a further layer of the framework:

1. **Grounding non-logical symbols.** Real Logic: how a constant, a
   predicate, a function or a variable becomes a tensor or a
   differentiable function.
2. **Grounding connectives.** Turning logical connectives (∧, ∨, ¬, ⇒)
   into differentiable fuzzy operations (product configuration).
3. **Operators and gradients.** The three gradient pitfalls (vanishing,
   single-passing, exploding) and the *stable* product configuration that
   avoids them.
4. **Knowledge base and learning.** Quantifiers (∀, ∃), generalized mean
   (`pMean`), and the satisfaction of a knowledge base (`SatAgg`) as the
   training objective.

### Examples (`code/examples/`)

This folder keeps all the official LTNtorch examples for reference.
Example **4, semi-supervised pattern recognition**, is the one developed in
the presentation: an LTN learns an MNIST digit classifier without ever
receiving the label of an individual digit, only the sum of two digits, and
its ability to generalize is compared with a purely supervised baseline on
single-digit and two-digit addition.

### Diagrams (`code/diagrams/`)

`connectives_and_quantifiers.py` draws four diagrams of how connectives and
quantifiers act on tensor shapes, in the graphical convention of the LTN
paper, for the examples of tutorial 2: a conjunction on a shared variable,
a disjunction with broadcasting, a negation on constants, and a universal
quantifier over a single axis. From `code/`:

```bash
python diagrams/connectives_and_quantifiers.py
```

The images, in French and English (`-en` suffix), are written to
`diagrams/images/`, which is not versioned. `--out` sets another folder and
`--lang` draws a single language.

## Getting started

### Code environment

**Windows (PowerShell)**
```powershell
cd code
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux / macOS**
```bash
cd code
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then start Jupyter from `code/`:
```bash
jupyter notebook
```

and open `examples/4-semi-supervised_pattern_recognition.ipynb` to
reproduce the case study.

### Building the presentation

Requires a LaTeX distribution with `latexmk` and `pdflatex`, plus Python
and [Pygments](https://pygments.org/) (needed by `minted` for syntax
highlighting, through `-shell-escape`).

From `presentation/`:

```bash
latexmk -pdf main.tex
```

## References

- Badreddine, S., d'Avila Garcez, A., Serafini, L., & Spranger, M. (2022).
  *Logic Tensor Networks*. Artificial Intelligence, 303, 103649.
- Manhaeve, R., Dumančić, S., Kimmig, A., Demeester, T., & De Raedt, L.
  (2018). *DeepProbLog: Neural Probabilistic Logic Programming*. NeurIPS.
- Carraro, T. LTNtorch:
  [github.com/logictensornetworks/LTNtorch](https://github.com/logictensornetworks/LTNtorch)
- LeCun, Y., Cortes, C., & Burges, C. *The MNIST Database of Handwritten
  Digits*.
