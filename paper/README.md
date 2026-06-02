# Paper

*Training-Free Lexical–Dense Fusion for Conversational-Memory Retrieval*
— Christian Lysenstøen, 2026.

- `paper.pdf` — compiled paper (9 pages).
- `paper.tex` — LaTeX source (top-level file).
- `references.bib`, `paper.bbl` — bibliography.
- `fig_interaction.pdf`, `fig_category.pdf`, `fig_alpha.pdf` — figures.

## Build

```bash
pdflatex paper
bibtex   paper
pdflatex paper
pdflatex paper
```

Requires a standard TeX distribution (TeX Live / MiKTeX) with `natbib`,
`lmodern`, `booktabs`, `hyperref`, `graphicx`, `amsmath`, `microtype`,
`caption`. License: CC BY 4.0.
