# Quantitative Finance submission copy

`dy21.tex` is the publication-preparation source for the current manuscript.

The shared literature tables and bibliography live in
`quantdata/tex/models.tex` and `quantdata/tex/references.bib`. The manuscript
references those files by their repository-relative paths.

The copy follows the current submission requirements available for
*Quantitative Finance*: A4 paper with 2.5 cm margins, an unstructured abstract
below 200 words, six keywords, JEL codes, author--date citations, references
before appendices, and separately supplied figure files. The generated PDF is
within the journal's 35-page limit.

Build with:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error dy21.tex
```

Before submission, the authors must confirm or provide:

- final affiliations and corresponding-author details;
- funding and competing-interest statements;
- CRediT author roles in the submission system;
- the required generative-AI disclosure, including tool name and version;
- whether the conference-presentation footnote needs any additional prior-
  publication disclosure;
- higher-resolution originals for `SPX2020-2025.png`, `fit-short.png`, and
  `fit-long.png`. At their current print widths they are below the publisher's
  nominal 300-dpi target; the other plots are vector or exceed 300 dpi.

Submit `dy21.pdf` as the manuscript PDF and the source archive as the LaTeX
source file. The proprietary OptionMetrics data are not included.
