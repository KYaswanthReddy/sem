# IEEE LaTeX Project: Few-Shot HSI Domain Generalization

This directory contains the complete LaTeX source code for the research paper describing the **Semantic-Guided Progressive Multi-stage Generator (SG-PMG)** framework.

## Project Structure
- `main.tex`: The main document entry point.
- `abstract.tex`: Abstract.
- `introduction.tex`: Section I.
- `related_work.tex`: Section II.
- `methodology.tex`: Section III.
- `algorithm.tex`: Section IV.
- `experiments.tex`: Section V.
- `results.tex`: Section VI.
- `discussion.tex`: Section VII.
- `conclusion.tex`: Section VIII.
- `references.bib`: BibTeX citations database.
- `macros.tex`: Abbreviations macros.
- `tables.tex`: Table environments (included in results).
- `appendix.tex`: Appendix section.

## Compilation Instructions

To compile the LaTeX source code into a PDF:

### Option 1: Local pdflatex / latexmk
Run the following commands in this directory:
```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```
Or, if you have `latexmk` installed:
```bash
latexmk -pdf main.tex
```

### Option 2: Overleaf
1. Compress all files in this directory into a `.zip` archive.
2. Upload the archive to Overleaf and compile using the `pdfLaTeX` engine.
