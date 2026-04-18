# NeurIPS-Style LaTeX Template

This is the preferred report template for the project. It uses the cleaner
NeurIPS 2024 style in `preprint` mode by default because the project prompt does
not require a specific NeurIPS year.

The official NeurIPS 2026 template is still kept in `../neurips_2026/` in case a
strict 2026 submission format is required later.

## Files

- `main.tex`: starter paper.
- `references.tex`: bibliography section included by `main.tex`.
- `references.bib`: BibTeX references used by `main.tex`.
- `appendix.tex`: appendix sections included by `main.tex`.
- `checklist.tex`: optional NeurIPS 2024 paper checklist.
- `neurips_2024.sty`: official NeurIPS 2024 style file.
- `neurips_2024.tex`: official formatting-instructions example.

## Build

With `latexmk`:

```powershell
latexmk -pdf main.tex
```

Without `latexmk`:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Template Modes

In `main.tex`, change the package options when needed:

```tex
\usepackage[preprint]{neurips_2024} % report/preprint, no line numbers
\usepackage{neurips_2024}           % anonymous submission
\usepackage[final]{neurips_2024}    % camera-ready
```

Official source:
https://neurips.cc/Conferences/2024/CallForPapers
