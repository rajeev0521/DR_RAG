# STP-RAG Overleaf package

This package contains a single, self-contained LaTeX manuscript and its BibLaTeX bibliography.

## Included files

- `STP_RAG_IEEE_Paper.tex` — main IEEE two-column manuscript
- `references.bib` — APA-style BibLaTeX entries
- `README.md` — these build notes

## Build in Overleaf

1. Upload every file in this folder, or upload `STP_RAG_Overleaf_Package.zip`.
2. Set `STP_RAG_IEEE_Paper.tex` as the main document.
3. In **Menu → Settings**, select **Biber** as the bibliography tool.
4. Compile. Overleaf will run the required LaTeX/Biber passes automatically.

The document deliberately combines an IEEE two-column *page layout* with APA author-date citations and an APA reference list. It does **not** use IEEE numbered citations.

## Before submission

- Replace the placeholder author and institution details.
- Run the stated experiments and replace all `TBD` cells in the results table with real measurements.
- Check every reference against the venue's final author guidelines.
- Do not claim novelty or experimental improvements that are not supported by the completed evaluation.
