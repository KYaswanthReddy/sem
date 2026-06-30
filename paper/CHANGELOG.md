# Changelog - SG-PMG Manuscript Refinement Pass

This document details all structural, mathematical, and formatting improvements applied during the second pass refinement.

## Modifications and Refinements

### 1. Document Structure & Package Settings (`paper/main.tex`)
- Kept the standard class configuration (`IEEEtran` under journal template) and layout flow.
- Verified package imports for math rendering (`amsmath`, `amssymb`), algorithm compilation, and table design (`booktabs`).

### 2. Typographical & Compilation Fixes (`paper/introduction.tex`)
- **Fixed Compile-Breaking Command**: Replaced the non-standard `\IEEEinitialdropchapter` command with the standard IEEEtran drop-cap command `\IEEEPARstart`.
- Added verified bibliographic citations referencing `\cite{he2017deep}`, `\cite{wang2022domain}`, `\cite{li2022progressive}`, `\cite{radford2021learning}`, and `\cite{foret2020sharpness}`.

### 3. Bibliography Formatting & Baseline Expansion (`paper/references.bib`)
- **Fixed BibTeX Syntax Error**: Corrected the nested assignment error in `author={Wang={{Wang, Jindong...}}}` to standard BibTeX format.
- Added citation records for the complete set of baselines compared in the experimental section (SAGM, SDENet, LLURNet, S2AMSNet, ACB, FDGNet, EHSNet, DTAM).

### 4. Mathematical Equation Formatting (`paper/methodology.tex` & `paper/mathematical_formulation.tex` & `paper/appendix.tex`)
- **Standardized Display Math**: Converted all display math representations (originally using raw TeX double dollar `$$ ... $$` signs) into standard numbered and labeled LaTeX `\begin{equation} ... \end{equation}` environments.
- Ensured proper referencing labels (`\label{eq:encoder}`, `\label{eq:attn}`, `\label{eq:l_wd}`, etc.) to support automated cross-referencing using `\ref` and `\eqref`.
- Verified and aligned equations to correspond exactly to visual-semantic multi-head cross-attention outputs, conditional AdaLN3d modulation operations, and contrastive optimization losses.

### 5. Algorithmic Optimization Flow (`paper/algorithm.tex`)
- Refined comments inside `algorithmic` loop statements to use standard block format `\STATE \textbf{STAGE k: ...}` rather than inline comments to make them stand out.
- Ensured stage boundaries and data layout descriptions match PyTorch training parameters.

### 6. Results and Ablation Tables (`paper/tables.tex` & `paper/figures.tex`)
- Added clear `%% TODO:` comments indicating which values/plots are baseline placeholders to be filled with final experimental logs once actual runs are completed.
- Expanded the figures list in `figures.tex` to include parameters sensitivity surface, labeled sample scaling curve, and average prediction confidence progression plots.

### 7. Section Refinements & Flow (`paper/discussion.tex` & `paper/related_work.tex`)
- Rewrote the discussion section to explain the specific generated visualizations (t-SNE alignments, 3D parameter sensitivity surface grid searches, sample size scaling curves, and sample classification confidence distributions).
- Removed repeated sentences and generic AI-written templates to maintain publication-ready style suitable for IEEE TGRS or Remote Sensing.
