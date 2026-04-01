---
name: style
description: Applies TMLR (Transactions on Machine Learning Research) formatting rules to a LaTeX document.
---

# style

Instructions for the AI agent to format a LaTeX document according to TMLR journal submission guidelines.

## Usage

Use this skill when a user asks to format a LaTeX document, apply TMLR guidelines, or style a paper for TMLR submission.

## Steps

1. **Document Class and Packages:**
   - Ensure the document uses `\documentclass[10pt]{article}`.
   - Include the `tmlr` package (`\usepackage{tmlr}`). Use `[accepted]` or `[preprint]` options if specified by the user.

2. **General Formatting:**
   - Use 10-point type with 11-point vertical spacing.
   - The preferred typeface is Computer Modern Bright.
   - Verify paragraphs are separated by 1/2 line space, with no indentation.
   - Text must be confined within a rectangle 6.5 inches wide and 9 inches long. The left margin is 1 inch. All pages should start at 1 inch from the top.

3. **Title and Authors:**
   - Paper title must be 17-point, bold, and left-aligned.
   - Author names must be boldface, placed above the corresponding address. Emails should be on the same line as the name, italicized, and right-aligned. The lead author's name is listed first, and co-authors follow vertically.

4. **Abstract:**
   - Must be indented 1/2 inch on both left and right margins.
   - The word **Abstract** must be centered, bold, and 12-point. Two line spaces precede the abstract.
   - Limited to exactly one paragraph.

5. **Headings:**
   - **First level:** Bold, flush left, 12-point. One line space before, 1/2 line space after.
   - **Second level:** Bold, flush left, 10-point. One line space before, 1/2 line space after.
   - **Third level:** Bold, flush left, 10-point. One line space before, 1/2 line space after.

6. **Citations and References:**
   - Use the `natbib` package.
   - Use `\citet{}` when authors are part of the sentence (e.g., "See \citet{Hinton06} for more information.").
   - Use `\citep{}` for parenthetical citations (e.g., "...towards AI \citep{Bengio+chapter2007}.").
   - References must be in alphabetical order in the References section.

7. **Footnotes:**
   - Place at the bottom of the page on which they appear.
   - Precede the footnote with a horizontal rule of 2 inches.

8. **Figures:**
   - Figure number and caption appear _after_ the figure.
   - One line space before the figure caption, one line space after the figure.
   - The figure caption is lower case (except for the first word and proper nouns).
   - Ensure color figures make sense if printed in black and white.

9. **Tables:**
   - All tables must be centered. Do not use hand-drawn tables.
   - Table number and title appear _before_ the table.
   - One line space before the table title, one line space after the table title, one line space after the table.
   - The table title must be lower case (except for the first word and proper nouns).

10. **Math/Notation:**
    - Optionally use `math_commands.tex` from `\input{math_commands.tex}` for standardized notation (from the textbook _Deep Learning_).

11. **Broader Impact / Acknowledgments:**
    - Use unnumbered third-level headings (`\subsubsection*{...}`).
    - Acknowledgments go at the end of the paper.
    - Only add these sections once the submission is accepted and deanonymized.
