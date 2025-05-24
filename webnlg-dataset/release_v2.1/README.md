The WebNLG_version2.1 dataset contains 16,095 data inputs and 42,873 data-text pairs.

Changes compared to version2:

Typos were corrected; missing verbalisations for triples were added.
In total, 5,667 texts were modified. 19 texts were deleted, since, after correction, they matched other texts for the same data input.

More on changes:
* correct typos spotted by Abelardo Vieira Mota ([here](https://github.com/abevieiramota/challenge-webnlg/blob/master/notebook/14%20-%20Searching%20misspellings%20in%20references.ipynb)) and Amit Moryossef ([here](https://github.com/AmitMY/chimera/blob/d44eff6d09b89b9c6cd3f357c22ab94b92df25ae/data/WebNLG/reader.py#L70))
* correct common misspellings (e.g., `it's/its, who's/whose`)
* correct punctuation (e.g., spaces before comma)
* correct texts with `toFix` comment
* look through all the texts where entities from triples were not found (see [history](https://gitlab.com/shimorina/webnlg-dataset/commit/d3d62d02df991ae74fa0c95ae619a5e4a3ced6bf)) and either correct them or leave them as they are (if paraphrasing is fine).

All the changes are available in the commit history.
