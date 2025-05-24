# WebNLG Dataset Summary

This repository presents the evolution of the WebNLG corpus.

Each folder contains *the same data* in two formats: `xml` and `json`.

* 27 November 2020: `release_v3.0` 

	WebNLG+ data used in the [WebNLG challenge 2020](https://webnlg-challenge.loria.fr/challenge_2020/): bi-lingual (English, Russian) and bi-directional (generation and parsing). More in this [README](./release_v3.0/README.md).

* 25 November 2019: `release_v2.1` (16,095 data inputs and 42,873 data-text pairs) and `release_v2.1_constrained`

	It is the latest release. It has the same data as release_v2 and release_v2_constrained.

	5,667 texts were cleaned: misspellings were corrected and missing verbalisations were added to some texts. More in this [README](./release_v2.1/README.md).

* 14 September 2018: `release_v2` (16,095 data inputs and 42,892 data-text pairs)

	It includes release_v1 and test data (seen categories) from the WebNLG challenge.

	We split it into train/dev/test, ensuring equal representation of DBpedia categories and tripleset sizes.

	Tree shapes and types (_sibling, chain, mixed_) were added for each input RDF tree.

* 14 September 2018: `release_v2_constrained`

	It has the same data as release_v2.

	The split into train/dev/test is more challenging. That split ensures that a triple occurring in train/dev is not present in test (more info in the INLG 2018 paper below).

* 22 January 2018: `release_v1` (14,237 data inputs and 37,975 data-text pairs)

	It doesn't include test data (seen categories) from the challenge.

	No split into train/dev/test was provided.
	
	Covers 15 DBpedia categories.

* 14 April 2017: `webnlg_challenge_2017` (9,674 data inputs and 25,298 data-text pairs)

	Contains the data used in the [WebNLG Challenge 2017](https://webnlg-challenge.loria.fr/challenge_2017/).
	
	Covers 10 DBpedia categories (the _City_ category only partially).
	
	The [enriched version](https://github.com/ThiagoCF05/webnlg) is available.

## Documentation

* Official: <https://webnlg-challenge.loria.fr/docs/>
* WebNLG Dataset Card at GEM: <https://gem-benchmark.com/data_cards/WebNLG>
* WebNLG Dataset Card at HuggingFace Datasets: <https://huggingface.co/datasets/web_nlg#dataset-card-for-webnlg>

### WebNLG XML Reader

* <https://gitlab.com/webnlg/corpus-reader>
* HuggingFace dataset reader: <https://huggingface.co/datasets/viewer/?dataset=web_nlg>
* TensorFlow dataset reader (not all WebNLG versions are supported): <https://www.tensorflow.org/datasets/catalog/web_nlg>

### Other related resources
* Enriched version of WebNLG (referring expressions, delexicalised templates, German silver-standard translation, etc): [link](https://github.com/ThiagoCF05/webnlg)
* Detect typos in WebNLG: [link](https://github.com/abevieiramota/challenge-webnlg/blob/master/notebook/14%20-%20Searching%20misspellings%20in%20references.ipynb).
Those were corrected in the release_v2.1.

## Publications
* [Creating Training Corpora for NLG Micro-Planners](http://www.aclweb.org/anthology/P17-1017). C. Gardent, A. Shimorina, S. Narayan, L. Perez-Beltrachini. ACL 2017.

* [The WebNLG Challenge: Generating Text from RDF Data](http://aclweb.org/anthology/W17-3518). C. Gardent, A. Shimorina, S. Narayan, L. Perez-Beltrachini. INLG 2017.

* [Handling Rare Items in Data-to-Text Generation](http://aclweb.org/anthology/W18-6543). A. Shimorina, C. Gardent. INLG 2018.

## Citing

* If you use the WebNLG corpus, please cite

```
@InProceedings{gardent2017creating,
  author = 	"Gardent, Claire
		and Shimorina, Anastasia
		and Narayan, Shashi
		and Perez-Beltrachini, Laura",
  title = 	"Creating Training Corpora for NLG Micro-Planners",
  booktitle = 	"Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  year = 	"2017",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"179--188",
  location = 	"Vancouver, Canada",
  doi = 	"10.18653/v1/P17-1017",
  url = 	"http://www.aclweb.org/anthology/P17-1017"
}
```

* If you use _release\_*\_constrained_ in particular, please cite

```
@InProceedings{shimorina2018handling,
  author = 	"Shimorina, Anastasia
		and Gardent, Claire",
  title = 	"Handling Rare Items in Data-to-Text Generation",
  booktitle = 	"Proceedings of the 11th International Conference on Natural Language Generation",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"360--370",
  location = 	"Tilburg University, The Netherlands",
  url = 	"http://aclweb.org/anthology/W18-6543"
}
```

## License
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Contact
* webnlg2017@inria.fr
* or create an issue in this repository

## Contribute
If you work with WebNLG and spot an error somewhere, do not hesitate to write to us or create a PR. Your changes will be integrated in a new WebNLG version.

New release is usually prepared in a new branch. Have a look at branches to see recent corrections in the corpus.
