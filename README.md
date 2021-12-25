# XLAnalysis5K
An empirical analysis of cross-linguality in shared embedding spaces for 101 languages and 5000+ language pairs. Using cross-lingual sentence models LaBSE and LASER, we investigate the factors that predict cross-lingual alignment and isomorphism between the embedding (sub)spaces of any two arbitrary languages. We uncover significant effects of basic word order and morphological complexity, in addition to other variables.

## Replicating the experiments
The modules in [/src/Data Generation](https://github.com/AlexJonesNLP/XLAnalysis5K/tree/main/src/Data%20Generation) may be used to generate features for the languages in the dataset, as well as the alignment and isomorphism metrics we use as dependent variables. [generate_features.py](https://github.com/AlexJonesNLP/XLAnalysis5K/blob/main/src/Data%20Generation/generate_features.py) does both these tasks. The main dataset we use is the [superparallel Bible corpus](https://christos-c.com/bible/) from [Christodouloupoulos and Steedman (2014)](https://doi.org/10.1007/s10579-014-9287-y). The [UDHR multiparallel corpus](http://research.ics.aalto.fi/cog/data/udhr/) and [Nunavut Hansard English-Inuktitut parallel corpus](https://nrc-digital-repository.canada.ca/eng/view/object/?id=c7e34fa7-7629-43c2-bd6d-19b32bf64f60) are used for supplemental experiments in our paper. The relevant portions and versions of these datasets are provided in this repo.

[/src/Analysis](https://github.com/AlexJonesNLP/XLAnalysis5K/tree/main/src/Analysis) contains notebooks for carrying out the statistical analysis (bible_bitexts_analysis.ipynb) and visualization (bible_bitexts_tsne.ipynb) portions of our project. If you just want to replicate the analyses without re-generating the data, you can use the data from [/Bible experimental vars](https://github.com/AlexJonesNLP/XLAnalysis5K/tree/main/Bible%20experimental%20vars). 

Some additional notebooks used in our experiments are provided in [/Additional Notebooks](https://github.com/AlexJonesNLP/XLAnalysis5K/tree/main/Additional%20Notebooks). These contain essentially "rough draft" code and we recommend using files from /src/Data Generation instead for rerunning experiments.

## Dependencies 

**Core libraries** \
[Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html) \
[scikit-learn](https://scikit-learn.org/stable/install.html) \
[PyTorch](https://pytorch.org) \
**Sentence embeddings** \
[Sentence Transformers](https://www.sbert.net) \
[laserembeddings](https://pypi.org/project/laserembeddings/) \
**Statistical analysis \& plotting** \
[Seaborn](https://seaborn.pydata.org/installing.html) \
[Pingouin](https://pingouin-stats.org) \
[Matplotlib](https://matplotlib.org/stable/users/installing.html) \
**Computational tools** \
[Faiss](https://github.com/facebookresearch/faiss) \
[Gudhi](http://gudhi.gforge.inria.fr/python/latest/installation.html) \
**Typological vectors** \
[lang2vec](https://pypi.org/project/lang2vec/) \
**Misc** \
[tqdm](https://pypi.org/project/tqdm/)

## Citation

Please cite our paper if you use code or data from this repo:

```
@inproceedings{jones-etal-2021-massively,
    title = "A Massively Multilingual Analysis of Cross-linguality in Shared Embedding Space",
    author = "Jones, Alexander  and
      Wang, William Yang  and
      Mahowald, Kyle",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.471",
    pages = "5833--5847",
    }
```
