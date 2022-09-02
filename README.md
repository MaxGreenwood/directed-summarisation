# directed-summarisation-w-amr

*Submitted in partial fulfillment of the requirements for the MSc Artificial Intelligence of Imperial College London.*

This repository contains the software used to conduct the experiments in my thesis *Directed summarisation with Abstract Meaning Representation and Moral Foundation Theory*. The pipeline described in the thesis report can be run on the `Project Pipeline.ipynb` Colab notebook when this repository and models from the listed repositories below are uploaded into Google Drive storage.

Original code is found in the `.ipynb` files and `u-sas` modules which are supplemented with new functions. The pipeline makes use of the following open-source models and source code:

* `amrlib`: https://github.com/bjascob/amrlib
* `argument_classification`: https://github.com/UKPLab/acl2019-BERT-argument-classification-and-clustering
* `fast_align`: https://github.com/clab/fast_align
* `stanford-corenlp-full-2018-10-05`: https://stanfordnlp.github.io/CoreNLP/

The unsupervised subgraph extraction algorithm in the `u-sas` folder is an adaptation of the following code from Dohare and Gupta: https://github.com/vgupta123/Unsupervised-SAS.

The `e2e-coref` package is functionally the same as the original but has been edited to run smoothly in the Colab environment. The original package can be found at: https://github.com/kentonl/e2e-coref

The `graph_visualisation` package (not used in final pipeline) is adapted from: https://github.com/xdqkid/AMR-Visualization

Please contact mgg21@ic.ac.uk for any questions relating to this software archive.
