# Uncertainty Estimation for Debiased Models: Does Fairness Hurt Reliability?

This repository contains the implementation for the "Uncertainty Estimation for Debiased Models: Does Fairness Hurt Reliability?" paper from IJCNLP-AACL2023. The code provides a variety of methods for debiasing and scenarios for its joint usage with Uncertainty Estimation (UE) methods for neural network models.

The debiasing methods implementation uses the [*fairlib*](https://github.com/HanXudong/fairlib) framework. The UE part will be added later.

## Abstract

When deploying a machine learning model, one should aim not only to optimize performance metrics such as accuracy but also care about model fairness and reliability. Fairness means that the model is prevented from learning spurious correlations between a target variable and socio-economic attributes, and is generally achieved by applying debiasing techniques. Model reliability stems from the ability to determine whether we can trust model predictions for the given data. This can be achieved using uncertainty estimation (UE) methods. Debiasing and UE techniques potentially interfere with each other, raising the question of whether we can achieve both reliability and fairness at the same time. This work aims to answer this question empirically based on an extensive series of experiments combining state-of-the-art UE and debiasing methods, and examining the impact on model performance, fairness, and reliability.

## How to run experiments

All of the scripts can be found in the scripts directory. This directory contains hyperparameter tuning scripts, scripts for model training, and scripts for UE.

## Citation

```bibtex
@inproceedings{kuzmin-EtAl:2023:ijcnlp,
  author    = "Kuzmin, Gleb  and  Vazhentsev, Artem  and  Shelmanov, Artem  and  Han, Xudong  and  Suster, Simon  and  Panov, Maxim  and  Panchenko, Alexander  and  Baldwin, Timothy",
  title     = "Uncertainty Estimation for Debiased Models: Does Fairness Hurt Reliability?",
  booktitle      = "Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics",
  month          = "November",
  year           = "2023",
  address        = "Nusa Dua, Bali",
  publisher      = "Association for Computational Linguistics",
  pages     = "744--770",
  url       = "https://aclanthology.org/2023.ijcnlp-long.48"
}
```
