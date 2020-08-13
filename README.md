# WGAN Financial Time-Series

By Casper Hogenboom

Thesis project done on Generation Financial Time-Series with GANs. The project was a collaboration between University Maastricht and Wholesale Banking Advanced Analytics team with ING, Spring 2020.

This repository is set-up to run on APPL historic price data for which the results are also included. 


## Abstract: 
The unparalleled success of Generative Adversarial Networks (GAN) in generating realistic synthetic images has initiated a new field of research in machine learning. The GAN models in image and video processing have made impressive improvements by setting new benchmarks in consecutive fashion. However, due to the inherent difficulty of evaluating synthetic generated data instances, this progress has been largely limited to applications like images and video, where visual inspection can serve as a guide.
In this work, we will extend state-of-the-art concepts within GAN training to the (financial) time series domain. Advancements in this domain are still in an early stage, and this work will explore a universal GAN framework that is generalizable across different financial time series datasets.  A Wasserstein GAN with gradient penalty (WGAN-GP) was implemented to learn the underlying distribution of synthethic and financial datasets. The results were obtained on EUR USD Forex historical data, and evaluated against multiple relevant statistical measures. The results showed strong performance in capturing the underlying data distribution. It was concluded that WGAN-GP can be used as a probabilistic prediction model on financial time series.
