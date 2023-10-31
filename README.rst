* Written by Camila Riccio, Camilo Rocha and Jorge Finke
* Last update: 12/10/23 

CoSynthEx
=============

This is a Python3 implementation to o generate synthetic expression data conditioned on phenotypic traits and sample conditions, such as control or stress.
This tool leverages the capabilities of a Conditional Generative Adversarial Network (cGAN)
to create realistic and customizable synthetic expression data for a wide range of applications
in genomics and bioinformatics.

The generation of synthetic expression data is a solution to critical challenges in various scientific
disciplines, addressing issues related to data availability, and experimentation. Synthetic
data plays a pivotal role in augmenting real-world datasets, enriching them with additional
data points and features. This augmentation significantly enhances the performance of machine
learning models, particularly in scenarios with limited real data.
Furthermore, synthetic data serves as a valuable tool for testing and validating novel
methods in gene expression analysis, including emerging machine learning algorithms and feature
selection techniques. This approach allows researchers to thoroughly assess the strengths
and weaknesses of these methods before their application to real-world datasets, ensuring
robust and reliable results.
Cost-effectiveness is another compelling advantage of synthetic data generation. The
process is notably more efficient and economical compared to collecting real biological data,
making it particularly advantageous for preliminary research, algorithm development, and
feasibility studies.
Despite the existence of various methods for generating synthetic expression data, many
struggle to accurately replicate key properties of gene expression data, as noted in `previous
research <https://doi.org/10.1093/bioinformatics/btt438>`_. 
In response to this challenge, we introduce CoSynthEx,
a software tool designed for the conditional generation of synthetic expression data. This
innovative approach aims to create a more realistic simulation of expression data by incorporating
additional contextual information, such as phenotypic traits and sample conditions
(e.g., control or stress).

CoSynthEx employs a conditional generative adversarial network (cGAN) as its foundational model.
This cGAN takes real expression data, phenotypic trait data, and sample
condition information as inputs. Through training and parameter adjustment, the model’s
loss curves are optimized to exhibit ideal behavior. 
During training, the generator and discriminator engage in a competitive and adaptive
process. The generator strives to produce increasingly realistic data, challenging the
discriminator’s ability to differentiate between real and synthetic data. Once the model is
successfully trained, it can generate synthetic expression data starting from noise conditioned
on phenotypic traits and sample conditions.


Setup
------
Clone the repository::

  git clone git@github.com/criccio35/CoSynthEx


Requirements
------------
Install all of the modules listed in the Python requirements file into the project environment::

  pip install -r requirements.txt

How to use
----------

Make sure the CoSynthEx.py file is in the same folder
as the notebook or python script where you want to use the module.
Load the module as shown below::

  import CoSynthEx as cse

Find a complete example and the documentation `here <docs/cosynthex material auxiliar.pdf>`_.
Find a complete description `here <docs/cosynthex description.pdf>`_.
