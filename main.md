# Pulling out the big GANs

## From representation learning to faking things

<div style="font-size: 24pt; text-align: left; margin-top: 96px; margin-bottom: 24px;">Eduardo Pinho</div>
<br>
<!-- <div style="font-size: 20pt; text-align: left;">DSPT</div> -->

<div style="font-size: 12pt; text-align: right; margin: 1cm;">5th June 2019</div>
<div class="foot" style="margin-left: 0pt; margin-right: 0pt; margin-bottom: 0pt; background-color: white;">
    <img class="foot" src="img/ua.png"/>
    <img class="foot" src="img/ieeta.png"/>
    <img class="foot" src="img/logobio.png"/>
    <img class="foot" src="https://www.datascienceportugal.com/wp-content/uploads/2018/08/LogoDSPT_Grey-300x225.png"/>
</div>

---

## Table of Contents

<br/>

1. Representation Learning
2. Generative Adversarial Networks
3. Use case: concept detection from medical images

---

## Deep Learning

<div class="container">
<div class="column column-one">
<ul>
  <li>Achieving significant milestones over the last decade.</li>
  <li>Evolving quickly.</li>
  <li>Ever ubiquitous and attainable.</li>
  <ul>
    <li>Good literature is open-access (ArXiv)</li>
    <li>Open-source software (TensorFlow, Torch, ...)</li>
  </ul>
  <li class="fragment" data-fragment-index="0">Annotated data is often limited.</li>
  <li class="fragment" data-fragment-index="0">How to explore unlabeled data?</li>
<ul>
</div>
<div class="column column-two">
<img style="width: 6cm" src="img/xkcd-tasks.png " />
</div>
</div>

.

<img style="height: 13cm" src="img/ai-venn.png" />

<span class="fragment" data-fragment-index="1"><b>Representation Learning</b></span>

<span class="cite">Goodfellow et al. <em>"Deep Learning"</em>. 2016. <a href="https://www.deeplearningbook.org">www.deeplearningbook.org</a></span>

.

## Representation Learning

 - Also called **feature learning**.
 - Given a data set $X$, learning a function $f(x) \rightarrow z$, mapping samples to a new domain $Z$ that makes other problems easier to solve.
 - <!-- .element: class="fragment" data-fragment-index="0" --> <em>Feature extraction</em>?
 - Often posed as either deterministic or probabilistic. <!-- .element: class="fragment" data-fragment-index="1" -->
    - Original samples in distribution: $x$ ; $p(x)$.
    - Representation: $z = f(x)$ ; $p(z | x)$
    - Classification: $y = c(z)$ ; $p(y | z)$

.

General priors of representation learning:

<ul>
<li>Smoothness ($a \approx b \implies f(a) \approx f(b)$)</li>
<li>Hierarchical organization</li>
<li>Sparsity</li>
<li>Semi-supervised learning</i>
<li>Temporal and spatial coherence</li>
<li>Multiple explanatory factors</li>
<li>Shared factors across tasks</li>
<li>Simplicity of factor dependencies</li>
<li>...</li>
<li>Manifolds</li>
</ul>
</div>

<span class="cite">Bengio et al. <em>"Representation Learning: A review and new perspectives"</em>. 2013.</span>

.

## What is a manifold?

> [[wikipedia](https://en.wikipedia.org/wiki/Manifold)] In mathematics, a **manifold** is a topological space that locally resembles Euclidean space near each point. More precisely, each point of an _n_-dimensional manifold has a neighbourhood that is homeomorphic to the Euclidean space of dimension n.
<!-- .element: style="font-size: 12pt" -->

![](img/spiked-math-poker.png) <!-- .element: class="fragment" data-fragment-index="0" -->

Notes: So what is a manifold? Well, it's the reason why mathematicians can't play poker. They usually end up with a bad hand and go "man I fold". except not.

.

The **manifold hypothesis**:

> [...] real-world data presented in high-dimensional spaces are expected to concentrate in the vicinity of a manifold $\mathcal{M}$ of much lower dimensionality $d_{\mathcal{M}}$, embedded in high-dimensional input space $\mathbb{R}^{d_x}$.

<span class="fragment" data-fragment-index="0">ImageNet: over 64,000s pixels / image, but not $256^{3 \times 64,000}$ possible images</span>

<span class="fragment" data-fragment-index="1">Manifold learning $X \rightarrow \mathcal{M}$</span>

Notes: Think of it as a subspace of the original data domain, to which our data tends to concentrate to. Considering the domain of natural images for a moment: if they are resized to 256x256, we have over 64 thousand RGB pixels / image, which makes an absurdly huge dimensionality. However, we know that we won't find every possible RGB image combination in ImageNet. It will be incredibly more specific. Hence, these images will sit in this manifold, and they can be mapped into a domain of lower dimensionality that can yet retain all the information from the previous domain of the data.

.

### Early Representation Learning

Principal Component Analysis

- models a linear manifold
- $z = f(x) = \mathrm{W}^Tx + b$
- $\mathrm{W}$: orthogonal basis of greater variance
- Decorrelated features in $z$

.

### Autoencoder

<img width="300px" src="img/autoencoder.svg" />

- Encoder-decoder with a bottleneck (AKA latent code)
- Minimize reconstruction loss: $D(E(x)) \sim x$
- Constrained $z$
<li class="fragment" data-fragment-index="0"> Can be <em>overcomplete</em>!</li>
  <ul class="fragment" data-fragment-index="0">
  <li>Regularize the representation (e.g. induce sparsity)</li>
  </ul>

.

### Variational Autoencoder

<div style="height: 10cm">
<img src="img/vae.svg" />
</div>

<ul style="font-size: 20pt">
<li>Autoencoder with Kulback-Leibler divergence for variational inference</li>
<li>Generative model: can sample images from the latent space</li>
</ul>

<span class="cite">Kingma et al. <em>"Auto-Encoding Variational Bayes"</em>. 2014</span>

.

#### VAE Sampling

<img src="img/vae_noise_sampling.png" />

.

### Honorable mentions:

- k-means clustering
   - plus other clustering algorithms
- Sparse Coding
- Boltzmann Machines
   - and Restricted Boltzmann Machines (RBMs)


---

# Generative Adversarial Networks

.

A min-max game between a **Generator** and a **Discriminator**

<!-- Definition here -->

- $G$: given a prior $z$, create samples $q(x|z)$ close to $p(x)$
- $D$: distinguish real samples $p(x)$ from generated samples $q(x)$

$$\min_{G} \max_{D} V(D, G) = \mathbb{E}_{x \sim p(x)} \big(\log (D(x))\big)$$

$$ + \mathbb{E}_{z \sim p(z)} \big(\log (1 - D(G(z))\big)$$

In practice, a non-saturating adaptation is used.

.

<img src="img/gan_samples.png" />

.

<img src="img/gan_lolcat.jpg" />


.

### Latent Space Arithmetic

<img src="img/gan_vector_arithmetic.png" />

<span class="cite">Radford et al. <em>"Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"</em>. 2016</span>

.

### Domain Transfer: Cycle GAN

<!-- TODO -->

.

<img src="horse2zebra.gif" />

.

#### Super-resolution

<img src="img/srgan.png" />

<span class="cite">Ledig et al. <em>"Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"</em>. 2017</span>

.


## Big GAN

Large GAN for natural images.

<!-- TODO add samples -->

- Original source of dogball <!-- .element: class="fragment" data-fragment-index="0" -->

<!-- TODO add dogball -->

<span class="cite">Brock et al. <em>"Large Scale GAN Training for High Fidelity Natural Image Synthesis"</em>. 2018</span>

.

## Self Attention GAN (SAGAN)

<img src="img/sagan_samples.png" />

- Integrate self attention modules
- Faithful replication of local content from global context

<span class="cite">Zhang et al.<em>"Self attention Generative Adversarial Networks"</em>. 2018</span>

.

## Progressive GAN

<img src="img/prog_gan_actor.png" />

- Progressively grow the generator and discriminator.
   - 4x4, 8x8, ..., **1024x1024**

<!-- TODO insert GIF -->

<span  class="cite">Karras et al. <em>"Progressive Growing of GANs for Improved Quality, Stability, and Variation"</em>. 2018</span>

.

## StyleGAN

- NVIDIA's next step
- Unsupervised learning of style (coarse + fine)

<!-- TODO insert image or something of the like -->

<span class="cite">Karras et al. "A Style-Based Generator Architecture for Generative Adversarial Networks". 2018</span>

.

### [www.ThisPersonDoesNotExist.com](https://www.thispersondoesnotexist.com)

<iframe width="960" height="600" src="https://www.thispersondoesnotexist.com"></iframe>

.

### Shortcomings

Easy to shoot your foot with:

- Hard to train ─ balancing discriminator and generator
- Mode collapse

.

### Attempts of improving the training process

- Feature Matching; Minibatch Discrimination; Batch Renormalization
   - Salimans et al. 2016
- Wasserstein GAN (Arjovsky et al. 2017)
- Gradient Penalization (Gulrajani et al. 2017)
- Adding noise to inputs 
- Spectral Normalization (Miyamoto et al. 2018)
- Relativistic GAN loss (Jolicoeur-Martineau. 2018)

.

Relativistic GAN

<img src="https://ajolicoeur.files.wordpress.com/2018/06/screenshot-from-2018-06-30-11-04-05.png?w=656" />

---

## Use case: concept detection from medical images

.

- Medical imaging data sets, very few to no annotations.
   - Expertise is usually required

.

ImageCLEF Caption 2017 + 2018 + 2019

<img style="height: 6.5cm" src="img/imageclef2017-examples.png" />

- Pubmed Central (PMC): images from biomedical literature.
- Annotations: lists of CUI identifiers extracted from captions.

<div class="container">
<div class="column column-one">
<h3>2017</h3>
<ul>
<li>Challenge's pilot year</li>
<li>Training: 164,614 images</li>
<li>Validation: 10,000 images</li>
<li>Testing: 10,000 images</li>
<ul>
</div>
<div class="column column-two">
<h3>2018</h3>
<ul>
<li>No subfigures, pre-filtering</li>
<li>Training: 223,859 images</li>
<li>Testing: 9,938 images</li>
<li>Over 100 thousand unique concepts</li>
<ul>
</div>
</div>

.

### Method Outline

<img style="height: 95%" src="img/feature-learning-pipeline-generic.svg" />

<!--
.

#### Bags of Visual Words

<div style="height: 8cm">
<img src="img/visual_bows_orb.svg" />
</div>

<ul style="font-size: 20pt">
<li>Extract visual keypoints:</li>
<ul>
<li>Scale Invariant Feature Transform (SIFT) - Lowe et al. 2004</li>
<li>Oriented FAST and Rotated BRIEF (ORB) - Rublee et al. 2011</li>
</ul>
<li>Construct visual vocabulary (k-means clustering)</li>
<li>Quantify keypoints into "bags"</li>
</ul>

-->

.

#### Sparse Denoising autoencoder

<div style="height: 10cm">
<img src="img/sdae.svg" />
</div>

<ul style="font-size: 20pt">
<li>autoencoder with sparsity-inducing regularization</li>
<ul>
<li>Vincent et al. 2010</li>
</ul>
<li>Added Gaussian noise $\tilde{x}$</li>
<li>Features extracted from the bottleneck vector</li>
</ul>

.


#### Bidirectional Generative Adversarial Network

<div><img src="img/bigan.svg" /></div>

<ul style="font-size: 20pt">
<li>Generative adversarial network (GAN) with an encoder</li>
<ul><li>Donahue et al. 2016</li></ul>
</ul>
<div style="font-size: 20pt">
$$V(G, E, D) = \min_{G, E} \max_{D} \mathbb{E}_{\mathrm{x} \sim p_{\mathrm{x}}} [\log{D(\mathrm{x}, E(\mathrm{x}))}] + \mathbb{E}_{\mathrm{z} \sim p_\mathrm{z}} [\log{(1 - D(G(\mathrm{z}), \mathrm{z}))}]$$
</div>
<ul style="font-size: 20pt">
<li>Features extracted with $E(x)$</li>
</ul>

.

#### Adversarial autoencoder

<div><img src="img/2018aae.svg" /></div>

<ul style="font-size: 20pt">
<li>autoencoder with adversarial loss for regularization</li>
<ul><li>Makhzani et al. 2015</li></ul>
<li>$D$ forces $E$ to approximate a prior distribution $\mathcal{N}(0, I)$</li>
<li>Features extracted from the bottleneck vector</li>
</ul>

.

#### Flipped-Adversarial autoencoder

<div><img src="img/faae_2lvl.svg" /></div>

<ul style="font-size: 20pt">
<li>A GAN with a latent regressor $E$</li>
<ul><li>Zhang et al. 2018</li></ul>
<li>2-level for stability</li>
<li>Baseline, poor performance is expected</li>
<li>Features extracted with $E(x)$</li>
</ul>

.

### Multi-label Classification

#### Logistic Regression

- Attempt to classify the $n$ most frequent concepts.
   - _750_ in ImageCLEF 2017, _500_ in ImageCLEF 2018
- Loose threshold fine tuning to optimize F1 score.

<img src="img/feature-learning-pipeline-logistic-regression.svg" />

.

### Multi-label Classification

#### k-Nearest Neighbors

- Index training set
- Similarity search over the given features (CBIR)
- Combine the concepts of the first $k$ search hits

<img src="img/feature-learning-pipeline-search.svg" />

.

### Results - ImageCLEF 2018

Official participation

|Rank | Run file name                                    | Kind  | Classifier   | **Test $F_1$** |
|-----|--------------------------------------------------|-------|--------------|----------------|
|**1**| <small>aae-500-o0-2018-04-30\_1217</small>       |  AAE  | linear(500)  | **0.1102**     |
|  2  | <small>aae-2500-merge-2018-04-30\_1812</small>   |  AAE  | linear(2500) | 0.1082         |
|  3  | <small>lin-orb-500-o0-2018-04-30\_1142</small>   |  ORB  | linear(500)  | 0.0978         |
|  9  | <small>faae-500-o0-2018-04-27\_1744</small>      | F-AAE | linear(500)  | 0.0825         |
| 11  | <small>knn-ip-aae-train-2018-04-27\_1259</small> |  AAE  | k-NN(cosine) | 0.0570         |
| 12  | <small>knn-aae-all-2018-04-26\_1233</small>      |  AAE  | k-NN($L^2$)  | 0.0559         |
| 19  | <small>knn-orb-all-2018-04-24\_1620</small>      |  ORB  | k-NN($L^2$)  | 0.0314         |
| 21  | <small>knn-ip-faae-all-2018-04-27\_1512</small>  | F-AAE | k-NN(cosine) | 0.0280         |
| 22  | <small>knn-faae-all-2018-04-26\_0933</small>     | F-AAE | k-NN($L^2$)  | 0.0272         |

.

#### Final Remarks

- Unsupervised learning methods are very promising.
- Representations can be applied in CBIR.

<br/><br/>

- Generative adversarial networks are still a hot topic. <!-- .element: class="fragment" data-fragment-index="0" -->
- Many other feature learning methods. <!-- .element: class="fragment" data-fragment-index="0" -->
- Explore non-visual information in representation learning. <!-- .element: class="fragment" data-fragment-index="0" -->

Open-source: [github.com/bioinformatics-ua/imageclef-toolkit](https://github.com/bioinformatics-ua/imageclef-toolkit)

---

# Conclusion

<img src="img/silhouette-kungfu.png" />

Systems today are very likely to shift to these paradigms. <!-- .element: class="arrow-bullet fragment" data-fragment-index="0" -->

- Content-based queries → meaningful results <!-- .element: class="arrow-bullet fragment" data-fragment-index="0" -->
- Enhanced computer systems → assist medical staff <!-- .element: class="arrow-bullet fragment" data-fragment-index="0" -->

---

### Thank you!


<div style="font-size">
</div>
<div style="font-size: 12pt; text-align: right; margin: 1cm;">5th June 2019</div>
<div class="foot" style="margin-left: 0pt; margin-right: 0pt; margin-bottom: 0pt; background-color: white;">
   <a>GitHub: Enet4</a>
   <a href="https://www.twitter.com/E_net4>Twitter: @E_net4</a>
    <img class="foot" src="https://www.datascienceportugal.com/wp-content/uploads/2018/08/LogoDSPT_Grey-300x225.png"/>
</div>

Note: This concludes my presentation.