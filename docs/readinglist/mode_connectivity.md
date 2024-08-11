# Mode-Connectivity

This is a collection of AWESOME things about recent researches about Mode Connectivity, mainly about the findings of Geometric Properties of learned neural networks. 
Mode connectivity is an important property of the loss landscape of deep neural networks and is crucial for understanding weight interpolation-based model merging methods.
For the detailed methods side, we include papers which try to rebasin the different learned models and how to directly induce a better model (generalization) from one-shot training. For the application side, downstream taks such as federated learning, continual learning and sparse neural network which may potentially benefit from the advances of the model connectivity research. Waited to be researched...

## Geometric Properties

### The LMC

- **Topology and Geometry of Half-Rectified Network Optimization.**

  *C. Daniel Freeman, Joan Bruna* [[ICLR17]](https://openreview.net/pdf?id=Bk0FWVcgx)

### The Nonlienar MC
- **Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs.** ![](https://img.shields.io/badge/Nonlinear_Connectivity-blue)
  
  *Timur Garipov, Pavel Izmailov, Dmitrii Podoprikhin, Dmitry Vetrov, Andrew Gordon Wilson* [[Neurips18]](https://proceedings.neurips.cc/paper/2018/file/be3087e74e9100d4bc4c6268cdbe8456-Paper.pdf)[[codes]](https://github.com/timgaripov/dnn-mode-connectivity)

- **Essentially No Barriers in Neural Network Energy Landscape.** ![](https://img.shields.io/badge/Nonlinear_Connectivity-blue)

  *Felix Draxler, Kambis Veschgini, Manfred Salmhofer, Fred A. Hamprecht* [[ICML18]](http://proceedings.mlr.press/v80/draxler18a/draxler18a.pdf)[[codes]](https://github.com/fdraxler/PyTorch-AutoNEB)

 - :eyes: **Loss Surface Simplexes for Mode Connecting Volumes and Fast Ensembling.**
  
   *Gregory W. Benton, Wesley J. Maddox, Sanae Lotfi, Andrew Gordon Wilson* [[ICML21]](http://proceedings.mlr.press/v139/benton21a/benton21a.pdf)[[codes]](https://github.com/g-benton/loss-surface-simplexes)

### Findings
- **Geometry of the Loss Landscape in Overparameterized Neural Networks: Symmetries and Invariances.**

  *Berfin Şimşek, François Ged, Arthur Jacot, Francesco Spadaro, Clément Hongler, Wulfram Gerstner, Johanni Brea* [[ICML21]](https://proceedings.mlr.press/v139/simsek21a/simsek21a.pdf)[[codes]](https://github.com/EPFL-LCN/pub_simsek2021_icml)
  
- (Initialisations) **Random initialisations performing above chance and how to find them.** 

  *Frederik Benzing, Simon Schug, Robert Meier, Johannes von Oswald, Yassir Akram, Nicolas Zucchet, Laurence Aitchison, Angelika Steger* [[NeurIPS22 OPT]](https://arxiv.org/pdf/2209.07509.pdf)[[codes]](https://github.com/freedbee/permuted_initialisations)

- **Linear mode connectivity and the lottery ticket hypothesis.** ![](https://img.shields.io/badge/Linear_Connectivity-blue)

  *Jonathan Frankle, Gintare Karolina Dziugaite, Daniel Roy, Michael Carbin* [[ICML20]](http://proceedings.mlr.press/v119/frankle20a/frankle20a.pdf)[[codes]](https://github.com/facebookresearch/open_lth)

- (Functional behaviors of end points) **On Convexity and Linear Mode Connectivity in Neural Networks.** 

  *David Yunis, Kumar Kshitij Patel, Pedro Henrique Pamplona Savarese, Gal Vardi, Jonathan Frankle, Matthew Walter, Karen Livescu, Michael Maire* [[NeurIPS22 OPT]](https://openreview.net/pdf?id=TZQ3PKL3fPr)

- **Large Scale Structure of Neural Network Loss Landscapes.**

  *Stanislav Fort, Stanislaw Jastrzebski* [[NeurIPS19]](https://openreview.net/pdf?id=rJe_i4rxIS)
  
- **Plateau in Monotonic Linear Interpolation -- A "Biased" View of Loss Landscape for Deep Networks.** 

  *Xiang Wang, Annie N. Wang, Mo Zhou, Rong Ge* [[ICLR23]](https://arxiv.org/abs/2210.01019)
  
- **Analyzing Monotonic Linear Interpolation in Neural Network Loss Landscapes.** 

  *James Lucas, Juhan Bae, Michael R. Zhang, Stanislav Fort, Richard Zemel, Roger Grosse* [[ICML21]](http://proceedings.mlr.press/v139/lucas21a/lucas21a.pdf)[[codes]](https://github.com/AtheMathmo/mli-release)
  
### Theory
- **Explaining landscape connectivity of low-cost solutions for multilayer nets.**

  *Rohith Kuditipudi, Xiang Wang, Holden Lee, Yi Zhang, Zhiyuan Li, Wei Hu, Sanjeev Arora, Rong Ge* [[NeurIPS19]](https://arxiv.org/pdf/1906.06247.pdf)

## Methods for rebasin
- (Width, Depth)(Simulated Annealing) **The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks.**

  *Rahim Entezari, Hanie Sedghi, Olga Saukh, Behnam Neyshabur* [[ICLR22]](https://openreview.net/forum?id=dNigytemkL)[[codes]](https://github.com/rahimentezari/PermutationInvariance)

- **Optimizing mode connectivity via neuron alignment.**

  *N. Joseph Tatro, Pin-Yu Chen, Payel Das, Igor Melnyk, Prasanna Sattigeri, Rongjie Lai* [[NeurIPS19]](https://arxiv.org/pdf/2009.02439.pdf)[[codes]](https://github.com/IBM/NeuronAlignment)
  
- 🔥 :thumbsup: (Three methods) **Git Re-Basin: Merging Models modulo Permutation Symmetries.** 

  *Samuel K. Ainsworth, Jonathan Hayase, Siddhartha Srinivasa* [[ICLR23]](https://openreview.net/forum?id=CQsmMYmlP5T)[[codes]](https://github.com/samuela/git-re-basin)[[pytorch]](https://github.com/themrzmaster/git-re-basin-pytorch)
  
- **Re-basin via implicit Sinkhorn differentiation.**

  *Fidel A. Guerrero Peña, Heitor Rapela Medeiros, Thomas Dubail, Masih Aminbeidokhti, Eric Granger, Marco Pedersoli* [[paper22]](https://arxiv.org/pdf/2212.12042.pdf)
  
- **Linear Mode Connectivity of Deep Neural Networks via Permutation Invariance and Renormalization.**
  
  *Rahim Entezari, Hanie Sedghi, Olga Saukh, Behnam Neyshabur* [[ICLR23]](https://openreview.net/forum?id=gU5sJ6ZggcX)[[codes]](https://github.com/KellerJordan/REPAIR)

## Model merging
 - :thumbsup: [Stochastic Weight Averaging] **Averaging Weights Leads to Wider Optima and Better Generalization.** 
 
   *Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, Andrew Gordon Wilson* [[UAI18]](http://auai.org/uai2018/proceedings/papers/313.pdf)[[codes]](https://github.com/timgaripov/swa)
   
- **Subspace Inference for Bayesian Deep Learning.**

  *Pavel Izmailov, Wesley J. Maddox, Polina Kirichenko, Timur Garipov, Dmitry Vetrov, Andrew Gordon Wilson.* [[UAI19]](http://auai.org/uai2019/proceedings/papers/435.pdf)[[codes]](https://github.com/wjmaddox/drbayes)
 
- **Bayesian Nonparametric Federated Learning of Neural Networks.**

  *Mikhail Yurochkin, Mayank Agarwal, Soumya Ghosh, Kristjan Greenewald, Trong Nghia Hoang, and Yasaman Khazaeni.* [[ICML19]](https://htnghia87.github.io/files/icml19b.pdf)[[codes]](https://github.com/IBM/probabilistic-federated-neural-matching)

- **Model fusion via optimal transport.**

  *Singh, Sidak Pal and Jaggi, Martin* [[NeurIPS20]](https://proceedings.neurips.cc/paper/2020/file/fb2697869f56484404c8ceee2985b01d-Paper.pdf)[[codes]](https://github.com/sidak/otfusion)

- [Averaging merge] **Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time.** ![](https://img.shields.io/badge/Large_Model-green)

  *Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, Ludwig Schmidt* [[ICML22]](https://proceedings.mlr.press/v162/wortsman22a/wortsman22a.pdf)[[codes]](https://github.com/mlfoundations/model-soups)

- **lo-fi: distributed fine-tuning without communication.** ![](https://img.shields.io/badge/Pretrained_connectivity-blue) ![](https://img.shields.io/badge/Large_Model-green)

  *Mitchell Wortsman, Suchin Gururangan, Shen Li, Ali Farhadi, Ludwig Schmidt, Micheal Rabbat, Ari S. Morcos* [[TMLR23]](https://openreview.net/pdf?id=1U0aPkBVz0)
 
- :thumbsup: **Learning Neural Network Subspaces.**

  *Mitchell Wortsman, Maxwell Horton, Carlos Guestrin, Ali Farhadi, Mohammad Rastegari* [[ICML21]](http://proceedings.mlr.press/v139/wortsman21a/wortsman21a.pdf)[[codes]](https://github.com/apple/learning-subspaces)

- **Robust fine-tuning of zero-shot models.** ![](https://img.shields.io/badge/Large_Model-green)

  *Mitchell Wortsman, Gabriel Ilharco, Jong Wook Kim, Mike Li, Simon Kornblith, Rebecca Roelofs, Raphael Gontijo-Lopes, Hannaneh Hajishirzi, Ali Farhadi, Hongseok Namkoong, Ludwig Schmidt* [[CVPR22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.pdf)[[codes]](https://github.com/mlfoundations/wise-ft)

- [Fisher merge] **Merging Models with Fisher-Weighted Averaging.**
  
  *Michael Matena, Colin Raffel* [[NeurIPS22]](https://openreview.net/pdf?id=LSKlp_aceOC)[[codes]](https://github.com/mmatena/model_merging)
  
- [Regression Mean merge] **Dataless Knowledge Fusion by Merging Weights of Language Models.** ![](https://img.shields.io/badge/Large_Model-green)
  
  *Xisen Jin, Xiang Ren, Daniel Preotiuc-Pietro, Pengxiang Cheng* [[ICLR23]](https://openreview.net/forum?id=FCnohuR6AnM)
  
- **Wasserstein Barycenter-based Model Fusion and Linear Mode Connectivity of Neural Networks.** 

  *Aditya Kumar Akash, Sixu Li, Nicolás García Trillos* [[paper23]](https://openreview.net/pdf?id=qHbyR1MKG8K)[[codes]](https://openreview.net/forum?id=qHbyR1MKG8K)

- **PopulAtion Parameter Averaging (PAPA).**

  *Alexia Jolicoeur-Martineau, Emy Gervais, Kilian Fatras, Yan Zhang, Simon Lacoste-Julien* [[paper23]](https://arxiv.org/pdf/2304.03094.pdf)

- **ZipIt! Merging Models from Different Tasks without Training.**

  *George Stoica, Daniel Bolya, Jakob Bjorner, Taylor Hearn, and Judy Hoffman* [[paper23]](http://arxiv.org/abs/2305.03053)


## Pretrained model connectivity
- **What is being transferred in transfer learning?** 

  *Behnam Neyshabur, Hanie Sedghi, Chiyuan Zhang* [[NeurIPS20]](https://proceedings.neurips.cc/paper/2020/file/0607f4c705595b911a4f3e7a127b44e0-Paper.pdf)[[codes]](https://github.com/google-research/understanding-transfer-learning)
  
- **Exploring Mode Connectivity for Pre-trained Language Models.** ![](https://img.shields.io/badge/Large_Model-green)
  
  *Yujia Qin, Cheng Qian, Jing Yi, Weize Chen, Yankai Lin, Xu Han, Zhiyuan Liu, Maosong Sun, Jie Zhou* [[EMNLP22]](https://arxiv.org/pdf/2210.14102.pdf)[[codes]](https://github.com/thunlp/Mode-Connectivity-PLM)
  
- **Knowledge is a Region in Weight Space for Fine-tuned Language Models.** ![](https://img.shields.io/badge/Large_Model-green)
  
  *Almog Gueta, Elad Venezian, Colin Raffel, Noam Slonim, Yoav Katz, Leshem Choshen* [[paper23]](https://arxiv.org/pdf/2302.04863.pdf)

## Equivariant Network Design
- 👀 **Equivariant Architectures for Learning in Deep Weight Spaces.** 
  
  *Aviv Navon, Aviv Shamsian, Idan Achituve, Ethan Fetaya, Gal Chechik, Haggai Maron* [[paper23]](https://arxiv.org/pdf/2301.12780.pdf)[[codes]](https://github.com/AvivNavon/DWSNets)
  
- 👀 **Permutation Equivariant Neural Functionals.** 

  *Allan Zhou, Kaien Yang, Kaylee Burns, Yiding Jiang, Samuel Sokota, J. Zico Kolter, Chelsea Finn* [[paper23]](https://arxiv.org/pdf/2302.14040.pdf)

## Related paper
- **Entropy-SGD optimizes the prior of a PAC-Bayes bound: Generalization properties of Entropy-SGD and data-dependent priors.** ![](https://img.shields.io/badge/Smoothness_Generalization-blue)

  *Gintare Karolina Dziugaite, Daniel Roy* [[ICML18]](http://proceedings.mlr.press/v80/dziugaite18a/dziugaite18a.pdf)

- **Sharpness-Aware Minimization for Efficiently Improving Generalization.** ![](https://img.shields.io/badge/Smoothness_Generalization-blue)

  *Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur* [[ICLR21]](https://arxiv.org/pdf/2010.01412.pdf)[[codes]](https://github.com/google-research/sam)
  
- **Model Ratatouille: Recycling Diverse Models for Out-of-Distribution Generalization.**

  *Alexandre Ramé, Kartik Ahuja, Jianyu Zhang, Matthieu Cord, Léon Bottou, David Lopez-Paz* [[paper23]](https://arxiv.org/pdf/2212.10445.pdf)[[codes]](https://github.com/facebookresearch/ModelRatatouille)

## Applications
- (FL) **Connecting Low-Loss Subspace for Personalized Federated Learning.** 

  *Seok-Ju Hahn, Minwoo Jeong, Junghye Lee* [[KDD22]](https://dl.acm.org/doi/pdf/10.1145/3534678.3539254)[[codes]](https://github.com/vaseline555/SuPerFed)
  
- **Linear Mode Connectivity in Multitask and Continual Learning.** 

  *Seyed Iman Mirzadeh, Mehrdad Farajtabar, Dilan Gorur, Razvan Pascanu, Hassan Ghasemzadeh* [[ICLR21]](https://openreview.net/forum?id=Fmg_fQYUejf)[[codes]](https://github.com/imirzadeh/MC-SGD)
  
- **All Roads Lead to Rome? On Invariance of BERT Representations.** ![](https://img.shields.io/badge/Large_Model-green)

  *Yuxin Ren, Qipeng Guo, Zhijing Jin, Shauli Ravfogel, Mrinmaya Sachan, Ryan Cotterell, Bernhard Schölkopf* [[TACL23]](http://zhijing-jin.com/files/papers/BERTSimilarity_TACL2023.pdf)
  
- (Meta Learning) **Subspace Learning for Effective Meta-Learning.**

  *Weisen Jiang, James Kwok, Yu Zhang* [[ICML22]](https://proceedings.mlr.press/v162/jiang22b/jiang22b.pdf)[[codes]](https://openreview.net/forum?id=C_RTGckbu-A)

- (Incremental Learning) **Towards better plasticity-stability trade-off in incremental learning: a simple linear connector.**

  *Guoliang Lin, Hanlu Chu, Hanjiang Lai* [[CVPR22]](https://openaccess.thecvf.com/content/CVPR2022/papers/Lin_Towards_Better_Plasticity-Stability_Trade-Off_in_Incremental_Learning_A_Simple_Linear_CVPR_2022_paper.pdf)[[codes]](https://github.com/lingl1024/Connector)
  
- :thumbsup: (Sparsity) **Unmasking the Lottery Ticket Hypothesis: What's Encoded in a Winning Ticket's Mask?** 

  *Mansheej Paul, Feng Chen, Brett W. Larsen, Jonathan Frankle, Surya Ganguli, Gintare Karolina Dziugaite* [[ICLR23]](https://openreview.net/pdf?id=xSsW2Am-ukZ)

- :eyes: **Improving Ensemble Distillation With Weight Averaging and Diversifying Perturbation.**

  *Giung Nam, Hyungi Lee, Byeongho Heo, Juho Lee* [[ICML22]](https://arxiv.org/pdf/2206.15047.pdf)[[codes]](https://github.com/cs-giung/distill-latentbe)

- **LCS: Learning Compressible Subspaces for Efficient, Adaptive, Real-Time Network Compression at Inference Time**

  *Elvis Nunez, Maxwell Horton, Anish Prabhu, Anurag Ranjan, Ali Farhadi, Mohammad Rastegari*[[WACV23]](https://openaccess.thecvf.com/content/WACV2023/papers/Nunez_LCS_Learning_Compressible_Subspaces_for_Efficient_Adaptive_Real-Time_Network_Compression_WACV_2023_paper.pdf)[[codes]](https://github.com/apple/learning-compressible-subspaces)
 
- (OOD) **Diverse Weight Averaging for Out-of-Distribution Generalization**

  *Alexandre Ramé, Matthieu Kirchmeyer, Thibaud Rahier, Alain Rakotomamonjy, Patrick Gallinari, Matthieu Cord*[[NeurIPS22]](https://arxiv.org/pdf/2205.09739.pdf)[[codes]](https://github.com/alexrame/diwa)

- **Linear Connectivity Reveals Generalization Strategies.** 
  
  *Jeevesh Juneja, Rachit Bansal, Kyunghyun Cho, João Sedoc, Naomi Saphra* [[ICLR23]](https://arxiv.org/pdf/2205.12411.pdf)[[codes]](https://github.com/anonwhymoos/connectivity)


## Tools

- **Loss landscapes.** For loss landscape visualization and analysis.
    [[pypi]](https://pypi.org/project/loss-landscapes)
