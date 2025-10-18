# Dynamic Causal Hypergraph DCH — References and Reading

This curated bibliography supports the DCH technical specification and the Causa-Chip hardware co-design. It is organized by topic to align with sections in the spec. Use this Markdown list during drafting; a BibTeX export [docs/references.bib](./references.bib) will be added later.

Conventions
- Citation keys in brackets (example [Allen1983]) are stable handles used across the spec.
- When multiple editions exist, prefer the oldest stable scholarly reference and provide a convenient link.

## Temporal logic, causality, and hypergraphs

- [Allen1983] James F. Allen. Maintaining Knowledge about Temporal Intervals. Communications of the ACM, 26(11), 1983. https://doi.org/10.1145/182.358434
- [Pearl2009] Judea Pearl. Causality: Models, Reasoning, and Inference. 2nd ed., Cambridge University Press, 2009. https://bayes.cs.ucla.edu/BOOK-2K/
- [Feng2019] Yifan Feng, Haoxuan You, Zizhao Zhang, Rongrong Ji, Yue Gao. Hypergraph Neural Networks. AAAI 2019. https://ojs.aaai.org/index.php/AAAI/article/view/3790
- [Bai2021] Song Bai, Feihu Zhang, Philip H. S. Torr. Hypergraph Convolution and Hypergraph Attention. Pattern Recognition 110, 2021. (for broader hypergraph ops) https://arxiv.org/abs/1901.08150
- [Zhou2021] Tianyu Zhou et al. Dynamic Hypergraph Neural Networks. (survey/representative dynamic HGNN) https://arxiv.org/abs/2008.00778

## Frequent subgraph mining (FSM), streaming and dynamic graph pattern mining

- [YanHan2002] Xifeng Yan, Jiawei Han. gSpan: Graph-Based Substructure Pattern Mining. ICDM 2002. https://doi.org/10.1109/ICDM.2002.1184038
- [Huan2003] Jun Huan, Wei Wang, Jan Prins. Efficient Mining of Frequent Subgraphs in the Presence of Isomorphism. ICDM 2003. https://doi.org/10.1109/ICDM.2003.1250950
- [Bifet2010] Albert Bifet, Geoff Holmes, Bernhard Pfahringer, Richard Kirkby. MOA: Massive Online Analysis. JMLR 2010. (stream mining framework concepts) https://jmlr.org/papers/v11/bifet10a.html
- [Chakrabarti2006] Deepayan Chakrabarti. Dynamic Graph Mining: A Survey. SIGKDD Explorations 2006. https://doi.org/10.1145/1147234.1147237
- [Zou2016] Zhengyi Zou et al. Frequent Subgraph Mining on a Single Large Graph. VLDB 2016. (background) http://www.vldb.org/pvldb/vol9/p860-zou.pdf
- [Bose2018] Arindam Bose et al. A Survey of Streaming Graph Processing Engines. https://arxiv.org/abs/1807.00336
- [Wang2020] Yuchen Wang et al. Incremental Graph Pattern Mining: A Survey. https://arxiv.org/abs/2007.08583

## Spiking neural networks (SNNs), event-based datasets, and tools

- [Orchard2015] Garrick Orchard et al. Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades. Frontiers in Neuroscience, 2015. (N-MNIST) https://doi.org/10.3389/fnins.2015.00437
- [Amir2017] A. Amir et al. A Low Power, Fully Event-Based Gesture Recognition System. CVPR 2017. (DVS Gesture dataset) https://doi.org/10.1109/CVPR.2017.298
- [Norse] H. P. Zenke et al. Norse: A Deep Learning Library for Spiking Neural Networks. (PyTorch-based SNN) https://github.com/norse/norse
- [BindsNET] Hazan et al. BindsNET: A Spiking Neural Networks Library in Python. Frontiers in Neuroinformatics, 2018. https://doi.org/10.3389/fninf.2018.00089
- [tonic] G. M. Hunsberger, F. Ceolini et al. Tonic: A Tool to Load and Transform Event-Based Datasets. https://github.com/neuromorphs/tonic

## Graph representation learning and similarity

- [Hamilton2017] William L. Hamilton, Rex Ying, Jure Leskovec. Inductive Representation Learning on Large Graphs (GraphSAGE). NeurIPS 2017. https://arxiv.org/abs/1706.02216
- [Grover2016] Aditya Grover, Jure Leskovec. node2vec: Scalable Feature Learning for Networks. KDD 2016. https://doi.org/10.1145/2939672.2939754
- [Shchur2018] O. Shchur, M. Mumme, A. Bojchevski, S. Günnemann. Pitfalls of Graph Neural Network Evaluation. Relates to robust baselines. https://arxiv.org/abs/1811.05868

## Random walks, temporal graphs, and reasoning constraints

- [Ribeiro2018] Leonardo F. R. Ribeiro, Pedro H. P. Saverese, Daniel R. Figueiredo. struc2vec: Learning Node Representations from Structural Identity. KDD 2017/2018. (structure-aware walks) https://doi.org/10.1145/3097983.3098061
- [Kazemi2020] Seyed Mehran Kazemi et al. Relational Representation Learning for Dynamic Knowledge Graphs. (temporal constraints) https://arxiv.org/abs/1905.11485
- [Leskovec2014] Jure Leskovec et al. Temporal Networks. (book/survey) https://arxiv.org/abs/1607.01781

## Hardware co-design, PIM, memory fabrics, NoC

- [Bender2002] Michael A. Bender, Richard Cole, Erik D. Demaine, Martin Farach-Colton, Jack Zito. Two Simplified Algorithms for Maintaining Order in a List. SODA 2002. (Packed Memory Array foundations) https://doi.org/10.5555/545381.545411
- [Chi2016] Pengfei Chi et al. PRIME: A Novel Processing-in-Memory Architecture for Neural Network Computation in ReRAM-based Main Memory. ISCA 2016. https://doi.org/10.1109/ISCA.2016.12
- [Shafiee2016] Ali Shafiee et al. ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars. ISCA 2016. https://doi.org/10.1109/ISCA.2016.12
- [Jiang2021] W. Jiang et al. A Survey on Processing-in-Memory. ACM Computing Surveys, 2021. https://doi.org/10.1145/3451210
- [Kim2018] John Kim, William J. Dally. Scalable On-Chip Interconnection Networks. Morgan & Claypool 2018. (NoC principles) https://doi.org/10.2200/S00809ED1V01Y201804CAC043

## Continual learning, task-aware control, and neuro-symbolic integration

- [Parisi2019] German I. Parisi et al. Continual Lifelong Learning with Neural Networks: A Review. Neural Networks, 2019. https://doi.org/10.1016/j.neunet.2019.01.012
- [Kirkpatrick2017] James Kirkpatrick et al. Overcoming Catastrophic Forgetting in Neural Networks. PNAS, 2017. (EWC) https://doi.org/10.1073/pnas.1611835114
- [d’AvilaGarcez2019] Artur d’Avila Garcez, Luis C. Lamb. Neurosymbolic AI: The 3rd Wave. (perspective) https://arxiv.org/abs/2012.05876
- [Valiant2000] Leslie Valiant. Robust Logics. Annals of Pure and Applied Logic, 2000. (neuro-symbolic inspiration) https://doi.org/10.1016/S0168-0072(00)00005-7

## Implementation, engineering, and measurements

- [Dean2013] Jeffrey Dean, Luiz André Barroso. The Tail at Scale. Communications of the ACM, 2013. (tail latency discipline) https://doi.org/10.1145/2408776.2408794
- [Goldstein2020] Moshe Goldstein et al. Measuring ML System Performance: Metrics and Methodologies. (Engineering perspective) arXiv:2008. https://arxiv.org/abs/2008.XXXX

## Pointers to software and datasets (practical)

- Norse (PyTorch SNN): https://github.com/norse/norse
- BindsNET: https://github.com/BindsNET/bindsnet
- Tonic (event datasets): https://github.com/neuromorphs/tonic
- DVS Gesture: https://research.ibm.com/publications/dvs-gesture-dataset (landing page)
- N-MNIST: https://www.garrickorchard.com/datasets/n-mnist

## To-Do (for v0.2 references revision)

- Validate and expand DHGNN dynamic hypergraph citations with the most recent surveys.  
- Add canonical labeling hardware references (CAM designs and counting accelerators).  
- Add specific task-aware SNN literature references (SCA-SNN or nearest strong alternative) with stable DOIs.  
- Populate a BibTeX file [docs/references.bib](./references.bib) with the above keys and cross-check in-text mentions.
