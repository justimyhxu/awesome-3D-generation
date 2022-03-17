# Awesome 3D Generation

## Overview

This repository collects the studies on 3D generation, including both [3D shape generation](#3d-shape-generation) and [3D-aware image generation](#3d-aware-image-generation). Different from 3D reconstruction, which focuses on per-instance recovery (*i.e.*, the data already exists in the real world), 3D generation targets learning the real distribution and hence allows sampling new data.

Overall, the paper collection is organized as follows. *If you find some work is missing, feel free to raise an issue or create a pull request. We appreciate contributions in any form.*

- [3D Shape Generation](#3d-shape-generation)
  - [Point Cloud](#point-cloud)
  - [Voxel](#voxel)
  - [Mesh](#mesh)
  - [Implicit Function](#implicit-function)
  - [Parametric Surface](#parametric-surface)
  - [Primitive Shape](#primitive-shape)
  - [Hybrid Representation](#hybrid-representation)
  - [Program](#program)
- [3D-aware Image Generation](#3d-aware-image-generation)
  - [Point Cloud](#point-cloud-1)
  - [Voxel](#voxel-1)
  - [Depth](#depth)
  - [Implicit Function](#implicit-function-1)
  - [Hybrid Representation](#hybrid-representation-1)
- [3D Control of 2D Generative Models](#3d-control-of-2d-generative-models)

## 3D Shape Generation

We categorize the studies on 3D shape generation according to the representation used.

### Point Cloud

- Learning Representations and Generative Models for 3D Point Clouds <br>
  [ICML 2018](https://arxiv.org/abs/1707.02392) / [Code](https://github.com/optas/latent_3d_points)
- Multiresolution Tree Networks for 3D Point Cloud Processing <br>
  [ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/papers/Matheus_Gadelha_Multiresolution_Tree_Networks_ECCV_2018_paper.pdf) / [Code](https://github.com/matheusgadelha/MRTNet) / [Project Page](http://mgadelha.me/mrt/)
- 3D Point Cloud Generative Adversarial Network Based on Tree Structured Graph Convolutions <br>
  [ICCV 2019](https://arxiv.org/abs/1905.06292) / [Code](https://github.com/prajwalsingh/TreeGCN-GAN)
- Point Cloud GAN <br>
  [ICLR 2019](https://arxiv.org/abs/1810.05795) / [Code](https://github.com/chunliangli/Point-Cloud-GAN)
- Learning Localized Generative Models for 3D Point Clouds via Graph Convolution <br>
  [ICLR 2019](https://openreview.net/pdf?id=SJeXSo09FQ) / [Code](https://github.com/diegovalsesia/GraphCNN-GAN)
- PointFlow : 3D Point Cloud Generation with Continuous Normalizing Flows <br>
  [ICCV 2019](https://arxiv.org/abs/1906.12320) / [Code](https://github.com/stevenygd/PointFlow)
- Spectral-GANs for High-Resolution 3D Point-Cloud Generation <br>
  [IROS 2020](https://arxiv.org/abs/1912.01800) / [Code](https://github.com/samgregoost/Spectral-GAN)
- Progressive Point Cloud Deconvolution Generation Network <br>
  [ECCV 2020](https://arxiv.org/abs/2007.05361) / [Code](https://github.com/fpthink/PDGN)
- A Progressive Conditional Generative Adversarial Network for Generating Dense and Colored 3D Point Clouds <br>
  [3DV 2020](https://arxiv.org/abs/2010.05391) / [Code](https://github.com/robotic-vision-lab/Progressive-Conditional-Generative-Adversarial-Network)
- Adversarial Autoencoders for Generating 3D Point Clouds <br>
  [ICLR 2020](https://arxiv.org/abs/1811.07605) / [Code](https://github.com/MaciejZamorski/3d-AAE)
- Learning Gradient Fields for Shape Generation <br>
  [ECCV 2020](https://arxiv.org/abs/2008.06520) / [Code](https://github.com/RuojinCai/ShapeGF) / [Project Page](https://www.cs.cornell.edu/~ruojin/ShapeGF/)
- SoftFlow: Probabilistic Framework for Normalizing Flow on Manifolds <br>
  [NeurIPS 2020](https://arxiv.org/abs/2006.04604) / [Code](https://github.com/ANLGBOY/SoftFlow)
- Discrete Point Flow Networks for Efficient Point Cloud Generation <br>
  [ECCV 2020](https://arxiv.org/abs/2007.10170) / [Code](https://github.com/Regenerator/dpf-nets)
- Pointgrow: Autoregressively Learned Point Cloud Generation with Self-Attention <br>
  [WACV 2020](https://arxiv.org/abs/1810.05591) / [Code](https://github.com/syb7573330/PointGrow) / [Project Page](https://liuziwei7.github.io/projects/PointGrow)
- MRGAN: MultiRooted 3D Shape Generation with Unsupervised Part Disentanglement <br>
  [ICCVW 2021](https://arxiv.org/abs/2007.12944)
- SP-GAN: Sphere-Guided 3D Shape Generation and Manipulation <br>
  [SIGGRAPH 2021](https://arxiv.org/abs/2108.04476) / [Code](https://github.com/liruihui/sp-gan)
- Generative PointNet: Deep Energy-Based Learning on Unordered Point Sets for
3D Generation, Reconstruction and Classification <br>
  [CVPR 2021](https://arxiv.org/abs/2004.01301) / [Code](https://github.com/fei960922/GPointNet) / [Project Page](http://www.stat.ucla.edu/~jxie/GPointNet/)

### Voxel

- 3d Shapenets: A Deep Representation for Volumetric Shapes <br>
  [CVPR 2015](https://arxiv.org/abs/1406.5670) / [Code](https://github.com/zhirongw/3DShapeNets) / [Project Page](http://3dshapenets.cs.princeton.edu/)
- Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling <br>
  [NeurIPS 2016](https://arXiv.org/abs/1610.07584) / [Code](https://github.com/zck119/3dgan-release) / [Project Page](http://3dgan.csail.mit.edu/)
- Generative and Discriminative Voxel Modeling with Convolutional Neural Networks <br>
  [arXiv 2016](https://arxiv.org/abs/1608.04236) / [Code](https://github.com/ajbrock/Generative-and-Discriminative-Voxel-Modeling)
- Octree Generating Networks: Efficient Convolutional Architectures for High-Resolution 3D Outputs <br>
  [ICCV 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Tatarchenko_Octree_Generating_Networks_ICCV_2017_paper.pdf) / [Code](https://github.com/lmb-freiburg/ogn)
- SAGNet: Structure-aware Generative Network for 3D-Shape Modeling <br>
  [SIGGRAPH 2019](https://dilincv.github.io/papers/SAGNet_sig2019.pdf) / [Code](https://github.com/zhijieW94/SAGNet) / [Project Page](https://vcc.tech/research/2019/SAGnet/)
- Generalized Autoencoder for Volumetric Shape Generation <br>
  [CVPRW 2020](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w17/Guan_Generalized_Autoencoder_for_Volumetric_Shape_Generation_CVPRW_2020_paper.pdf) / [Code](https://github.com/IsaacGuan/3D-GAE)
- PQ-NET: A Generative Part Seq2Seq Network for 3D Shapes <br>
  [CVPR 2020](https://arxiv.org/abs/1911.10949) / [Code](https://github.com/ChrisWu1997/PQ-NET)
- Learning Part Generation and Assembly for Structure-Aware Shape Synthesis <br>
  [AAAI 2020](https://arxiv.org/abs/1906.06693)
- Generative VoxelNet: Learning Energy-Based Models for 3D Shape Synthesis and Analysis <br>
  [TPAMI 2020](http://www.stat.ucla.edu/~jxie/3DDescriptorNet/3DDescriptorNet_file/doc/3DDescriptorNet.pdf) / [Code](https://github.com/jianwen-xie/3DDescriptorNet) / [Project Page](http://www.stat.ucla.edu/~jxie/3DDescriptorNet/3DDescriptorNet.html)
- DLGAN: Depth-Preserving Latent Generative Adversarial Network for 3D Reconstruction <br>
  [TMM 2020](https://ieeexplore.ieee.org/document/9174748)
- DECOR-GAN: 3D Shape Detailization by Conditional Refinement <br>
  [CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_DECOR-GAN_3D_Shape_Detailization_by_Conditional_Refinement_CVPR_2021_paper.pdf) / [Code](https://github.com/czq142857/DECOR-GAN)
- Octree Transformer: Autoregressive 3D Shape Generation on Hierarchically Structured Sequences <br>
  [arXiv 2021](https://arxiv.org/abs/2111.12480)

### Mesh

- SDM-Net: Deep Generative Network for Structured Deformable Mesh <br>
  [SIGGRAPH Asia 2019](https://arxiv.org/abs/1908.04520) / [Project Page](http://geometrylearning.com/sdm-net/)
- PolyGen: An Autoregressive Generative Model of 3D Meshes <br>
  [ICML 2020](https://arxiv.org/abs/2002.10880) / [Code](https://github.com/deepmind/deepmind-research/tree/master/polygen)
- TM-NET: Deep Generative Networks for Textured Meshes <br>
  [TOG 2021](https://arxiv.org/abs/2010.06217) / [Code](https://github.com/IGLICT/TM-NET) / [Project Page](http://geometrylearning.com/TM-NET/)

### Implicit Function

- Learning Implicit Fields for Generative Shape Modeling <br>
  [CVPR 2019](https://arxiv.org/abs/1812.02822) / [Code](https://github.com/czq142857/implicit-decoder) / [Project Page](https://www.sfu.ca/~zhiqinc/imgan/Readme.html)
- Adversarial Generation of Continuous Implicit Shape Representations <br>
  [arXiv 2020](https://arxiv.org/abs/2002.00349) / [Code](https://github.com/marian42/shapegan)
- DualSDF: Semantic Shape Manipulation using a Two-Level Representation <br>
  [CVPR 2020](https://arxiv.org/abs/2004.02869) / [Code](https://github.com/zekunhao1995/DualSDF) / [Project Page](https://www.cs.cornell.edu/~hadarelor/dualsdf/)
- SurfGen: Adversarial 3D Shape Synthesis with Explicit Surface Discriminators <br>
  [ICCV 2021](https://arxiv.org/abs/2201.00112)
- 3D Shape Generation with Grid-Based Implicit Functions <br>
  [CVPR 2021](https://arxiv.org/abs/2107.10607)
- gDNA: Towards Generative Detailed Neural Avatars <br>
  [CVPR 2022](https://arxiv.org/abs/2201.04123) / [Code](https://github.com/xuchen-ethz/gdna) / [Project Page](https://xuchen-ethz.github.io/gdna/)
- Deformed Implicit Field: Modeling 3D shapes with Learned Dense Correspondence <br>
  [CVPR 2021](https://arxiv.org/abs/2011.13650) / [Code](https://github.com/microsoft/DIF-Net)

### Parametric Surface

- Multi-Chart Generative Surface Modeling <br>
  [SIGGRAPH Asia 2018](https://arxiv.org/abs/1806.02143) / [Code](https://github.com/helibenhamu/multichart3dgans)

### Primitive Shape

- Physically-Aware Generative Network for 3D Shape Modeling <br>
  [CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Mezghanni_Physically-Aware_Generative_Network_for_3D_Shape_Modeling_CVPR_2021_paper.html)

### Hybrid Representation

- Coupling Explicit and Implicit Surface Representations for Generative 3D Modeling <br>
  [ECCV 2020](https://arxiv.org/abs/2007.10294)
- Deep Marching Tetrahedra: a Hybrid Representation for High-Resolution 3D Shape Synthesis <br>
  [NeurIPS 2021](https://arxiv.org/abs/2111.04276) / [Project Page](https://nv-tlabs.github.io/DMTet/)

### Program

- ShapeAssembly: Learning to Generate Programs for 3D Shape Structure Synthesis <br>
  [SIGGRAPH Asia 2020](https://arxiv.org/abs/2009.08026) / [Code](https://github.com/rkjones4/ShapeAssembly) / [Project Page](https://rkjones4.github.io/shapeAssembly.html)

## 3D-aware Image Generation

We categorize the studies on 3D-aware image generation according to the representation used.

### Point Cloud

- Points2Pix: 3D Point-Cloud to Image Translation using conditional Generative Adversarial Networks <br>
  [arXiv 2019](https://arXiv.org/abs/1901.09280)

### Voxel

- Visual Object Networks: Image Generation with Disentangled 3D Representation <br>
  [NeurIPS 2018](https://arXiv.org/abs/1812.02725) / [Code](https://github.com/junyanz/VON) / [Project Page](http://von.csail.mit.edu/)
- HoloGAN: Unsupervised Learning of 3D representations from Natural Images <br>
  [ICCV 2019](https://arXiv.org/abs/1904.01326) / [Code](https://github.com/thunguyenphuoc/HoloGAN)
- BlockGAN: Learning 3D Object-aware Scene Representations from Unlabelled Images <br>
  [NeurIPS 2020](https://arXiv.org/abs/2002.08988) / [Code](https://github.com/thunguyenphuoc/BlockGAN)
- Towards a Neural Graphics Pipeline for Controllable Image Generation <br>
  [Computer Graphics Forum 2021](https://arXiv.org/abs/2006.10569) / [Project Page](http://geometry.cs.ucl.ac.uk/projects/2021/ngp/)

### Depth

- Generative Image Modeling using Style and Structure Adversarial Networks <br>
  [ECCV 2016](https://arXiv.org/abs/1603.05631) / [Code](https://github.com/xiaolonw/ss-gan)
- Geometric Image Synthesis <br>
  [ACCV 2018](https://arXiv.org/abs/1809.04696)
- RGBD-GAN: Unsupervised 3D Representation Learning From Natural Image Datasets via RGBD Image Synthesis <br>
  [ICLR 2020](https://arXiv.org/abs/1909.12573) / [Code](https://github.com/nogu-atsu/RGBD-GAN)
- 3D-Aware Indoor Scene Synthesis with Depth Priors <br>
  [arXiv 2022](https://arXiv.org/abs/2202.08553) / [Code](https://github.com/VivianSZF/depthgan) / [Project Page](https://vivianszf.github.io/depthgan/)

### Implicit Function

- GRAF: Generative Radiance Fields for 3D-Aware Image Synthesis <br>
  [NeurIPS 2020](https://arXiv.org/abs/2007.02442) / [Code](https://github.com/autonomousvision/graf) / [Project Page](https://autonomousvision.github.io/graf/)
- GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields <br>
  [CVPR 2021](https://arXiv.org/abs/2011.12100) / [Code](https://github.com/autonomousvision/giraffe) / [Project Page](https://m-niemeyer.github.io/project-pages/giraffe/index.html)
- pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis <br>
  [CVPR 2021](https://arXiv.org/abs/2012.00926) / [Code](https://github.com/marcoamonteiro/pi-GAN) / [Project Page](https://marcoamonteiro.github.io/pi-GAN-website/)
- Unconstrained Scene Generation with Locally Conditioned Radiance Fields <br>
  [ICCV 2021](https://arXiv.org/abs/2104.00670) / [Code](https://github.com/apple/ml-gsn) / [Project Page](https://apple.github.io/ml-gsn/)
- A Shading-Guided Generative Implicit Model for Shape-Accurate 3D-Aware Image Synthesis <br>
  [NeurIPS 2021](https://arXiv.org/abs/2110.15678) / [Code](https://github.com/xingangpan/shadegan) / [Project Page](https://xingangpan.github.io/projects/ShadeGAN.html)
- Campari: Camera-aware Decomposed Generative Neural Radiance Fields <br>
  [3DV 2021](https://arXiv.org/abs/2103.17269) / [Code](https://github.com/autonomousvision/campari)
- CIPS-3D: A 3D-Aware Generator of GANs Based on Conditionally-Independent Pixel Synthesis <br>
  [arXiv 2021](https://arXiv.org/abs/2110.09788) / [Code](https://github.com/PeterouZh/CIPS-3D)
- GANcraft: Unsupervised 3D Neural Rendering of Minecraft Worlds <br>
  [ICCV 2021](https://arXiv.org/abs/2104.07659) / [Code](https://github.com/NVlabs/GANcraft) / [Project Page](https://nvlabs.github.io/GANcraft/)
- Generative Occupancy Fields for 3D Surface-Aware Image Synthesis <br>
  [NeurIPS 2021](https://arXiv.org/abs/2111.00969) / [Code](https://github.com/SheldonTsui/GOF_NeurIPS2021) / [Project Page](https://sheldontsui.github.io/projects/GOF)
- 3D-Aware Semantic-Guided Generative Model for Human Synthesis <br>
  [arXiv 2021](https://arXiv.org/abs/2112.01422)
- FENeRF: Face Editing in Neural Radiance Fields <br>
  [CVPR 2022](https://arXiv.org/abs/2111.15490)
- StyleNeRF: A Style-Based 3D-Aware Generator for High-resolution Image Synthesis <br>
  [ICLR 2022](https://arXiv.org/abs/2110.08985) / [Code](https://github.com/facebookresearch/StyleNeRF) / [Project Page](http://jiataogu.me/style_nerf/)
- StyleSDF: High-Resolution 3D-Consistent Image and Geometry Generation <br>
  [CVPR 2022](https://arXiv.org/abs/2112.11427) / [Code](https://github.com/royorel/StyleSDF) / [Project Page](https://stylesdf.github.io/)
- GRAM: Generative Radiance Manifolds for 3D-Aware Image Generation <br>
  [CVPR 2022](https://arXiv.org/abs/2112.08867) / [Project Page](https://yudeng.github.io/GRAM/)
- A Generative Model for 3D Face Synthesis with HDRI Relighting <br>
  [arXiv 2022](https://arxiv.org/abs/2201.04873)
- Pix2NeRF: Unsupervised Conditional π-GAN for Single Image to Neural Radiance Fields Translation <br>
  [CVPR 2022](https://arXiv.org/abs/2202.13162)
- 3D-GIF: 3D-Controllable Object Generation via Implicit Factorized Representations <br>
  [arXiv 2022](https://arxiv.org/abs/2203.06457)
- Generating Videos with Dynamics-aware Implicit Generative Adversarial Networks <br>
  [ICLR 2022](https://openreview.net/forum?id=Czsdv-S4-w9) / [Code](https://github.com/sihyun-yu/digan) / [Project Page](https://sihyun-yu.github.io/digan/)

### Hybrid Representation

- 3D-aware Image Synthesis via Learning Structural and Textural Representations <br>
  [CVPR 2022](https://arXiv.org/abs/2112.10759) / [Code](https://github.com/genforce/volumegan) / [Project Page](https://genforce.github.io/volumegan/)
- Efficient Geometry-aware 3D Generative Adversarial Networks <br>
  [CVPR 2022](https://arXiv.org/abs/2112.07945) / [Code](https://github.com/NVlabs/eg3d) / [Project Page](https://matthew-a-chan.github.io/EG3D/)

## 3D Control of 2D Generative Models

Besides explicitly learning a 3D generative model, there are also some attempts working on the 3D controllability of 2D models.

- Interpreting the latent space of gans for semantic face editing <br>
  [CVPR 2020](https://arxiv.org/abs/1907.10786) / [Code](https://github.com/genforce/interfacegan) / [Project Page](https://genforce.github.io/interfacegan/)
- StyleRig: Rigging StyleGAN for 3D Control over Portrait Images <br>
  [CVPR 2020](https://arxiv.org/abs/2004.00121) / [Project Page](https://vcai.mpi-inf.mpg.de/projects/StyleRig/)
- Disentangled and Controllable Face Image Generation via 3D Imitative-Contrastive Learning <br>
  [CVPR 2020](https://arxiv.org/abs/2004.11660) / [Code](https://github.com/microsoft/DiscoFaceGAN)
- Towards Unsupervised Learning of Generative Models for 3D Controllable Image Synthesis <br>
  [CVPR 2020](https://avg.is.tuebingen.mpg.de/publications/liao2020cvpr) / [Code](https://github.com/autonomousvision/controllable_image_synthesis)
- DiscoFaceGAN: Disentangled and Controllable Face Image Generation via 3D Imitative-Contrastive Learning <br>
  [CVPR 2020](https://arxiv.org/pdf/2004.11660.pdf) / [Code](https://github.com/microsoft/DiscoFaceGAN)
- FreeStyleGAN: Free-view Editable Portrait Rendering with the Camera Manifold <br>
  [SIGGRAPH Asia 2021](https://arxiv.org/abs/2109.09378) / [Code](https://gitlab.inria.fr/fungraph/freestylegan) / [Project Page](https://repo-sam.inria.fr/fungraph/freestylegan/)
- Do 2D GANs Know 3D Shape? Unsupervised 3D Shape Reconstruction from 2D Image GANs <br>
  [ICLR 2021](https://arxiv.org/abs/2011.00844) / [Code](https://github.com/XingangPan/GAN2Shape) / [Project Page](https://xingangpan.github.io/projects/GAN2Shape.html)
- Semantic Hierarchy Emerges in Deep Generative Representations for Scene Synthesis <br>
  [IJCV 2021](https://arxiv.org/abs/1911.09267) / [Code](https://github.com/genforce/higan) / [Project Page](https://genforce.github.io/higan/)
- Cross-Domain and Disentangled Face Manipulation with 3D Guidance <br>
  [TVCG 2021](https://arxiv.org/abs/2104.11228) / [Code](https://github.com/cassiePython/cddfm3d) / [Project Page](https://cassiepython.github.io/cddfm3d/index)
- Image GANs meet Differentiable Rendering for Inverse Graphics and Interpretable 3D Neural Rendering <br>
  [ICLR 2021](https://arxiv.org/abs/2010.09125) / [Project Page](https://nv-tlabs.github.io/GANverse3D/)
- Pose with Style: Detail-Preserving Pose-Guided Image Synthesis with Conditional StyleGAN <br>
  [TOG 2021](https://arxiv.org/abs/2109.06166) / [Code](https://github.com/BadourAlBahar/pose-with-style) / [Project Page](https://pose-with-style.github.io/)
- SofGAN: A Portrait Image Generator with Dynamic Styling <br>
  [TOG 2021](https://arxiv.org/abs/2007.03780) / [Code](https://github.com/apchenstu/sofgan) / [Project Page](https://apchenstu.github.io/sofgan/)
- MOST-GAN: 3D Morphable StyleGAN for Disentangled Face Image Manipulation <br>
  [AAAI 2022](https://arxiv.org/abs/2111.01048)
