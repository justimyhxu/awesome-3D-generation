# Awesome 3D Generation

## Papers

#### Point Clouds

##### GAN model

- Learning representations and generative models for 3d point clouds, ICML 2018
- 3d point cloud generative adversarial network based on tree structured graph convolutions, ICCV 2019
- Point cloud gan, ICLR 2019
- Spectral-gans for high-resolution 3d point-cloud generation, iros 2020
- Progressive point cloud deconvolution generation network, ECCV 2020
- MRGAN: MultiRooted 3D Shape Generation with Unsupervised Part Disentanglement, ICCVw 2021
- A progressive conditional generative adversarial network for generating dense and colored 3D point clouds, 3dv 2020
- SP-GAN: Sphere-guided 3D shape generation and manipulation, SIGGRAPH 2021
- Multimodal shape completion via conditional generative adversarial networks, ECCV 2020

##### Flow-based model

- Learning gradient fields for shape generation, ECCV 2020
- PointFlow : 3D Point Cloud Generation with Continuous Normalizing Flows, ICCV 2019
- SoftFlow: Probabilistic framework for normalizing flow on manifolds, NeurIPS 2020
- Discrete point flow networks for efficient point cloud generation, ECCV 2020

##### Auto-regressive model

- Pointgrow: Autoregressively learned point cloud generation with self-attention, wacv 2020

##### 未分类

- Learning to reconstruct shapes from unseen classes, nips 2018
- Multiresolution tree networks for 3d point cloud processing, ECCV 2018
- Learning localized generative models for 3d point clouds via graph convolution, ICLR 2019
- A point set generation network for 3d object reconstruction from a single image, CVPR 2017
- Adversarial autoencoders for generating 3d point clouds, ICLR 2020
- Learning to generate dense point clouds with textures on multiple categories, wacv 2021
- OctNet: Learning Deep 3D Representations at High Resolutions, CVPR 2017
- Generating 3D Adversarial Point Clouds, CVPR 2019
- Neural Style Transfer for Point Clouds, arxiv 2019

#### Voxel Grids

- [Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling](https://arxiv.org/abs/1610.07584), NeurIPS, 2016
- Generalized Autoencoder for Volumetric Shape Generation, CVPRw 2020
- PQ-NET: A Generative Part Seq2Seq Network for 3D Shapes, CVPR 2020
- DECOR-GAN: 3D Shape Detailization by Conditional Refinement, CVPR 2021
- 3d shapenets: A deep representation for volumetric shapes,  CVPR 2015
- Learning part generation and assembly for structure-aware shape synthesis, aaai 2020
- Generative VoxelNet: learning energy-based models for 3D shape synthesis and analysis, tpami 2020
- Octree Transformer: Autoregressive 3D Shape Generation on Hierarchically Structured Sequences, arxiv 2021
- DLGAN: Depth-Preserving Latent Generative Adversarial Network for 3D Reconstruction, tmm 2020
- Shape completion using 3d-encoder-predictor cnns and shape synthesis, CVPR 2017
- Scancomplete: Large-scale scene completion and semantic segmentation for 3d scans, CVPR 2018
- Generative and discriminative voxel modeling with convolutional neural networks, arxiv 2016
- Octree generating networks: Efficient convolutional architectures for high-resolution 3d outputs, ICCV 2017
- Deep octree-based cnns with output-guided skip connections for 3d shape and scene completion, CVPR 2020

#### Mesh

- Sdm-net: Deep generative network for structured deformable mesh, siggraph asia 2019

#### Implicit function

- Learning Implicit Fields for Generative Shape Modeling, CVPR 2019
- Adversarial generation of continuous implicit shape representations, arxiv 2020
- DualSDF: Semantic Shape Manipulation using a Two-Level Representation, CVPR 2020
- SurfGen: Adversarial 3D Shape Synthesis with Explicit Surface Discriminators, ICCV 2021
- 3d shape generation with grid-based implicit functions, CVPR 2021
- gDNA: Towards Generative Detailed Neural Avatars, CVPR 2022
- Deformed implicit field: Modeling 3d shapes with learned dense correspondence, CVPR 2021

#### Parametric surface

- Multi-chart generative surface modeling, SIGGRAPH Asia 2018

#### Primitive shapes

- Physically-aware generative network for 3d shape modeling, CVPR 2021

#### Hybrid representation

- Deep Marching Tetrahedra: a Hybrid Representation for High-Resolution 3D Shape Synthesis, NeurIPS 2021
- Coupling explicit and implicit surface representations for generative 3d modeling, ECCV 2020

#### Program

- Shapeassembly: Learning to generate programs for 3d shape structure synthesis, siggraph asia 2020

#### 3D-aware image synthesis

##### Explicit representation

- [HoloGAN: Unsupervised Learning of 3D representations from Natural Images](https://arxiv.org/abs/1904.01326), ICCV 2019 | [Code](https://github.com/thunguyenphuoc/HoloGAN)
- [Points2Pix: 3D Point-Cloud to Image Translation using conditional Generative Adversarial Networks](https://arxiv.org/abs/1901.09280), arxiv 2019

##### StyleGAN

- [Generative Image Modeling using Style and Structure Adversarial Networks](https://arxiv.org/abs/1603.05631), ECCV 2016 | [Code](https://github.com/xiaolonw/ss-gan)
- [Geometric Image Synthesis](https://arxiv.org/abs/1809.04696), ACCV 2018
- [BlockGAN: Learning 3D Object-aware Scene Representations from Unlabelled Images](https://arxiv.org/abs/2002.08988), NeurIPS 2020 | [Code](https://github.com/thunguyenphuoc/BlockGAN)
- [RGBD-GAN: Unsupervised 3D Representation Learning From Natural Image Datasets via RGBD Image Synthesis](https://arxiv.org/abs/1909.12573), ICLR 2020 | [Code](https://github.com/nogu-atsu/RGBD-GAN)
- [Towards a Neural Graphics Pipeline for Controllable Image Generation](https://arxiv.org/abs/2006.10569), Comput. Graph. Forum, 2021
- [3D-Aware Indoor Scene Synthesis with Depth Priors](https://arxiv.org/abs/2202.08553), arxiv 2022 | [Code](https://github.com/VivianSZF/depthgan)

##### Implicit function

- [GRAF: Generative Radiance Fields for 3D-Aware Image Synthesis](https://arxiv.org/abs/2007.02442), NeurIPS 2020 | [Code](https://github.com/autonomousvision/graf)
- [GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields](https://arxiv.org/abs/2011.12100), CVPR 2021 | [Code](https://github.com/autonomousvision/giraffe)
- [pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis](https://arxiv.org/abs/2012.00926), CVPR 2021 | [Code](https://github.com/marcoamonteiro/pi-GAN)
- [Unconstrained Scene Generation with Locally Conditioned Radiance Fields](https://arxiv.org/abs/2104.00670), ICCV 2021 | [Code](https://github.com/apple/ml-gsn)
- [A Shading-Guided Generative Implicit Model for Shape-Accurate 3D-Aware Image Synthesis](https://arxiv.org/abs/2110.15678), NeurIPS 2021 | [Code](https://github.com/xingangpan/shadegan) 
- [StyleNeRF: A Style-based 3D-Aware Generator for High-resolution Image Synthesis](https://arxiv.org/abs/2110.08985), ICLR 2022 | [Code](https://github.com/facebookresearch/StyleNeRF)
- [Efficient Geometry-aware 3D Generative Adversarial Networks](https://arxiv.org/abs/2112.07945), CVPR 2022 | [Code](https://github.com/NVlabs/eg3d)
- [3D-aware Image Synthesis via Learning Structural and Textural Representations](https://arxiv.org/abs/2112.10759), CVPR 2022 | [Code](https://github.com/genforce/volumegan)
- [StyleSDF: High-Resolution 3D-Consistent Image and Geometry Generation](https://arxiv.org/abs/2112.11427), CVPR 2022 | [Code](https://github.com/royorel/StyleSDF)
- [GRAM: Generative Radiance Manifolds for 3D-Aware Image Generation](https://arxiv.org/abs/2112.08867), CVPR 2022
- [Campari: Camera-aware Decomposed Generative Neural Radiance Fields](https://arxiv.org/abs/2103.17269), 3DV 2021
- [CIPS-3D: A 3D-Aware Generator of GANs Based on Conditionally-Independent Pixel Synthesis](https://arxiv.org/abs/2110.09788), arxiv 2021 | [Code](https://github.com/PeterouZh/CIPS-3D)
- [GANcraft: Unsupervised 3D Neural Rendering of Minecraft Worlds](https://arxiv.org/abs/2104.07659), ICCV 2021 | [Code](https://github.com/NVlabs/GANcraft)
- [Generative Occupancy Fields for 3D Surface-Aware Image Synthesis](https://arxiv.org/abs/2111.00969), NeurIPS 2021 | [Code](https://github.com/SheldonTsui/GOF_NeurIPS2021)
- [3D-Aware Semantic-Guided Generative Model for Human Synthesis](https://arxiv.org/abs/2112.01422), arxiv 2021

##### Per-scene optimization with a discriminator

- [Putting NeRF on a Diet: Semantically Consistent Few-Shot View Synthesis](https://arxiv.org/abs/2104.00677), ICCV 2021 | [Code](https://github.com/codestella/putting-nerf-on-a-diet)
- [GNeRF: GAN-based Neural Radiance Field without Posed Camera](https://arxiv.org/abs/2103.15606), ICCV 2021 | [Code](https://github.com/MQ66/gnerf)
- [RegNeRF: Regularizing Neural Radiance Fields for View Synthesis from Sparse Inputs](https://arxiv.org/abs/2112.00724), CVPR 2022 | [Code](https://github.com/google-research/google-research/tree/master/regnerf)

##### 3D editing

###### StyleGAN

- [Learning Realistic Human Reposing using Cyclic Self-Supervision with 3D Shape, Pose, and Appearance Consistency](https://arxiv.org/abs/2110.05458), ICCV 2021

###### Implicit function

- [FENeRF: Face Editing in Neural Radiance Fields](https://arxiv.org/abs/2111.15490), arxiv 2021
- [Pix2NeRF: Unsupervised Conditional π-GAN for Single Image to Neural Radiance Fields Translation](https://arxiv.org/abs/2202.13162), CVPR 2022

##### 未分类

- [Visual Object Networks: Image Generation with Disentangled 3D Representation](https://arxiv.org/abs/1812.02725), NeurIPS 2018
