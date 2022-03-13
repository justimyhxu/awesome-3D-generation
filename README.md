# Awesome 3D Generation

## Point Cloud Generation

### GAN-based Model

- Learning representations and generative models for 3d point clouds <br>
  [ICML 2018](https://arxiv.org/abs/1707.02392) / [Code](https://github.com/optas/latent_3d_points)
- 3d point cloud generative adversarial network based on tree structured graph convolutions <br>
  [ICCV 2019](https://arxiv.org/abs/1905.06292) / [Code](https://github.com/prajwalsingh/TreeGCN-GAN)
- Point cloud gan <br>
  [ICLR 2019](https://arxiv.org/abs/1810.05795) / [Code](https://github.com/chunliangli/Point-Cloud-GAN)
- Spectral-gans for high-resolution 3d point-cloud generation <br>
  [IROS 2020](https://arxiv.org/abs/1912.01800) / [Code](https://github.com/samgregoost/Spectral-GAN)
- Progressive point cloud deconvolution generation network <br>
  [ECCV 2020](https://arxiv.org/abs/2007.05361) / [Code](https://github.com/fpthink/PDGN)
- MRGAN: MultiRooted 3D Shape Generation with Unsupervised Part Disentanglement <br>
  [ICCVW 2021](https://arxiv.org/abs/2007.12944)
- A progressive conditional generative adversarial network for generating dense and colored 3D point clouds <br>
  [3DV 2020](https://arxiv.org/abs/2010.05391) / [Code](https://github.com/robotic-vision-lab/Progressive-Conditional-Generative-Adversarial-Network)
- SP-GAN: Sphere-guided 3D shape generation and manipulation <br>
  [SIGGRAPH 2021](https://arxiv.org/abs/2108.04476) / [Code](https://github.com/liruihui/sp-gan)
- Multimodal shape completion via conditional generative adversarial networks <br>
  [ECCV 2020](https://arxiv.org/abs/2003.07717) / [Code](https://github.com/ChrisWu1997/Multimodal-Shape-Completion) / [Project Page](https://chriswu1997.github.io/files/multimodal-pc/index.html)
- Learning localized generative models for 3d point clouds via graph convolution <br>
  [ICLR 2019](https://openreview.net/pdf?id=SJeXSo09FQ) / [Code](https://github.com/diegovalsesia/GraphCNN-GAN)

### Flow-based Model

- Learning gradient fields for shape generation <br>
  [ECCV 2020](https://arxiv.org/abs/2008.06520) / [Code](https://github.com/RuojinCai/ShapeGF) / [Project Page](https://www.cs.cornell.edu/~ruojin/ShapeGF/)
- PointFlow : 3D Point Cloud Generation with Continuous Normalizing Flows <br>
  [ICCV 2019](https://arxiv.org/abs/1906.12320) / [Code](https://github.com/stevenygd/PointFlow)
- SoftFlow: Probabilistic framework for normalizing flow on manifolds <br>
  [NeurIPS 2020](https://arxiv.org/abs/2006.04604) / [Code](https://github.com/ANLGBOY/SoftFlow)
- Discrete point flow networks for efficient point cloud generation <br>
  [ECCV 2020](https://arxiv.org/abs/2007.10170) / [Code](https://github.com/Regenerator/dpf-nets)

### Auto-regressive Model

- Pointgrow: Autoregressively learned point cloud generation with self-attention <br>
  [WACV 2020](https://arxiv.org/abs/1810.05591) / [Code](https://github.com/syb7573330/PointGrow) / [Project Page](https://liuziwei7.github.io/projects/PointGrow)

### VAE

- Multiresolution tree networks for 3d point cloud processing <br>
  [ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/papers/Matheus_Gadelha_Multiresolution_Tree_Networks_ECCV_2018_paper.pdf) / [Code](https://github.com/matheusgadelha/MRTNet) / [Project Page](http://mgadelha.me/mrt/)
- Adversarial autoencoders for generating 3d point clouds <br>
  [ICLR 2020](https://arxiv.org/abs/1811.07605) / [Code](https://github.com/MaciejZamorski/3d-AAE)

## Voxel Grids

- Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling <br>
  [NeurIPS 2016](https://arXiv.org/abs/1610.07584) / [Code](https://github.com/zck119/3dgan-release) / [Project Page](http://3dgan.csail.mit.edu/)
- Generalized Autoencoder for Volumetric Shape Generation <br>
  [CVPRW 2020](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w17/Guan_Generalized_Autoencoder_for_Volumetric_Shape_Generation_CVPRW_2020_paper.pdf) / [Code](https://github.com/IsaacGuan/3D-GAE)
- PQ-NET: A Generative Part Seq2Seq Network for 3D Shapes <br>
  [CVPR 2020](https://arxiv.org/abs/1911.10949) / [Code](https://github.com/ChrisWu1997/PQ-NET)
- DECOR-GAN: 3D Shape Detailization by Conditional Refinement <br>
  [CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_DECOR-GAN_3D_Shape_Detailization_by_Conditional_Refinement_CVPR_2021_paper.pdf) / [Code](https://github.com/czq142857/DECOR-GAN)
- 3d shapenets: A deep representation for volumetric shapes <br>
  [CVPR 2015](https://arxiv.org/abs/1406.5670) / [Code](https://github.com/zhirongw/3DShapeNets) / [Project Page](http://3dshapenets.cs.princeton.edu/)
- Learning part generation and assembly for structure-aware shape synthesis <br>
  [AAAI 2020](https://arxiv.org/abs/1906.06693)
- SAGNet: Structure-aware Generative Network for 3D-Shape Modeling <br>
  [SIGGRAPH 2019](https://dilincv.github.io/papers/SAGNet_sig2019.pdf) / [Code](https://github.com/zhijieW94/SAGNet) / [Project Page](https://vcc.tech/research/2019/SAGnet/)
- Generative VoxelNet: learning energy-based models for 3D shape synthesis and analysis <br>
  [TPAMI 2020](http://www.stat.ucla.edu/~jxie/3DDescriptorNet/3DDescriptorNet_file/doc/3DDescriptorNet.pdf) / [Code](https://github.com/jianwen-xie/3DDescriptorNet) / [Project Page](http://www.stat.ucla.edu/~jxie/3DDescriptorNet/3DDescriptorNet.html)
- Octree Transformer: Autoregressive 3D Shape Generation on Hierarchically Structured Sequences <br>
  [arXiv 2021](https://arxiv.org/abs/2111.12480)
- DLGAN: Depth-Preserving Latent Generative Adversarial Network for 3D Reconstruction <br>
  [TMM 2020](https://ieeexplore.ieee.org/document/9174748)
- Shape completion using 3d-encoder-predictor cnns and shape synthesis <br>
  [CVPR 2017](https://arxiv.org/abs/1612.00101) / [Code](https://github.com/angeladai/cnncomplete) / [Project Page](http://graphics.stanford.edu/projects/cnncomplete/)
- Scancomplete: Large-scale scene completion and semantic segmentation for 3d scans <br>
  [CVPR 2018](https://arxiv.org/abs/1712.10215) / [Code](https://github.com/angeladai/ScanComplete)
- Generative and discriminative voxel modeling with convolutional neural networks <br>
  [arXiv 2016](https://arxiv.org/abs/1608.04236) / [Code](https://github.com/ajbrock/Generative-and-Discriminative-Voxel-Modeling)
- Octree generating networks: Efficient convolutional architectures for high-resolution 3d outputs <br>
  [ICCV 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Tatarchenko_Octree_Generating_Networks_ICCV_2017_paper.pdf) / [Code](https://github.com/lmb-freiburg/ogn)
- Deep octree-based cnns with output-guided skip connections for 3d shape and scene completion <br>
  [CVPR 2020](https://arxiv.org/abs/2006.03762) / [Code](https://github.com/microsoft/O-CNN)

## Mesh

- Sdm-net: Deep generative network for structured deformable mesh <br>
  [SIGGRAPH Asia 2019](https://arxiv.org/abs/1908.04520) / [Project Page](http://geometrylearning.com/sdm-net/)

## Implicit function

- Learning Implicit Fields for Generative Shape Modeling <br>
  [CVPR 2019](https://arxiv.org/abs/1812.02822) / [Code](https://github.com/czq142857/implicit-decoder) / [Project Page](https://www.sfu.ca/~zhiqinc/imgan/Readme.html)
- Adversarial generation of continuous implicit shape representations <br>
  [arXiv 2020](https://arxiv.org/abs/2002.00349) / [Code](https://github.com/marian42/shapegan)
- DualSDF: Semantic Shape Manipulation using a Two-Level Representation <br>
  [CVPR 2020](https://arxiv.org/abs/2004.02869) / [Code](https://github.com/zekunhao1995/DualSDF) / [Project Page](https://www.cs.cornell.edu/~hadarelor/dualsdf/)
- SurfGen: Adversarial 3D Shape Synthesis with Explicit Surface Discriminators <br>
  [ICCV 2021](https://arxiv.org/abs/2201.00112)
- 3d shape generation with grid-based implicit functions <br>
  [CVPR 2021](https://arxiv.org/abs/2107.10607)
- gDNA: Towards Generative Detailed Neural Avatars <br>
  [CVPR 2022](https://arxiv.org/abs/2201.04123) / [Code](https://github.com/xuchen-ethz/gdna) / [Project Page](https://xuchen-ethz.github.io/gdna/)
- Deformed implicit field: Modeling 3d shapes with learned dense correspondence <br>
  [CVPR 2021](https://arxiv.org/abs/2011.13650) / [Code](https://github.com/microsoft/DIF-Net)

## Parametric surface

- Multi-chart generative surface modeling <br>
  [SIGGRAPH Asia 2018](https://arxiv.org/abs/1806.02143) / [Code](https://github.com/helibenhamu/multichart3dgans)

## Primitive shapes

- Physically-aware generative network for 3d shape modeling <br>
  [CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Mezghanni_Physically-Aware_Generative_Network_for_3D_Shape_Modeling_CVPR_2021_paper.html)

## Hybrid representation

- Deep Marching Tetrahedra: a Hybrid Representation for High-Resolution 3D Shape Synthesis <br>
  [NeurIPS 2021](https://arxiv.org/abs/2111.04276) / [Project Page](https://nv-tlabs.github.io/DMTet/)
- Coupling explicit and implicit surface representations for generative 3d modeling <br>
  [ECCV 2020](https://arxiv.org/abs/2007.10294)

## Program

- Shapeassembly: Learning to generate programs for 3d shape structure synthesis <br>
  [SIGGRAPH Asia 2020](https://arxiv.org/abs/2009.08026) / [Code](https://github.com/rkjones4/ShapeAssembly) / [Project Page](https://rkjones4.github.io/shapeAssembly.html)

## 3D-aware Image Synthesis

### Explicit representation

- HoloGAN: Unsupervised Learning of 3D representations from Natural Images <br>
  [ICCV 2019](https://arXiv.org/abs/1904.01326) / [Code](https://github.com/thunguyenphuoc/HoloGAN)
- Points2Pix: 3D Point-Cloud to Image Translation using conditional Generative Adversarial Networks <br>
  [arXiv 2019](https://arXiv.org/abs/1901.09280)
- BlockGAN: Learning 3D Object-aware Scene Representations from Unlabelled Images <br>
  [NeurIPS 2020](https://arXiv.org/abs/2002.08988) / [Code](https://github.com/thunguyenphuoc/BlockGAN)
- Visual Object Networks: Image Generation with Disentangled 3D Representation <br>
  [NeurIPS 2018](https://arXiv.org/abs/1812.02725) / [Code](https://github.com/junyanz/VON) / [Project Page](http://von.csail.mit.edu/)

### Image-space rendering

- Generative Image Modeling using Style and Structure Adversarial Networks <br>
  [ECCV 2016](https://arXiv.org/abs/1603.05631) / [Code](https://github.com/xiaolonw/ss-gan)
- Geometric Image Synthesis <br>
  [ACCV 2018](https://arXiv.org/abs/1809.04696)
- RGBD-GAN: Unsupervised 3D Representation Learning From Natural Image Datasets via RGBD Image Synthesis <br>
  [ICLR 2020](https://arXiv.org/abs/1909.12573) / [Code](https://github.com/nogu-atsu/RGBD-GAN)
- Towards a Neural Graphics Pipeline for Controllable Image Generation <br>
  [Computer Graphics Forum 2021](https://arXiv.org/abs/2006.10569) / [Project Page](http://geometry.cs.ucl.ac.uk/projects/2021/ngp/)
- 3D-Aware Indoor Scene Synthesis with Depth Priors <br>
  [arXiv 2022](https://arXiv.org/abs/2202.08553) / [Code](https://github.com/VivianSZF/depthgan) / [Project Page](https://vivianszf.github.io/depthgan/)
- M3D-VTON: A Monocular-to-3D Virtual Try-On Network <br>
  [ICCV 2021](https://arxiv.org/abs/2108.05126) / [Code](https://github.com/fyviezhao/m3d-vton)
- FreeStyleGAN: Free-view Editable Portrait Rendering with the Camera Manifold <br>
  [SIGGRAPH Asia 2021](https://arxiv.org/abs/2109.09378) / [Code](https://gitlab.inria.fr/fungraph/freestylegan) / [Project Page](https://repo-sam.inria.fr/fungraph/freestylegan/)
- Do 2D GANs Know 3D Shape? Unsupervised 3D Shape Reconstruction from 2D Image GANs <br>
  [ICLR](https://arxiv.org/pdf/2011.00844.pdf) / [Code](https://github.com/XingangPan/GAN2Shape) / [Project Page](https://xingangpan.github.io/projects/GAN2Shape.html)
- Interpreting the latent space of gans for semantic face editing <br>
  [CVPR 2020](https://arxiv.org/abs/1907.10786) / [Code](https://github.com/genforce/interfacegan) / [Project Page](https://genforce.github.io/interfacegan/)
- Closed-form factorization of latent semantics in gans <br>
  [CVPR 2021](https://arxiv.org/abs/2007.06600) / [Code](https://github.com/genforce/sefa) / [Project Page](https://genforce.github.io/sefa/)
- GANSpace: Discovering Interpretable GAN Controls <br>
  [NeurIPS 2020](https://arxiv.org/abs/2004.02546) / [Code](https://github.com/harskish/ganspace)
- StyleRig: Rigging StyleGAN for 3D Control over Portrait Images <br>
  [CVPR 2020](https://arxiv.org/abs/2004.00121) / [Project Page](https://vcai.mpi-inf.mpg.de/projects/StyleRig/)
- Disentangled and Controllable Face Image Generation via 3D Imitative-Contrastive Learning <br>
  [CVPR 2020](https://arxiv.org/abs/2004.11660) / [Code](https://github.com/microsoft/DiscoFaceGAN)
- Rotate-and-Render: Unsupervised Photorealistic Face Rotation from Single-View Images <br>
  [CVPR 2020](https://arxiv.org/abs/2003.08124) / [Code](https://github.com/Hangz-nju-cuhk/Rotate-and-Render)
- A Style-Based Generator Architecture for Generative Adversarial Networks <br>
  [CVPR 2019](https://arxiv.org/abs/1812.04948) / [Code](https://github.com/NVlabs/stylegan)
- Analyzing and Improving the Image Quality of StyleGAN <br>
  [CVPR 2020](https://arxiv.org/abs/1912.04958) / [Code](https://github.com/NVlabs/stylegan2)
- Alias-Free Generative Adversarial Networks <br>
  [NeurIPS 2021](https://arxiv.org/abs/2106.12423) / [Code](https://github.com/NVlabs/alias-free-gan) / [Project Page](https://nvlabs.github.io/stylegan3/)
- Disentangled Controls for StyleGAN Image Generation <br>
  [CVPR 2021](https://arxiv.org/abs/2011.12799) / [Code](https://github.com/betterze/StyleSpace)
- Pose with Style: Detail-Preserving Pose-Guided Image Synthesis with Conditional StyleGAN <br>
  [TOG 2021](https://arxiv.org/abs/2109.06166) / [Code](https://github.com/BadourAlBahar/pose-with-style) / [Project Page](https://pose-with-style.github.io/)
- Cross-Domain and Disentangled Face Manipulation with 3D Guidance <br>
  [TVCG 2021](https://arxiv.org/abs/2104.11228) / [Code](https://github.com/cassiePython/cddfm3d) / [Project Page](https://cassiepython.github.io/cddfm3d/index)
- Image GANs meet Differentiable Rendering for Inverse Graphics and Interpretable 3D Neural Rendering <br>
  [ICLR 2021](https://arxiv.org/abs/2010.09125) / [Project Page](https://nv-tlabs.github.io/GANverse3D/)

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
- StyleNeRF: A Style-based 3D-Aware Generator for High-resolution Image Synthesis <br>
  [ICLR 2022](https://arXiv.org/abs/2110.08985) / [Code](https://github.com/facebookresearch/StyleNeRF) / [Project Page](http://jiataogu.me/style_nerf/)
- Efficient Geometry-aware 3D Generative Adversarial Networks <br>
  [CVPR 2022](https://arXiv.org/abs/2112.07945) / [Code](https://github.com/NVlabs/eg3d) / [Project Page](https://matthew-a-chan.github.io/EG3D/)
- 3D-aware Image Synthesis via Learning Structural and Textural Representations <br>
  [CVPR 2022](https://arXiv.org/abs/2112.10759) / [Code](https://github.com/genforce/volumegan) / [Project Page](https://genforce.github.io/volumegan/)
- StyleSDF: High-Resolution 3D-Consistent Image and Geometry Generation <br>
  [CVPR 2022](https://arXiv.org/abs/2112.11427) / [Code](https://github.com/royorel/StyleSDF) / [Project Page](https://stylesdf.github.io/)
- GRAM: Generative Radiance Manifolds for 3D-Aware Image Generation <br>
  [CVPR 2022](https://arXiv.org/abs/2112.08867) / [Project Page](https://yudeng.github.io/GRAM/)
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

### Per-scene optimization with a discriminator

- Putting NeRF on a Diet: Semantically Consistent Few-Shot View Synthesis <br>
  [ICCV 2021](https://arXiv.org/abs/2104.00677) / [Code](https://github.com/codestella/putting-nerf-on-a-diet) / [Project Page](https://ajayj.com/dietnerf/)
- GNeRF: GAN-based Neural Radiance Field without Posed Camera <br>
  [ICCV 2021](https://arXiv.org/abs/2103.15606) / [Code](https://github.com/MQ66/gnerf)
- RegNeRF: Regularizing Neural Radiance Fields for View Synthesis from Sparse Inputs <br>
  [CVPR 2022](https://arXiv.org/abs/2112.00724) / [Code](https://github.com/google-research/google-research/tree/master/regnerf) / [Project Page](https://m-niemeyer.github.io/regnerf/)

### 3D editing

- Learning Realistic Human Reposing using Cyclic Self-Supervision with 3D Shape, Pose, and Appearance Consistency <br>
  [ICCV 2021](https://arXiv.org/abs/2110.05458)
- FENeRF: Face Editing in Neural Radiance Fields <br>
  [arXiv 2021](https://arXiv.org/abs/2111.15490)
- Pix2NeRF: Unsupervised Conditional π-GAN for Single Image to Neural Radiance Fields Translation <br>
  [CVPR 2022](https://arXiv.org/abs/2202.13162)
- HeadGAN: One-shot Neural Head Synthesis and Editing <br>
  [ICCV 2021](https://arxiv.org/abs/2012.08261) / [Project Page](https://michaildoukas.github.io/HeadGAN/)
- Editing Conditional Radiance Fields <br>
  [ICCV 2021](http://editnerf.csail.mit.edu/paper.pdf) / [Code](https://github.com/stevliu/editnerf) / [Project Page](http://editnerf.csail.mit.edu/)
- MoFaNeRF: Morphable Facial Neural Radiance Field <br>
  [arXiv 2021](https://arxiv.org/abs/2112.02308) / [Code](https://github.com/zhuhao-nju/mofanerf)
- CoNeRF: Controllable Neural Radiance Fields <br>
  [CVPR 2022](https://arxiv.org/abs/2112.01983) / [Project Page](https://conerf.github.io/)
