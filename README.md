Here is the comprehensive, professional analysis of your domain generalization repository for hyperspectral image (HSI) classification. Every detail has been verified directly against the implementation files.

PART 1 — Repository Structure
File-by-File Purpose


main.py
: The central orchestration module. It parses arguments, configures device settings (CPU/CUDA/MPS), handles HSI dataset splitting/few-shot selection, initializes networks/optimizers, and runs the Stage 1 pre-training and Stage 2 classifier training loops.


datasets.py
: Manages loading and preprocessing datasets (Pavia University/Center, Houston 13/18, HyRANK Dioni/Loukia, etc.). It handles band extraction, class remapping, spectral L2 unit-norm normalization, and custom training (HyperX) and test (HyperX_test) PyTorch dataset wrappers with augmentations.


con_losses.py
: Implements supervised and unsupervised contrastive loss (SupConLoss) supporting standard contrastive aggregation and an adversarial configuration (pushing features apart).


utils_HSI.py
: Contains classification evaluation metrics (OA, AA, Kappa, Confusion Matrix), training helper functions (seed workers, learning rate schedulers), dataset split utilities (sample_gt), and dataset preprocessing summary printers.


sam.py
: Implements the Sharpness-Aware Minimization (SAM) optimizer wrapper. It performs dual-gradient ascent/descent steps to seek flatter loss minima for Stage 2 training.


plotting.py
: Generates training curve visualizations (Stage 1 progressive losses and Stage 2 classification overall accuracy/loss trends).


verify_text_guidance.py
: Sanity check test harness. It mocks input tensors and checks shape/forward compatibility across all vision-language guidance modules (AdaLN3d, MultiHeadCrossAttention, LightweightImageEncoder).


generate_visualizations.py
: Evaluates trained models by generating paper-style visualizations: t-SNE projections, target classification maps with zoom insets, hyperparameter sensitivity surfaces, and sample size accuracy plots.


model/pmg.py
: Implements the progressive multi-stage generator components (Generator, Generator_3DCNN_SupCompress_pca, Spa_Spe_Randomization) and progressive sub-discriminators (Dis).


model/discriminator.py
: Contains the standalone Stage 2 classifier (discriminator). It duplicates the architecture of the final sub-discriminator branch but structures it for final domain inference and features a CLIP projection head.


model/text_guidance.py
: Vision-Language grounding module. It defines the residual image encoder (LightweightImageEncoder), the gated vision-text cross-attention layer (MultiHeadCrossAttention), the conditional 3D normalization layer (AdaLN3d), dataset prompts, and the TextFeatureCache wrapping OpenAI CLIP.
Import Dependency Graph
mermaid
graph TD
    %% Main Execution
    main[main.py] --> verify[verify_text_guidance.py]
    main --> datasets[datasets.py]
    main --> utils[utils_HSI.py]
    main --> pmg[model/pmg.py]
    main --> discriminator[model/discriminator.py]
    main --> text_guidance[model/text_guidance.py]
    main --> con_losses[con_losses.py]
    main --> sam[sam.py]
    main --> plotting[plotting.py]
    main --> gen_vis[generate_visualizations.py]
    
    %% Utilities & Datasets
    datasets --> utils
    verify --> pmg
    verify --> discriminator
    verify --> text_guidance
    
    %% Visualizations & Plotting
    gen_vis --> datasets
    gen_vis --> discriminator
    gen_vis --> pmg
    gen_vis --> plotting
    
    %% Model Internals
    pmg --> text_guidance
PART 2 — Complete Execution Flow
When executing python main.py, the system flows as follows:

[main.py] parses CLI args & initialises devices
   │
   ├──> [main.py] calls get_dataset(source_name) & get_dataset(target_name) inside [datasets.py]
   │       └──> Reads .mat/.tif raw data, crops bands (pavia -> 102), and normalizes values
   │
   ├──> [main.py] loads CLIP ViT-B/32 on device & instantiates TextFeatureCache in [model/text_guidance.py]
   │       └──> Tokenizes dataset-specific text prompts & caches prompt embeddings (C, 512)
   │
   ├──> [main.py] calls preprocess_dataset() inside [utils_HSI.py]
   │       └──> Prints metadata statistics of normalized source and target datasets
   │
   ├──> [main.py] samples Ground Truths via sample_gt() inside [utils_HSI.py]
   │       └──> Extracts train mask on Source (ratio: 0.8) and test mask on Target (ratio: 1.0)
   │
   ├──> [main.py] selects a few-shot subset from the Source dataset
   │       └──> Filters exact sample_nums (default: 10) random training indices per class
   │
   ├──> [main.py] instantiates train_loader & test_loader with HyperX wrapper in [datasets.py]
   │
   ├──> [main.py] instantiates networks:
   │       ├──> g1, g2 = Generator() from [model/pmg.py]
   │       └──> d1, d2 = Dis() from [model/pmg.py]
   │
   ├──> [Stage 1 Pre-training Loop (Progressive Pre-training)]
   │       ├──> Loops over pre_epoch = layers_num * pre_epoch_per_step
   │       ├──> Sets progressive step: current_step = int(epoch / pre_epoch_per_step) + 1
   │       ├──> Feeds batches (x, y) to Generator and sub-discriminators:
   │       │       ├──> g1(x, current_step, text_features, all_class_features) -> x_g1, x_down1
   │       │       ├──> d1(x_down1, current_step) -> predictions, features, clip_projections
   │       │       └──> SupConLoss / CrossEntropy loss calculations (see Part 6)
   │       └──> Evaluates metrics on train_loader to checkpoint best G1/G2 and D1/D2
   │
   ├──> [main.py] saves best G1/G2 checkpoints and loads them
   │
   ├──> [main.py] instantiates D_net = discriminator() from [model/discriminator.py]
   │       └──> Copies final sub-discriminator branch (sub_d[-1]) weights into D_net
   │
   ├──> [main.py] binds SAM / SGD optimizer to D_net
   │
   ├──> [Stage 2 Classifier Training Loop]
   │       ├──> Loops over max_epoch (default: 800)
   │       ├──> Feeds batch x to g1 and g2 in no_grad mode (current_step = layers_num)
   │       │       └──> Generates domain-generalized target variants x1 and x2
   │       ├──> Concatenates tensors: x_total = cat([x, x1, x2], dim=0) (Batch size: 3*B)
   │       ├──> Feeds x_total to D_net -> computes CE class loss + CLIP semantic alignment loss
   │       ├──> Performs SAM optimization step (dual gradient ascent-descent pass)
   │       ├──> Calls evaluate(D_net, test_loader) -> calculates accuracy metrics on Target domain
   │       └──> Saves best.pth checkpoint if Target domain accuracy improves
   │
   └──> [main.py] completes training
           ├──> Writes final results to train_log.txt and comparisons to prompt_comparison.csv
           ├──> Calls save_stage1_plots() and save_stage2_plots() from [plotting.py]
           └──> Calls save_run_tsne() & generate_local_run_plots() in [generate_visualizations.py]
PART 3 — Data Pipeline
The data transformation pipeline is defined in 

datasets.py
 and 

utils_HSI.py
:

Input Mat/TIFF HSI Image
                                │ (H, W, B)
                                ▼
                       Global Max Normalization 
                                │ X = X / max(X)
                                ▼
                      L2 Spectral Normalization 
                                │ X = X / ||X||_2  (along spectral dimension)
                                ▼
                         Spatial Padding
                                │ Pad spatial edges by symmetric extension (r = patch_size//2 + 1)
                                ▼
                       Patch & Index Extraction 
                                │ Extract non-ignored pixels -> (B, patch_size, patch_size)
                                ▼
                       Data Augmentation (Flip/Noise)
                                │ Random Horizontal/Vertical Flips, Radiation/Mixture Noise
                                ▼
                       Transpose to PyTorch Layout
                                │ Convert shape to (B, patch_size, patch_size) -> (B, Channels, 13, 13)
Dataset Loading: Done inside get_dataset() in 

datasets.py
. Reads remote-sensing spectral cubes from Matlab .mat files or GeoTIFF files and extracts coordinates and class maps.
Normalization:
Global Max Normalization: Divides the raw image by its global maximum value: $X = X / \max(X)$.
Spectral Vector Normalization: Reshapes HSI to a 2D matrix $(H \times W, B)$, calculates the L2 norm of the spectral vector at each pixel, and normalizes it to a unit vector: $$\mathbf{x}{norm} = \frac{\mathbf{x}}{\sqrt{\sum{i=1}^{B} x_i^2}}$$
PCA: The pipeline does not perform statistical CPU-bound PCA. Instead, it utilizes a learnable, supervised PCA approach. A $1 \times 1$ convolution layer (self.conv_pca = nn.Conv2d(in_channels, out_channels, 1)) inside Generator_3DCNN_SupCompress_pca learns the optimal spectral projection matrix directly from target constraints.
Band Selection: Pavia University (paviaU) and Pavia Center (paviaC) are cropped to exactly 102 bands to ensure domain compatibility:
paviaU = paviaU[:, :, :-1] (removes last band)
paviaC = paviaC[:, :, :102] (selects first 102 bands)
Patch Extraction: Evaluated dynamically inside the __getitem__ method of HyperX (lines 530-579). For a pixel coordinate $(x, y)$, a spatial window of shape $(P, P, B)$ is cropped, where $P = \text{patch_size}$ (default 13).
Data Augmentation: Three functions are applied to training patches:
Random Flip: Randomly flips patches horizontally or vertically with a 50% probability.
Radiation Noise: Introduces illumination distortion: $\mathbf{x}' = \alpha \mathbf{x} + \beta \mathbf{\epsilon}$, where $\alpha \sim \mathcal{U}(0.9, 1.1)$, $\mathbf{\epsilon} \sim \mathcal{N}(0, 1)$, and $\beta = 1/25$.
Mixture Noise: Interpolates spectra from other pixels sharing the same class label: $$\mathbf{x}' = \frac{\alpha_1 \mathbf{x} + \alpha_2 \mathbf{x}_{target_class}}{\alpha_1 + \alpha_2} + \beta \mathbf{\epsilon}$$
Train/Validation Split: Handled by sample_gt in 

utils_HSI.py
 using train_test_split(stratify=y). The source dataset is split into training (default 80%) and testing/validation sets. Next-step slicing in main.py extracts a specific number of samples per class (few-shot sample size $\leq 30$ per class).
Batch Generation: Tensors are stacked into batch dimensions inside PyTorch's DataLoader.
Tensor Dimension States
Raw HSI Cube: $(H, W, B)$ — e.g., $(610, 340, 103)$ for Pavia University
Post-cropping & Padding: $(H + 2r, W + 2r, B_{cropped})$ — where $r = \lfloor P/2 \rfloor + 1 = 7$, shape becomes $(624, 354, 102)$
Single Patch Extraction: $(P, P, B_{cropped}) = (13, 13, 102)$
Transpose (Dataset Output): $(B_{cropped}, P, P) = (102, 13, 13)$
PyTorch DataLoader Batch: $(Batch, B_{cropped}, P, P) = (512, 102, 13, 13)$
PART 4 — Model Architecture
The complete neural network consists of the progressive generator/discriminator components defined in 

model/pmg.py
, the classifier in 

model/discriminator.py
, and semantic components in 

model/text_guidance.py
.

1. Progressive Generator (Generator)
Consists of layers_num (default 10) cascaded Generator_3DCNN_SupCompress_pca units:

Learnable PCA Projection (self.conv_pca): nn.Conv2d(imdim, dim1=128, kernel_size=1, stride=1)
Lightweight Image Encoder:
Residual blocks containing: Conv2d(imdim, imdim, 3, padding=1) $\rightarrow$ BatchNorm2d(imdim) $\rightarrow$ ReLU $\rightarrow$ Conv2d(imdim, imdim, 3, padding=1) $\rightarrow$ BatchNorm2d(imdim).
Activation: $\mathbf{x} + \text{conv}(\mathbf{x})$ followed by a ReLU activation.
Gated Multi-Head Cross Attention:
Query Projector: nn.Linear(imdim, d_model=512)
Key/Value Projector: nn.Linear(text_dim=512, d_model=512)
Attention Core: nn.MultiheadAttention(embed_dim=512, num_heads=4, batch_first=True)
Output Projector: nn.Linear(d_model=512, imdim)
Gated Fusion: nn.Linear(imdim * 2, imdim). Combines original spatial features $\mathbf{F}{sp}$ and semantic attention features $\mathbf{F}{sem}$: $$\mathbf{G} = \sigma(\text{Linear}([\mathbf{F}{sp} \parallel \mathbf{F}{sem}]))$$ $$\mathbf{F}{fused} = \mathbf{G} \odot \mathbf{F}{sp} + (1.0 - \mathbf{G}) \odot \mathbf{F}_{sem}$$
LayerNorm: nn.LayerNorm(imdim) applied to spatial channels.
Fusion Scaling Layer: Additive residual bypass: $\mathbf{X}{fusion} = \mathbf{X}{feats} + \alpha \mathbf{X}_{sem}$, where $\alpha$ is a learnable parameter initialized to $0.1$.
3D CNN Feature Extractor:
Input layout reshaped to 3D: $(B, 1, \text{depth}=128, \text{height}=13, \text{width}=13)$.
3D Conv: nn.Conv3d(in_channels=1, out_channels=dim2=8, kernel_size=(3,3,3), stride=1, padding=0). Output channels = 8.
AdaLN3d Normalization:
Normalizes the 3D tensor across the channel dimension.
Modulates features using scale/shift parameters generated via nn.Linear(imdim, channels * 2=16) from the global average-pooled semantic feature map.
3D Transpose Conv (self.conv6): nn.ConvTranspose3d(in_channels=8, out_channels=1, kernel_size=(3,3,3), stride=1, padding=0).
Inverse PCA Reconstruction: nn.Conv2d(dim1=128, imdim, kernel_size=1, stride=1) to project back to the target spectral band dimension.
2. Standalone Classifier/Discriminator (D_net)
Layer 1 (Conv + MaxPool): nn.Conv2d(in_channels=B, 64, kernel_size=3, padding=0) $\rightarrow$ nn.MaxPool2d(2) $\rightarrow$ nn.ReLU()
Layer 2 (Conv + MaxPool): nn.Conv2d(64, 128, kernel_size=3, padding=0) $\rightarrow$ nn.MaxPool2d(2) $\rightarrow$ nn.ReLU()
Layer 3 (Fully Connected): nn.Linear(128, 512) $\rightarrow$ nn.ReLU()
Layer 4 (Fully Connected): nn.Linear(512, 512) $\rightarrow$ nn.ReLU()
Classification Head: nn.Linear(512, num_classes)
Projection Head: nn.Linear(512, proj_dim=128) followed by a nn.ReLU and L2 normalization.
CLIP Projection Head: nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512)) followed by L2 normalization.
Exact Tensor Dimensions (Forward Pipeline)
Input HSI Patch
                           │ (B, 102, 13, 13)
                           ▼
                 Lightweight Image Encoder
                           │ (B, 102, 13, 13)
                           ▼
                  Gated Cross-Attention
                           │ (B, 102, 13, 13)
                           ▼
                       1x1 Conv PCA
                           │ (B, 128, 13, 13)
                           ▼
                   Reshape to 5D Tensor
                           │ (B, 1, 128, 13, 13)
                           ▼
                     3D Convolution
                           │ (B, 8, 126, 11, 11)
                           ▼
                 AdaLN3D Parameter Modulation
                           │ (B, 8, 126, 11, 11)
                           ▼
                  3D Transpose Convolution
                           │ (B, 1, 128, 13, 13)
                           ▼
                 Reshape to 4D Tensor
                           │ (B, 128, 13, 13)
                           ▼
                   1x1 Conv Inv PCA
                           │ (B, 102, 13, 13)
PART 5 — Training Flow
The complete training flow is split into two distinct, sequential training stages:

[STAGE 1: Pre-training]
                       
                            Original Source HSI x
                                      │
                                      ▼
                        ┌─────────────┴─────────────┐
                        ▼                           ▼
                  Generator g1                 Generator g2
                        │                           │
                        ▼                           ▼
                 x_g1, x_down1                x_g2, x_down2
                        │                           │
                        ▼                           ▼
                 Discriminator d1             Discriminator d2
                        │                           │
                        ▼                           ▼
                 z_ED1, z_SD1                 z_ED2, z_SD2
                        │                           │
                        ▼                           ▼
                 SupConLoss (L1)              SupConLoss (L1)
                 Class Loss  (CE)             Class Loss  (CE)
                 Semantic    (Sem)            Semantic    (Sem)
                        │                           │
                        ▼                           ▼
                   Update d1, g1               Update d2, g2
                                  
                                      │
                                      ▼ (Weights loaded)
                                      
                         [STAGE 2: Classifier Training]
                       
                            Original Source HSI x
                                      │
                        ┌─────────────┼─────────────┐
                        │             ▼             ▼
                        │        Generator g1  Generator g2
                        │             │             │
                        ▼             ▼             ▼
                        x            x1            x2
                        └─────────────┼─────────────┘
                                      ▼ (Concatenate)
                                 Total X_total (3*B, 102, 13, 13)
                                      │
                                      ▼
                                 Classifier D_net
                                      │
                                      ▼
                                Predict Class (y)
                                CLIP Project  (z_clip)
                                      │
                                      ▼
                               CE Loss (Cls) + Clip Loss (CLIP)
                                      │
                                      ▼
                                 Update D_net (SAM Optimizer)
Stage 1 (Progressive Pre-training Loop)
Forward Pass:
A batch of source patches $x \in \mathbb{R}^{B \times 102 \times 13 \times 13}$ and labels $y \in \mathbb{R}^B$ are loaded.
Path features are generated: $x_{g1}, x_{down1} = g_1(x)$.
Discriminator features are extracted:
Generated: $p_{ED1}, z_{ED1}, \text{clip_proj}{ED1} = d_1(x{g1})$.
Downsampled: $p_{SD1}, z_{SD1}, \text{clip_proj}{SD1} = d_1(x{down1})$.
Real Source: $p_{src0_1}, z_{whole1}, \text{clip_proj}_{whole1} = d_1(x)$.
Loss Evaluation:
Downsampled consistency loss: $\mathcal{L}{wd} = \text{SupCon}(z{whole1}, z_{down1})$.
Discriminator classification and multi-view contrastive losses are computed.
Semantic Consistency Loss: Calculates the cross-entropy loss between normalized CLIP projections and text target class embeddings.
Backward & Update:
Gradients are backpropagated through $d_1, d_2$ and $g_1, g_2$. Parameters are updated using Adam optimizers (G1_opt, G2_opt, D1_opt, D2_opt).
Stage 2 (SAM-guided Classifier Training Loop)
Augmented Batch Construction:
The generators $g_1$ and $g_2$ run in evaluation mode to translate source patches $x$ into generalized target-domain variants $x_1$ and $x_2$.
Tensors are concatenated: $x_{total} = [x, x_1, x_2]^T \in \mathbb{R}^{3B \times 102 \times 13 \times 13}$.
First Forward & Backward Step:
Feeds $x_{total}$ to $D_{net}$ to output class logits and CLIP projections.
Total loss is computed: $\mathcal{L}{total} = \mathcal{L}{CE} + \lambda_{clip} \mathcal{L}_{clip}$.
Runs loss.backward() to compute gradients.
Sharpness Coordinate Search:
The SAM optimizer calls D_sam.first_step(zero_grad=True) to shift D_net's parameters along the gradient direction by step size $\rho=0.05$.
Second Forward & Backward Step:
Re-evaluates $x_{total}$ on the shifted parameters to calculate the local loss maximum.
Runs loss.backward() to compute the final update gradients.
Calls D_sam.second_step(zero_grad=True) to restore the original parameters and apply the updated weight step.
PART 6 — Loss Functions
The training loss formulations are implemented in 

main.py
 and 

con_losses.py
.

1. Progressive Consistency Loss ($\mathcal{L}_{wd}$)
Purpose: Encourages the learnable convolution-based PCA downsampling path inside the generators to preserve spatial-spectral structures by matching downsampled representations with those of the original full-band image.
Mathematical Formula: $$\mathcal{L}{wd} = \mathcal{L}{con}([\mathbf{z}{whole1} \parallel \mathbf{z}{down1}], y) + \mathcal{L}{con}([\mathbf{z}{whole2} \parallel \mathbf{z}{down2}], y)$$ where $\mathcal{L}{con}$ is the supervised contrastive loss.
Code Location: 

main.py:L370-376
.
Weight: 1.0 (unscaled).
Gradient Flow: Flows through sub-discriminators $d_1/d_2$ and back into generators $g_1/g_2$ to update learnable downsampling layers.
2. Multi-View Discriminator Loss ($\mathcal{L}{1_1}$ & $\mathcal{L}{1_2}$)
Purpose: Trains the progressive sub-discriminators to accurately classify real, downsampled, and generated samples while aligning their feature representations in latent space.
Mathematical Formula: $$\mathcal{L}{1_1} = \text{CE}(\mathbf{p}{SD1}, y) + \text{CE}(\mathbf{p}{ED1}, y) + \lambda_1 \mathcal{L}{con}([\mathbf{z}{tgt1} \parallel \mathbf{z}{SD1} \parallel \mathbf{z}{ED1}], y) + \text{CE}(\mathbf{p}{tgt1}, y) + \lambda_{sem} \mathcal{L}_{sem_SD1}$$
Code Location: 

main.py:L380-396
.
Weight: $\lambda_1 = 0.01$ (contrastive penalty); $\lambda_{sem} = 0.05$ (semantic alignment).
Gradient Flow: Updates only the active sub-discriminator parameters.
3. Generator Style Adversarial Loss ($\mathcal{L}{2_1}$ & $\mathcal{L}{2_2}$)
Purpose: Drives the generators to create diverse domain-generalized representations. The adversarial contrastive loss pushes generated features away from the downsampled features of the same class in latent space.
Mathematical Formula: $$\mathcal{L}{2_1} = \text{CE}(\mathbf{p}{tgt1}, y) + \lambda_2 \mathcal{L}{adv}([\mathbf{z}{SD_i1} \parallel \mathbf{z}{tgt_i1}], y{i1}) + \lambda_{sem} \mathcal{L}{sem_tgt1}$$ where $\mathcal{L}{adv}$ uses the negative log-probability option in con_losses.py: $$\mathcal{L}{adv} = -\frac{\tau}{\tau{base}} \log \left( 1 - \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_p / \tau)}{\sum \exp(\mathbf{z}_i \cdot \mathbf{z}_a / \tau)} + 1e-6 \right)$$
Code Location: 

main.py:L410-425
.
Weight: $\lambda_2 = 0.01$; $\lambda_{sem} = 0.05$.
Gradient Flow: Flows through discriminators into the generator networks to update generation weights.
4. Stage 2 Combined Classification & Alignment Loss ($\mathcal{L}_{stage2}$)
Purpose: Jointly optimizes D_net for target classification accuracy and aligns its final feature representations with the CLIP text embeddings.
Mathematical Formula: $$\mathcal{L}{stage2} = \text{CE}(\mathbf{p}, y{total}) + \lambda_{clip} \text{CE}(s \cdot \mathbf{z}{clip} \mathbf{E}{text}^T, y_{total})$$ where $s$ is the logit scale and $\mathbf{E}_{text}$ is the cached matrix of CLIP text embeddings.
Code Location: 

main.py:L608-610
.
Weight: $\lambda_{clip} = 0.1$.
Gradient Flow: Updates the parameters of the standalone classifier D_net during SAM optimization.
PART 7 — Architecture Diagram
The diagram below illustrates the exact flow of data through the network during training.

[Source Cube] (B, 102, 13, 13) ────> [Learnable PCA Conv2D] (B, 128, 13, 13)
             │                                     │
             │                                     ▼
             │                               [Reshape to 3D] (B, 1, 128, 13, 13)
             │                                     │
             │                                     ▼
             │                               [Conv3D Layer] (B, 8, 126, 11, 11)
             │                                     │
             │ (Lightweight Image Encoder)         ▼
             ├──────────────────────────> [AdaLN3d Modulation] <─── [CLIP Text Embeds]
             │                                     │
             │                                     ▼
             │                               [ConvTranspose3D] (B, 1, 128, 13, 13)
             │                                     │
             │                                     ▼
             │                               [Inverse PCA Conv2D]
             │                                     │
             ▼                                     ▼
      [Downsampled Variant]                 [Generated Variant]
       (B, Dim_k, 13, 13)                    (B, Dim_k, 13, 13)
             │                                     │
             └──────────────────┬──────────────────┘
                                │ (Concatenate batches)
                                ▼
                         [Classifier D_net]
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
 [Classification Logits] [Projection Features]    [CLIP Projection]
     (B, num_classes)         (B, 128)                 (B, 512)
        │                       │                       │
        ▼                       ▼                       ▼
  Cross Entropy Loss    Supervised Contrastive   Vision-Language Logits
  against target class       Loss (SupCon)        against CLIP Prompt Bank
PART 8 — Mathematical Formulation
1. Learnable PCA Compression & Reconstruction
Let $\mathbf{X}k \in \mathbb{R}^{B \times D_k \times H \times W}$ be the feature tensor at stage $k$. The Learnable PCA Downprojection and Inverse PCA Reconstruction are defined as: $$\mathbf{X}{pca} = \mathbf{W}{pca} * \mathbf{X}k + \mathbf{b}{pca}$$ $$\mathbf{X}{recon} = \mathbf{W}{inv} * \mathbf{X}{up} + \mathbf{b}{inv}$$ where $*$ denotes $2\text{D}$ convolution with a $1 \times 1$ kernel, $\mathbf{W}{pca} \in \mathbb{R}^{128 \times D_k \times 1 \times 1}$, and $\mathbf{W}_{inv} \in \mathbb{R}^{D_k \times 128 \times 1 \times 1}$.

2. Multi-Head Cross Attention
Let $\mathbf{F}{img} \in \mathbb{R}^{HW \times D_k}$ be the flattened spatial features and $\mathbf{E}{txt} \in \mathbb{R}^{C \times 512}$ be the class prompt embeddings. The query, key, and value matrices are computed as: $$\mathbf{Q} = \mathbf{F}{img} \mathbf{W}Q, \quad \mathbf{K} = \mathbf{E}{txt} \mathbf{W}K, \quad \mathbf{V} = \mathbf{E}{txt} \mathbf{W}V$$ The gated attention fusion is defined as: $$\mathbf{F}{attn} = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d{model}}}\right)\mathbf{V}$$ $$\mathbf{G} = \sigma([\mathbf{F}{img} \parallel \mathbf{F}{attn}] \mathbf{W}g + \mathbf{b}g)$$ $$\mathbf{F}{fused} = \text{LayerNorm}(\mathbf{G} \odot \mathbf{F}{img} + (1.0 - \mathbf{G}) \odot \mathbf{F}_{attn})$$

3. Adaptive 3D Layer Normalization (AdaLN3d)
For a 3D feature tensor $\mathbf{V} \in \mathbb{R}^{B \times C \times L \times H \times W}$ and a semantic conditioning vector $\mathbf{v}{sem} \in \mathbb{R}^{D_k}$: $$\text{AdaLN3d}(\mathbf{V}, \mathbf{v}{sem}) = (1.0 + \mathbf{\gamma}) \odot \left(\frac{\mathbf{V} - \mu(\mathbf{V})}{\sqrt{\sigma^2(\mathbf{V}) + \epsilon}}\right) + \mathbf{\beta}$$ where $\mathbf{\gamma}, \mathbf{\beta} = \text{Split}(\text{MLP}(\mathbf{v}_{sem}))$.

4. Sharpness-Aware Minimization (SAM)
To optimize the classifier parameters $\mathbf{w}$ in Stage 2, SAM solves: $$\min_{\mathbf{w}} \mathcal{L}{stage2}(\mathbf{w}) + \max{|\mathbf{\epsilon}|2 \le \rho} \left( \mathcal{L}{stage2}(\mathbf{w} + \mathbf{\epsilon}) - \mathcal{L}{stage2}(\mathbf{w}) \right)$$ The optimal parameter perturbation $\mathbf{\epsilon}^$ is approximated as: $$\mathbf{\epsilon}^ \approx \rho \frac{\nabla{\mathbf{w}} \mathcal{L}{stage2}(\mathbf{w})}{|\nabla{\mathbf{w}} \mathcal{L}_{stage2}(\mathbf{w})|_2}$$

PART 9 — Compare Against Original PMGDG
Original PMGDG Paper	Your Implementation	Technical Difference	Expected Benefit
Spectral PCA Preprocessing	Learnable 1x1 Conv PCA	CPU statistical PCA is replaced by learnable 1x1 Conv layers inside the sub-generators.	Allows the network to learn a task-specific spectral projection end-to-end.
Style Randomization (SSR)	Multi-Head Cross Attention	SSR style mixing is replaced with CLIP-based text prompt cross-attention.	Improves semantic grounding by aligning spatial features with textual class descriptions.
Standard Normalization	AdaLN3d Modulation	Layer normalization is replaced with semantic-conditioned 3D Adaptive Layer Norm.	Better aligns features by conditioning the normalization parameters on class semantics.
SGD/Adam Optimization	SAM (Sharpness-Aware Minimization)	Optimizes D_net using SAM's dual-step gradient formulation.	Implements a flatness-seeking optimizer that prevents overfitting to the source domain.
No Text Guidance	Vision-Language Contrastive Loss	Evaluates cross-entropy loss against CLIP text embeddings.	Imposes semantic constraints that help align vision and language features across domains.
PART 10 — Novel Contributions
1. Existing PMGDG Components
Progressive multi-stage generator framework using a cascade of sub-generators and sub-discriminators.
Learnable channel downsampling to bridge high-dimensional and low-dimensional spectral features.
The multi-view supervised contrastive loss design.
2. Your Modifications & New Modules


model/text_guidance.py
 - LightweightImageEncoder: Integrates a residual CNN block before the cross-attention layers.


model/text_guidance.py
 - MultiHeadCrossAttention: Implements a vision-text cross-attention mechanism with a learnable gated fusion layer.


model/text_guidance.py
 - AdaLN3d: Implements 3D Adaptive Layer Normalization to modulate features using scale/shift parameters generated from prompt embeddings.


sam.py
 - SAM: Integrates the Sharpness-Aware Minimization optimizer to improve generalization on target domains.
3. Your New Losses
Progressive Semantic Consistency Loss (Stage 1): Updates both generators and discriminators during pre-training using cross-entropy loss against CLIP text embeddings.
Classifier Vision-Language Loss (Stage 2): Aligns the classifier's visual features with textual prompt representations.
PART 11 — Research Paper Methodology
3. Proposed Methodology
3.1. Overview and Problem Formulation
In few-shot HSI domain generalization, we are provided with a labeled source domain $\mathcal{D}_s = {(\mathbf{x}i^s, y_i^s)}{i=1}^{N_s}$ containing only a few labeled samples per class, and an unlabeled target domain $\mathcal{D}_t = {\mathbf{x}j^t}{j=1}^{N_t}$. Our objective is to learn a robust mapping function $f: \mathcal{X} \to \mathcal{Y}$ that generalizes well to $\mathcal{D}_t$ without target domain training labels.

3.2. Semantic-Guided Progressive Multi-stage Generator (SG-PMG)
To bridge the domain gap, we propose the Semantic-Guided Progressive Multi-stage Generator (SG-PMG). For each stage $k \in {1, \dots, K}$, the sub-generator translates HSI patches via a learnable PCA projection layer and modulates features using semantic text prompts:

$$\mathbf{X}{feats} = \text{Encoder}(\mathbf{X}k)$$ $$\mathbf{F}{attn} = \text{CrossAttention}(\mathbf{X}{feats}, \mathbf{E}{txt})$$ $$\mathbf{X}{fused} = \mathbf{X}{feats} + \alpha \cdot \text{LayerNorm}(\text{Gate}(\mathbf{X}{feats}, \mathbf{F}_{attn}))$$

The fused representations are reshaped into 3D tensors and modulated using 3D Adaptive Layer Normalization (AdaLN3d), where scale ($\gamma$) and shift ($\beta$) parameters are computed dynamically from the class semantic representations:

$$\text{AdaLN3d}(\mathbf{V}, \mathbf{v}_{sem}) = (1 + \gamma) \odot \hat{\mathbf{V}} + \beta$$

3.3. Dual-Stage Optimization and Text Guidance
Training is split into two sequential phases:

Stage 1 (Pre-training): Optimizes the generators and sub-discriminators using a combination of supervised contrastive loss, downsampled consistency loss, and progressive semantic consistency loss against CLIP text embeddings: $$\mathcal{L}{G_step} = \text{CE}(\mathbf{p}{tgt}, y) + \lambda_2 \mathcal{L}{adv}(\mathbf{z}{SD}, \mathbf{z}{tgt}) + \lambda{sem} \mathcal{L}_{sem}$$
Stage 2 (SAM-guided Classifier Tuning): Trains the classifier on both source samples and generated target-domain variants. We optimize the network using Sharpness-Aware Minimization (SAM) and a vision-language contrastive loss to align final features with CLIP prompt embeddings: $$\mathcal{L}{stage2} = \text{CE}(\mathbf{p}, y) + \lambda{clip} \text{CE}(s \cdot \mathbf{z}{clip} \mathbf{E}{text}^T, y)$$
PART 12 — Evaluation
Novelty Score: 8.5/10 — Replaces standard random style-mixing with vision-language alignment (via CLIP prompts, cross-attention, and AdaLN3d), which is a significant architectural enhancement for domain generalization.
Research Quality: 8.0/10 — The methodology is mathematically grounded. It integrates contrastive learning and vision-language grounding, and the codebase includes verification tests.
Implementation Complexity: 8.5/10 — The codebase implements complex multi-dimensional operations: progressive 3D CNNs, custom 3D normalization, multi-head attention, and dual-step SAM updates.
Publication Readiness: 8.0/10 — The code is well-structured and ready for experiments. The visualization pipeline (generate_visualizations.py) is complete and matches typical CVPR/ICCV paper figures.
Strengths
Flatter Loss Minima: Using the SAM optimizer helps the model find flatter minima, which typically improves generalization performance.
End-to-End learnable PCA: Replacing static PCA with learnable 1x1 convolutions allows the model to learn optimal spectral projections directly from data.
Structured Prompt Options: Provides three prompt templates (original, ehsnet, rich), allowing you to study how prompt engineering impacts domain generalization.
Weaknesses
High GPU Memory Usage: Training multiple generators and sub-discriminators along with a frozen CLIP model in Stage 1 is computationally expensive.
Duplicated Code: The discriminator class is defined in both model/pmg.py and model/discriminator.py, which could be consolidated.
Suggestions for Future Work
Test Parameter Sensitivity: Evaluate performance across different values of $\lambda_1, \lambda_2, \lambda_{sem}, \lambda_{clip}$.
Expand Prompt Templates: Incorporate LLM-generated prompts or learnable prompt tuning weights.
Evaluate More HSI Datasets: Test on additional HSI transfer tasks (e.g., Indian Pines to Salinas) to demonstrate the robustness of the method.
11:47 AM
# sem
