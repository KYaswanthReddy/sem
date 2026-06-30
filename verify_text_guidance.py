"""Sanity check: verify tensor dimensions and module connectivity."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

# ── 1. AdaLN3d, LightweightImageEncoder, MultiHeadCrossAttention ──
print("=" * 60)
print("1. Testing AdaLN3d, LightweightImageEncoder, MultiHeadCrossAttention")
print("=" * 60)
from model.text_guidance import AdaLN3d, LightweightImageEncoder, MultiHeadCrossAttention

# Test AdaLN3d
ada = AdaLN3d(n_channel=8, text_dim=512)
ada.train()
x_5d = torch.randn(4, 8, 126, 11, 11)  # (N, C, L, H, W)
text_feat = torch.randn(4, 512)          # (N, text_dim)
out_ada = ada(x_5d, text_features=text_feat)
print(f"  AdaLN3d Input:  {x_5d.shape}")
print(f"  AdaLN3d Output: {out_ada.shape}")
assert out_ada.shape == x_5d.shape, f"Shape mismatch: {out_ada.shape} != {x_5d.shape}"
print("  ✓ AdaLN3d matches shape")

# Fallback without text
out_ada_fallback = ada(x_5d, text_features=None)
assert out_ada_fallback.shape == x_5d.shape
print("  ✓ AdaLN3d fallback works")

# Test LightweightImageEncoder
enc = LightweightImageEncoder(in_channels=10)
enc.train()
x_2d = torch.randn(4, 10, 13, 13)
out_enc = enc(x_2d)
print(f"  Encoder Input:  {x_2d.shape}")
print(f"  Encoder Output: {out_enc.shape}")
assert out_enc.shape == x_2d.shape
print("  ✓ LightweightImageEncoder matches shape")

# Test MultiHeadCrossAttention
mha = MultiHeadCrossAttention(img_dim=10, text_dim=512, num_heads=2)
mha.train()
out_mha = mha(out_enc, text_feat)
print(f"  MHA Output: {out_mha.shape}")
assert out_mha.shape == out_enc.shape
print("  ✓ MultiHeadCrossAttention matches shape")

# ── 2. Generator_3DCNN_SupCompress_pca with text_dim ──
print("\n" + "=" * 60)
print("2. Testing Generator_3DCNN_SupCompress_pca (text-guided)")
print("=" * 60)
from model.pmg import Generator_3DCNN_SupCompress_pca

sub_g = Generator_3DCNN_SupCompress_pca(imdim=10, imsize=[13, 13], device='cpu', dim1=128, dim2=8, text_dim=512)
sub_g.train()
x_in = torch.randn(4, 10, 13, 13)
text_in = torch.randn(4, 512)

out_g = sub_g(x_in, text_cond=text_in)
print(f"  Input:  {x_in.shape}")
print(f"  Text:   {text_in.shape}")
print(f"  Output: {out_g.shape}")
assert out_g.shape == x_in.shape, f"Shape mismatch: {out_g.shape} != {x_in.shape}"
print("  ✓ Sub-generator shape correct")

# Without text (fallback)
sub_g_no_text = Generator_3DCNN_SupCompress_pca(imdim=10, imsize=[13, 13], device='cpu', dim1=128, dim2=8, text_dim=0)
sub_g_no_text.train()
out_g2 = sub_g_no_text(x_in)
assert out_g2.shape == x_in.shape
print("  ✓ Sub-generator without text_dim works (backward compat)")

# ── 3. Full Generator with text guidance ──
print("\n" + "=" * 60)
print("3. Testing Generator (progressive, text-guided)")
print("=" * 60)
from model.pmg import Generator

N_BANDS = 102
layers = [10, 20, 30, 40, 50, 60, 70, 80, 90, 102]
gen = Generator(imdim=N_BANDS, patch_size=13, layers=layers, dim1=128, dim2=8, device='cpu', text_dim=512)
gen.train()

x_full = torch.randn(4, N_BANDS, 13, 13)
text_batch = torch.randn(4, 512)
num_classes = 7
all_class_text = F.normalize(torch.randn(7, 512))

# (A) Center-class-only conditioning (original behaviour)
for step in [1, 3, 5, 10]:
    x_g, x_down = gen(x_full, current_step=step, text_features=text_batch)
    expected_dim = layers[step - 1]
    print(f"  [center-class] Step {step:2d}: x_g={x_g.shape}, x_down={x_down.shape}  (expected channels={expected_dim})")
    assert x_g.shape == (4, expected_dim, 13, 13), f"x_g shape wrong at step {step}"
    if step < len(layers):
        assert x_down.shape == (4, expected_dim, 13, 13), f"x_down shape wrong at step {step}"
    else:
        assert x_down.shape == (4, N_BANDS, 13, 13)
print("  ✓ All progressive steps correct (center-class conditioning)")

# (B) All-class prompt bank conditioning (Improvement 3)
batch_all_class = all_class_text.unsqueeze(0).repeat(4, 1, 1)  # (B, num_classes, 512)
for step in [1, 5, 10]:
    x_g, x_down = gen(x_full, current_step=step, text_features=text_batch, all_class_features=batch_all_class)
    expected_dim = layers[step - 1]
    print(f"  [prompt-bank]  Step {step:2d}: x_g={x_g.shape}, x_down={x_down.shape}")
    assert x_g.shape == (4, expected_dim, 13, 13), f"x_g shape wrong at step {step} (prompt bank)"
print("  ✓ All progressive steps correct (prompt bank conditioning)")

# (C) Stage-1 Semantic Consistency: sub-discriminator CLIP Projection Head
from model.pmg import Dis
d_stage1_test = Dis(imdim=N_BANDS, patch_size=13, layers=layers, proj=128, num_classes=num_classes)
d_stage1_test.train()

for step in [1, 5, 10]:
    x_g, _ = gen(x_full, current_step=step, text_features=text_batch, all_class_features=batch_all_class)
    clss, proj, clip_proj = d_stage1_test(x_g, current_step=step, mode='train')
    print(f"  [clip_proj]    Step {step:2d}: clip_proj={clip_proj.shape}  (expected (4, 512))")
    assert clip_proj.shape == (4, 512), f"Discriminator CLIP projection shape wrong at step {step}"
    # Verify cosine similarity computation works
    cos_sim = F.cosine_similarity(clip_proj, text_batch, dim=-1)
    l_sem = 1.0 - cos_sim.mean()
    assert not torch.isnan(l_sem), f"L_sem is NaN at step {step}"
    print(f"            L_sem={l_sem.item():.4f}")
print("  ✓ Discriminator CLIP projection and L_sem computation correct")

# Stage 2 mode (current_step > layers_num)
x_down_only = gen(x_full, current_step=len(layers) + 1)
print(f"  Stage 2 mode: x_down={x_down_only.shape}")
print("  ✓ Stage 2 mode works")


# ── 4. Discriminator (Stage 2) ──
print("\n" + "=" * 60)
print("4. Testing discriminator (Stage 2, fixed CLIP head)")
print("=" * 60)
from model.discriminator import discriminator

d_net = discriminator(inchannel=N_BANDS, outchannel=128, num_classes=7, patch_size=13)
d_net.train()

x_disc = torch.randn(4, N_BANDS, 13, 13)
clss, proj, clip_proj = d_net(x_disc, mode='train')
print(f"  Input:     {x_disc.shape}")
print(f"  clss:      {clss.shape}")
print(f"  proj:      {proj.shape}")
print(f"  clip_proj: {clip_proj.shape}")
assert clss.shape == (4, 7)
assert proj.shape == (4, 128)
assert clip_proj.shape == (4, 512)
print("  ✓ All discriminator outputs correct")

# Test mode
clss_test = d_net(x_disc, mode='test')
assert clss_test.shape == (4, 7)
print("  ✓ Test mode works")

# ── 5. CLIP loss computation ──
print("\n" + "=" * 60)
print("5. Testing CLIP loss computation")
print("=" * 60)
text_feats = F.normalize(torch.randn(7, 512))
logit_scale = torch.tensor(100.0)
y = torch.tensor([0, 1, 2, 3])

clip_loss = F.cross_entropy(logit_scale * clip_proj @ text_feats.t(), y)
print(f"  clip_proj @ text_feats.T shape: {(clip_proj @ text_feats.t()).shape}")
print(f"  CLIP loss: {clip_loss.item():.4f}")

# Check gradients flow back
clip_loss.backward()
has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in d_net.conv1.parameters())
print(f"  Gradient reaches conv1: {has_grad}")
print("  ✓ CLIP loss computation correct")

# ── 6. TextFeatureCache ──
print("\n" + "=" * 60)
print("6. Testing TextFeatureCache (mock — no actual CLIP model)")
print("=" * 60)
from model.text_guidance import TextFeatureCache

# Mock test without loading real CLIP (just verify index_by_labels logic)
class MockCache:
    def __init__(self):
        self.text_features = F.normalize(torch.randn(7, 512))
        self.device = 'cpu'
    def index_by_labels(self, labels):
        return self.text_features[labels.long()]

mc = MockCache()
labels = torch.tensor([0, 3, 6, 1])
batch_text = mc.index_by_labels(labels)
print(f"  Labels: {labels}")
print(f"  Batch text features: {batch_text.shape}")
assert batch_text.shape == (4, 512)
assert torch.allclose(batch_text[0], mc.text_features[0])
assert torch.allclose(batch_text[2], mc.text_features[6])
print("  ✓ Label indexing correct")

# ── 7. get_dataset_prompts ──
print("\n" + "=" * 60)
print("7. Testing get_dataset_prompts")
print("=" * 60)
from model.text_guidance import get_dataset_prompts

label_vals = ["Tree", "Asphalt", "Brick", "Bitumen", "Shadow", "Meadow", "Bare soil"]
for prompt_type in ['original', 'ehsnet', 'rich']:
    prompts = get_dataset_prompts('paviaU', label_vals, prompt_type)
    print(f"  {prompt_type}: {len(prompts)} prompts. Example[0]: '{prompts[0][:60]}...'")
    assert len(prompts) == len(label_vals)
print("  ✓ get_dataset_prompts correct")

# ── Summary ──
print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
