#!/usr/bin/env python3
"""
Diagnose CUDA device-side assert issues in distributed training.
Run this BEFORE starting training to verify the fixes work.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import math

def test_bce_loss_numerical_stability():
    """Test BCELoss with edge case values."""
    print("=" * 60)
    print("TEST 1: BCELoss Numerical Stability")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCELoss()

    # Test cases that previously caused issues
    test_cases = [
        ("Perfect prediction (0.5)", torch.tensor([0.5], device=device), torch.tensor([0.5], device=device)),
        ("Near 0 (unclamped)", torch.tensor([1e-10], device=device), torch.tensor([1.0], device=device)),
        ("Near 1 (unclamped)", torch.tensor([1.0 - 1e-10], device=device), torch.tensor([0.0], device=device)),
        ("Clamped low", torch.clamp(torch.tensor([1e-10], device=device), min=1e-7, max=1-1e-7), torch.tensor([1.0], device=device)),
        ("Clamped high", torch.clamp(torch.tensor([1.0], device=device), min=1e-7, max=1-1e-7), torch.tensor([0.0], device=device)),
    ]

    for name, pred, target in test_cases:
        try:
            pred = pred.float().requires_grad_(True)
            loss = criterion(pred, target.float())
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  ❌ {name}: FAILED (NaN/Inf) - loss={loss}")
            else:
                loss.backward()
                print(f"  ✅ {name}: OK - loss={loss.item():.6f}")
        except Exception as e:
            print(f"  ❌ {name}: EXCEPTION - {type(e).__name__}: {str(e)[:60]}")

def test_mixed_precision_stability():
    """Test mixed precision training with fp16."""
    print("\n" + "=" * 60)
    print("TEST 2: Mixed Precision (fp16) Stability")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  ⚠️  CUDA not available, skipping fp16 test")
        return

    device = torch.device("cuda")
    scaler = GradScaler()
    model = nn.Linear(768, 1)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()

    try:
        # Simulate training step
        x = torch.randn(4, 768, device=device)
        y = torch.randint(0, 2, (4,), dtype=torch.float, device=device)

        model.train()
        with autocast(enabled=True):
            logits = model(x).squeeze(-1)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(model.parameters())
        scaler.update()

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  ❌ FAILED (NaN/Inf loss) - loss={loss}")
        else:
            print(f"  ✅ OK - loss={loss.item():.6f}")
    except Exception as e:
        print(f"  ❌ EXCEPTION - {type(e).__name__}: {str(e)[:80]}")

def test_v_claim_edge_cases():
    """Test V_claim computation with edge cases."""
    print("\n" + "=" * 60)
    print("TEST 3: V_claim Edge Cases")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_cases = [
        ("All claims present", torch.ones(4, 5, device=device), "should aggregate all"),
        ("No claims", torch.zeros(4, 5, device=device), "should return ~0.5"),
        ("Half claims", torch.tensor([[1,0,1,0,1],[0,1,0,1,0],[1,1,0,0,0],[0,0,1,1,0]], dtype=torch.float, device=device), "should aggregate valid"),
    ]

    for name, claim_mask, desc in test_cases:
        B, P = claim_mask.shape
        # Simulate claim scores
        claim_scores = torch.sigmoid(torch.randn(B, P, device=device))

        # Compute V_claim as in CEIN
        pool_logits = torch.randn(B, P, device=device)
        pool_logits = pool_logits + (claim_mask - 1.0) * 1e9
        pool_weights = F.softmax(pool_logits, dim=-1) * claim_mask
        V_claim = (pool_weights * claim_scores).sum(dim=1)

        # Apply clamping
        V_claim_clamped = torch.clamp(V_claim, min=1e-7, max=1-1e-7)

        # Check for NaN/Inf
        has_nan = torch.isnan(V_claim).any()
        has_inf = torch.isinf(V_claim).any()

        print(f"  {name}: {desc}")
        print(f"    V_claim range: [{V_claim.min():.6f}, {V_claim.max():.6f}]")
        print(f"    After clamp: [{V_claim_clamped.min():.6f}, {V_claim_clamped.max():.6f}]")
        if has_nan or has_inf:
            print(f"    ❌ Contains NaN/Inf (unclamped)")
        else:
            print(f"    ✅ OK")

def test_distributed_communication():
    """Test basic distributed communication (if world_size > 1)."""
    print("\n" + "=" * 60)
    print("TEST 4: Distributed Communication")
    print("=" * 60)

    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            print(f"  Rank {rank}/{world_size}")

            # Test all-reduce
            x = torch.tensor([1.0] * 4, device="cuda")
            dist.all_reduce(x)
            print(f"  ✅ All-reduce works: {x}")
        else:
            print("  ⚠️  Distributed not initialized (expected for single-process test)")
    except Exception as e:
        print(f"  ⚠️  {type(e).__name__}: {str(e)[:60]}")

if __name__ == "__main__":
    print("\n🔍 CUDA Device-Side Assert Diagnostic Tool\n")

    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA capability: {torch.cuda.get_device_capability()}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")

    test_bce_loss_numerical_stability()
    test_mixed_precision_stability()
    test_v_claim_edge_cases()
    test_distributed_communication()

    print("\n" + "=" * 60)
    print("✅ Diagnostic tests complete!")
    print("=" * 60)
