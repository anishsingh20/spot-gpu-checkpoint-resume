#!/usr/bin/env python3
"""
Checkpoint-and-resume training loop for DigitalOcean Spot GPU Droplets.

Three layers of protection, cheapest first:
  1. Periodic checkpoint every --every-steps to Spaces. This is what keeps
     the run safe no matter how or when the Droplet goes away.
  2. SIGTERM/SIGINT trap -> checkpoint now and exit cleanly. Costs five lines
     and makes `systemctl stop` (or any graceful shutdown) lose zero steps.
  3. Drain trigger for the two-hour reclaim email: touch /tmp/checkpoint-now
     (flush, keep going) or send SIGTERM (flush, exit). Same code path.

Resume is automatic. On start the script reads checkpoints/<run-id>/LATEST
from Spaces, restores model + optimizer + step, and continues. No checkpoint
means a fresh run. Run the identical command on the replacement Droplet.

Env (put these in /root/.spaces.env, chmod 600, `source` it before running):
  SPACES_KEY, SPACES_SECRET   Spaces access key pair
  SPACES_REGION               Spaces region, e.g. nyc3
  SPACES_BUCKET               bucket name

Run:
  python3 spot_train.py --run-id sweep-01 --total-steps 2000 --every-steps 400

Tested 2026-09-09 on gpu-mi355x1-288gb-spot (MEM1), ROCm 7.14, torch rocm7.14.
"""
import argparse
import io
import os
import signal
import sys
import time

import boto3
import torch
import torch.nn as nn

FLUSH_FLAG = "/tmp/checkpoint-now"
_exit_requested = False


def _on_signal(signum, frame):
    global _exit_requested
    _exit_requested = True
    print(f"[signal] {signal.Signals(signum).name} received, will checkpoint and exit",
          flush=True)


def spaces_client():
    region = os.environ["SPACES_REGION"]
    return boto3.client(
        "s3",
        region_name=region,
        endpoint_url=f"https://{region}.digitaloceanspaces.com",
        aws_access_key_id=os.environ["SPACES_KEY"],
        aws_secret_access_key=os.environ["SPACES_SECRET"],
    )


def save_checkpoint(s3, bucket, run_id, step, model, optimizer, keep):
    t0 = time.time()
    buf = io.BytesIO()
    torch.save({"step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict()}, buf)
    size_mb = buf.tell() / 1e6
    buf.seek(0)
    key = f"checkpoints/{run_id}/step-{step:09d}.pt"
    s3.upload_fileobj(buf, bucket, key)
    # Write the pointer only after the object is fully uploaded, so LATEST
    # never names a half-written checkpoint.
    s3.put_object(Bucket=bucket, Key=f"checkpoints/{run_id}/LATEST", Body=key.encode())
    dt = time.time() - t0
    print(f"[checkpoint] step {step} -> s3://{bucket}/{key} "
          f"({size_mb:,.0f} MB in {dt:.1f}s)", flush=True)
    prune_old(s3, bucket, run_id, keep)
    return dt


def prune_old(s3, bucket, run_id, keep):
    prefix = f"checkpoints/{run_id}/step-"
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    keys = sorted(o["Key"] for o in resp.get("Contents", []))
    for old in keys[:-keep]:
        s3.delete_object(Bucket=bucket, Key=old)
        print(f"[prune] deleted {old}", flush=True)


def load_latest(s3, bucket, run_id, model, optimizer):
    try:
        latest = s3.get_object(Bucket=bucket, Key=f"checkpoints/{run_id}/LATEST")
    except s3.exceptions.NoSuchKey:
        print(f"[resume] no checkpoint for run-id={run_id}, starting fresh", flush=True)
        return 0
    key = latest["Body"].read().decode()
    t0 = time.time()
    buf = io.BytesIO()
    s3.download_fileobj(bucket, key, buf)
    buf.seek(0)
    ckpt = torch.load(buf, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"[resume] restored {key} in {time.time() - t0:.1f}s, "
          f"continuing from step {ckpt['step']}", flush=True)
    return ckpt["step"]


class StandInModel(nn.Module):
    """A stack of MLP blocks sized so one step is real GPU work (~200M params).
    Replace with your model; nothing else in this file cares what it is."""

    def __init__(self, d_model=2048, d_hidden=8192, layers=6):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_hidden),
                          nn.GELU(), nn.Linear(d_hidden, d_model))
            for _ in range(layers)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = x + blk(x)
        return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--total-steps", type=int, default=2000)
    ap.add_argument("--every-steps", type=int, default=400)
    ap.add_argument("--keep", type=int, default=3, help="checkpoints to retain in Spaces")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seq", type=int, default=512)
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    if not torch.cuda.is_available():
        sys.exit("no GPU visible to torch (on AMD: check rocm-smi and the rocm wheel index)")
    device = "cuda"  # ROCm exposes the GPU through the same torch.cuda API
    print(f"[gpu] {torch.cuda.get_device_name(0)}, torch {torch.__version__}, "
          f"hip {torch.version.hip}", flush=True)

    torch.manual_seed(0)
    model = StandInModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params / 1e6:,.0f}M params", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    s3 = spaces_client()
    bucket = os.environ["SPACES_BUCKET"]
    step = load_latest(s3, bucket, args.run_id, model, optimizer)
    start_step, t_start, ckpt_time = step, time.time(), 0.0

    while step < args.total_steps:
        x = torch.randn(args.batch, args.seq, 2048, device=device)
        target = torch.roll(x, 1, dims=1)
        loss = loss_fn(model(x), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step += 1

        if step % 50 == 0:
            torch.cuda.synchronize()
            rate = (step - start_step) / (time.time() - t_start)
            print(f"[train] step {step}/{args.total_steps} loss {loss.item():.4f} "
                  f"{rate:.2f} steps/s", flush=True)

        flag = os.path.exists(FLUSH_FLAG)
        if (_exit_requested or flag or step % args.every_steps == 0
                or step == args.total_steps):
            torch.cuda.synchronize()
            ckpt_time += save_checkpoint(s3, bucket, args.run_id, step, model,
                                         optimizer, args.keep)
            if flag:
                os.remove(FLUSH_FLAG)
            if _exit_requested:
                break

    wall = time.time() - t_start
    done = step - start_step
    print(f"[summary] ran steps {start_step}->{step} in {wall:.0f}s wall, "
          f"{ckpt_time:.0f}s of that in checkpoints "
          f"({100 * ckpt_time / max(wall, 1e-9):.1f}% tax), "
          f"{done / max(wall - ckpt_time, 1e-9):.2f} steps/s while training",
          flush=True)
    if _exit_requested:
        print("[exit] checkpointed and exiting cleanly; rerun the same command to resume",
              flush=True)
        sys.exit(0)
    print(f"[done] {args.total_steps} steps complete", flush=True)


if __name__ == "__main__":
    main()
