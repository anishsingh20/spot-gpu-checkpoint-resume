# Checkpoint-and-resume training on DigitalOcean Spot GPU Droplets

A small, runnable PyTorch training loop that survives losing its machine. Every N steps it saves model, optimizer, and step counter to [DigitalOcean Spaces](https://docs.digitalocean.com/products/spaces/). If the [Spot GPU Droplet](https://docs.digitalocean.com/products/droplets/concepts/spot-vs-on-demand/) it runs on is reclaimed, you create a new one with the same command and it picks up where it left off. Nothing else to do.

This is the companion code for the DigitalOcean tutorial [**Fault-Tolerant Training on Spot GPU Droplets: Checkpoint, Resume, and Keep Your Run Alive**](https://www.digitalocean.com/community/tutorials/spot-gpu-droplets-fault-tolerance). Everything in the tutorial's numbers came from running exactly these files on a `gpu-mi355x1-288gb-spot` Droplet in MEM1 on September 9, 2026.

## What is in here

| File | What it does |
| --- | --- |
| `spot_train.py` | The training loop. Checkpoints to Spaces every `--every-steps`, resumes automatically from `checkpoints/<run-id>/LATEST`, flushes on SIGTERM, and flushes on demand when `/tmp/checkpoint-now` exists. |
| `spot-train.service` | systemd unit that runs the script as a service so it survives your SSH session, forwards SIGTERM on stop or shutdown, and waits up to 10 minutes for the final upload. |
| `cloud-init-resume.yaml` | Template for a replacement Droplet. Writes the credentials, the script, and the unit; installs PyTorch; starts training. |
| `make_user_data.py` | Fills the template from a local `KEY=VALUE` env file and inlines `spot_train.py`, so secrets never go into git. |
| `list_checkpoints.py` | Prints the checkpoints in Spaces for one run-id and where `LATEST` points. Handy for checking state from your laptop. |
| `evidence/` | The raw logs from the drill: both Droplets' training logs, Spaces listings, `doctl` output for the destroy and recreate, and the API create response. |

## The idea in three lines

1. Every N steps, upload a checkpoint to Spaces, then update a tiny `LATEST` pointer file. Upload first, pointer second, so `LATEST` never names a half-written file.
2. On start, read `LATEST`. If it exists, download that checkpoint and continue from its step. If not, start fresh.
3. Run the identical command with the identical `--run-id` on whatever Droplet comes next.

That is the whole trick. The SIGTERM handler and the `/tmp/checkpoint-now` flag are conveniences on top: they make a graceful stop (or the two-hour reclaim email) cost zero steps instead of "steps since the last upload."

## Quick start

On a GPU Droplet with ROCm or CUDA and Python 3:

```bash
# 1. PyTorch for your platform. AMD AI/ML Ready image (ROCm 7.14):
python3 -m venv /root/venv
/root/venv/bin/pip install --index-url https://download.pytorch.org/whl/rocm7.14 torch boto3
# NVIDIA AI/ML Ready image: drop --index-url, the default wheel carries CUDA.

# 2. Spaces credentials. Plain KEY=VALUE lines, no `export` (systemd EnvironmentFile syntax).
cat > /root/.spaces.env <<'EOF'
SPACES_KEY=DO00...
SPACES_SECRET=...
SPACES_REGION=nyc3
SPACES_BUCKET=your-checkpoint-bucket
EOF
chmod 600 /root/.spaces.env

# 3. Run it.
set -a; source /root/.spaces.env; set +a
/root/venv/bin/python spot_train.py --run-id sweep-01 --total-steps 3000 --every-steps 600
```

Or as a service (recommended, this is what the tutorial does):

```bash
cp spot_train.py /root/
cp spot-train.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now spot-train
tail -f /root/train.log
```

To simulate a graceful reclaim: `systemctl stop spot-train`. You will see a `[signal]` line, one final `[checkpoint]` line, and a clean exit in about 14 seconds.

To flush without stopping (what to run when the reclaim email arrives): `touch /tmp/checkpoint-now`.

## Bringing up a replacement Droplet

```bash
python3 make_user_data.py --env ~/.spaces.env > /tmp/user-data.yaml
doctl compute droplet create spot-trainer-02 \
  --region mem1 --size gpu-mi355x1-288gb-spot --image gpu-amd-base \
  --ssh-keys $SSH_KEY_FINGERPRINT --user-data-file /tmp/user-data.yaml \
  --tag-names spot-tutorial --wait
```

cloud-init installs the stack and starts the unit. Because the unit carries the same `--run-id`, the new Droplet finds `LATEST` and resumes on its own. In the drill this took 5 minutes 24 seconds from `droplet create` to "training again at step 1900", about four minutes of which was `pip install torch`. Bake a snapshot with the venv preinstalled if you want that closer to one minute.

## What this repo tests, and what it does not

**It tests:**

- That a PyTorch job can checkpoint model + optimizer + step to Spaces on a fixed cadence while training on a spot GPU, and what that costs in wall clock (about 6 percent at 600 steps on a 201M-parameter model, about 3 percent at 1,200).
- That a SIGTERM (from `systemctl stop`, a shutdown, or a reclaim that delivers one) produces a final checkpoint and a clean exit. Measured: 13.7 seconds.
- That a brand-new Droplet, given only the same command and run-id, finds the latest checkpoint and continues with zero steps lost. Measured: resumed from step 1892, finished at 3000, on a different machine.
- That `LATEST` is always safe to follow, because it is written only after the checkpoint object it names is fully uploaded.

**It does not test:**

- A real DigitalOcean-initiated reclaim. There is no button for that, so the drill simulates the two halves separately: a graceful stop, and a hard destroy followed by a recreate. The code path is identical either way.
- Model quality. The stand-in model trains on an unlearnable task on purpose (predict a random tensor from its own shift), so the loss hovers near 1.0. The point is the step counter and the timings. Swap in your model; nothing else in the file cares.
- Multi-node or elastic training. This is one Droplet, one process. The pattern extends to data-parallel jobs, but that is not what is measured here.
- Very large checkpoints. 2.4 GB uploads in about 10 seconds from MEM1 to NYC3. A 7B-parameter model with AdamW is roughly 84 GB per checkpoint, and you would want a longer interval and possibly a sharded upload.

## Adapting it

- **Your model.** Replace `StandInModel` and the two lines that build `x` and `target`. Keep `save_checkpoint` and `load_latest` as they are; they only see `state_dict()`.
- **Your interval.** Time one upload, divide by the share of wall clock you will spend on checkpoints (3 percent is a good default), and that is your training time per interval. See the table in the tutorial.
- **Batch inference instead of training.** The checkpoint becomes a done-list: a manifest object in Spaces listing finished item IDs, rewritten every K items with the same flush triggers. The tutorial has the two functions.
- **DOKS Spot GPU node pools.** Run `spot_train.py` as a Job with a toleration for the spot taint. Kubernetes gives you node conditions and Events you can use to trigger the flush programmatically.

## License

MIT. See `LICENSE`.
