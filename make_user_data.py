#!/usr/bin/env python3
"""Build the cloud-init user-data for a replacement Spot GPU Droplet.

Reads cloud-init-resume.yaml (template), inlines spot_train.py, and fills the
Spaces credentials from a KEY=VALUE env file that never enters the repo.

    python3 make_user_data.py --env ~/.spaces.env > /tmp/user-data.yaml
"""
import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).parent


def read_env(path):
    env = {}
    for line in pathlib.Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, help="KEY=VALUE file with SPACES_KEY/SECRET/REGION/BUCKET")
    ap.add_argument("--template", default=HERE / "cloud-init-resume.yaml")
    ap.add_argument("--script", default=HERE / "spot_train.py")
    args = ap.parse_args()

    env = read_env(args.env)
    missing = [k for k in ("SPACES_KEY", "SPACES_SECRET", "SPACES_REGION", "SPACES_BUCKET") if k not in env]
    if missing:
        sys.exit(f"missing in {args.env}: {', '.join(missing)}")

    script = pathlib.Path(args.script).read_text().rstrip("\n")
    indented = "\n".join(("      " + ln) if ln else "" for ln in script.splitlines())
    out = pathlib.Path(args.template).read_text().replace("__SPOT_TRAIN_PY__", indented)
    for k, v in env.items():
        out = out.replace(f"__{k}__", v)
    if "__SPACES" in out:
        sys.exit("unfilled placeholder left in template")
    sys.stdout.write(out)


if __name__ == "__main__":
    main()
