#!/usr/bin/env python3
"""List the checkpoints for one run in Spaces and show where LATEST points.

    source /root/.spaces.env && python3 list_checkpoints.py sweep-01
"""
import os
import sys

import boto3

run_id = sys.argv[1] if len(sys.argv) > 1 else "sweep-01"
region = os.environ["SPACES_REGION"]
bucket = os.environ["SPACES_BUCKET"]
s3 = boto3.client("s3", region_name=region,
                  endpoint_url=f"https://{region}.digitaloceanspaces.com",
                  aws_access_key_id=os.environ["SPACES_KEY"],
                  aws_secret_access_key=os.environ["SPACES_SECRET"])

prefix = f"checkpoints/{run_id}/"
for o in s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", []):
    print(f"{o['Size'] / 1e6:10,.1f} MB  {o['LastModified']:%Y-%m-%d %H:%M:%S}Z  {o['Key']}")
try:
    latest = s3.get_object(Bucket=bucket, Key=prefix + "LATEST")["Body"].read().decode()
    print(f"LATEST -> {latest}")
except s3.exceptions.NoSuchKey:
    print("LATEST -> (none)")
