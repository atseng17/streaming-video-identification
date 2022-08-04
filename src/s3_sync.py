#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to synchronize local files to the s3 bucket upstream.
"""
import argparse
import logging
from pathlib import Path
import subprocess

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


def main(s3, out_path):
    logging.info(
        "Syncing bucket %s to local folder %s. \
        This could take a moment."
        % (s3, out_path)
    )
    logging.info(
        "Please make sure your credentials are properly set up \
        before executing."
    )
    run_recap = out_path.parent / "aws_output.txt"
    command = f"aws s3 sync {s3} {out_path} > {run_recap.as_posix()}"
    subprocess.run(command, shell=True)
    logging.info("Done Syncing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sync local files to AWS s3 bucket of choice"
    )
    parser.add_argument(
        "--s3", type=str, default=None, help="link to s3 bucket in which to sync."
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/sleep_data/",
        help="Name of local output directory to sync to.",
    )
    args = parser.parse_args()
    _out_path = Path(args.out)
    main(args.s3, _out_path)
