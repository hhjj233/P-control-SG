#!/usr/bin/env python3
"""Authenticated NEW natural variable-context highD clip build; no overwrites."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcontrol.data.clips import ClipBuildConfig, build_clip_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    raw = args.config.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != args.config_sha256:
        raise ValueError("v2 build config SHA256 differs from explicitly authorized bytes")
    config = ClipBuildConfig.from_mapping(json.loads(raw))
    if args.validate_only:
        print(json.dumps(dict(status="config_valid", csv_contents_opened=False, outputs_created=False,
            protocol="natural_highd_variable_context_event_clips_v2", recordings=list(config.recordings))))
        return 0
    build_clip_dataset(config, config_sha256=actual)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
