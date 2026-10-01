#!/usr/bin/env python3
"""Validate configured GitHub GPU runners before starting a release build."""
import json
import os
from pathlib import PurePosixPath
import re

FAMILIES = {'k80': 'cuda11-legacy', 'm10': 'cuda12-maxwell',
            'm60': 'cuda12-maxwell', 'rtx': 'cuda13-rtx'}


def validate(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError('OLLAMA_GPU_TEST_TARGETS must list the provisioned GPU runners')
    seen = set()
    for row in rows:
        if row['family'] not in FAMILIES or row['variant'] != FAMILIES[row['family']]:
            raise ValueError('Hardware family and CUDA image do not match')
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,40}', row['id']) or row['id'] in seen:
            raise ValueError('Unique target IDs are required')
        seen.add(row['id'])
        if (not isinstance(row['runner'], list) or 'self-hosted' not in row['runner']
                or len(row['runner']) < 2 or not all(isinstance(v, str) and v for v in row['runner'])):
            raise ValueError('Use provisioned self-hosted runner labels, including a GPU host label')
        if not PurePosixPath(row['config']).is_absolute():
            raise ValueError('The host-owned target configuration must use an absolute path')
        if type(row['deploy']) is not bool:
            raise ValueError('Explicit deploy=true/false is required for every target')
    if {row['family'] for row in rows} != set(FAMILIES):
        raise ValueError('K80, M10, M60 and RTX hardware gates are all required')
    return {'include': rows}


if __name__ == '__main__':
    print(json.dumps(validate(json.loads(os.environ.get('OLLAMA_GPU_TEST_TARGETS', '[]')))))
