#!/usr/bin/env python3
"""A stand-in for the GitHub CLI in the share tests: it records every call in $FAKE_GH_LOG and answers like gh would
for the calls `enginesmith share` makes. With FAKE_GH_NO_CATEGORY set, EngineSmith's Discussions have no Engines
category."""
import json
import os
import subprocess
import sys
from pathlib import Path

args = sys.argv[1:]
with open(os.environ['FAKE_GH_LOG'], 'a') as log:
    log.write(json.dumps(args) + '\n')
if args[:2] == ['api', 'user']:
    print('alice')
elif args[:2] == ['api', 'graphql'] and 'createDiscussion' in args[3]:
    body = Path(next(a for a in args if a.startswith('body=@')).removeprefix('body=@')).read_text()
    Path(os.environ['FAKE_GH_LOG']).with_suffix('.body.md').write_text(body)
    print(json.dumps({'data': {'createDiscussion': {'discussion': {
        'url': 'https://github.com/EngineSmith/enginesmith/discussions/7'}}}}))
elif args[:2] == ['api', 'graphql']:
    nodes = [] if os.environ.get('FAKE_GH_NO_CATEGORY') else [{'id': 'C1', 'slug': 'engines'}]
    print(json.dumps({'data': {'repository': {'id': 'R1', 'discussionCategories': {'nodes': nodes}}}}))
elif args[:2] == ['repo', 'create']:
    source = Path(args[args.index('--source') + 1])
    assert subprocess.run(['git', 'log', '-1'], cwd=source, capture_output=True).returncode == 0
    print(f'https://github.com/{args[2]}')
else:
    sys.exit(f'fake gh: unexpected {args}')
