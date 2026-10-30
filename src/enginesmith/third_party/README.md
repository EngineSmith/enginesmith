# Third-party code

## instruction_following_eval

The IFEval checker from google-research (https://github.com/google-research/google-research, directory
`instruction_following_eval`), Apache License 2.0, copied unmodified (each file keeps its license header).
`enginesmith.quality` puts this directory on `sys.path` so the files' own imports (`from instruction_following_eval
import ...`) work as they are. SHA-256 of the copied files:

```
35decc06000718487f44d7deafa6d3f48a8ec0886281edf40162c0265b7d248c  evaluation_lib.py
60e086f5342a03ce8e18b64bbcccf86308f523c08aa826707a562150a52f3edf  instructions.py
ec92d72c264f6d906978613085db262356174300370a3fffe6fefd5969ce9cfc  instructions_registry.py
a73797261eee5bf447e279d82a2b700b1bdd3cb1193412dbab1270a85832bc6b  instructions_util.py
```

`__init__.py` is empty.
