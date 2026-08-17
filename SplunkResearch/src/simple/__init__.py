"""simple/ — a lean, single-purpose rewrite of the DRL log-injection framework.

One flat environment, one action decode, one reward, one measurement switch
(mock | live), one orchestrator. No wrapper stack, no strategy registries, no
reward-mode branching. See SplunkResearch/src/simple/README.md for the map.

The heavy legacy package under SplunkResearch/src/wrappers/ is left untouched for
reference; nothing here imports from it except a few proven, stateless helpers.
"""
