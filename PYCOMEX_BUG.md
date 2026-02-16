# PyComex Bug: `Experiment.extend()` crashes when base experiment uses `@experiment.testing`

**Version**: pycomex 0.9.5

## Summary

`Experiment.extend()` raises `TypeError: object of type 'function' has no len()` when the base experiment registers a `@experiment.testing` hook.

## Root Cause

`Experiment.testing()` stores the callback as a bare function:

```python
# pycomex/functional/experiment.py — Experiment.testing()
self.hook_map["__TESTING__"] = func
```

But `Experiment.hook()` wraps callbacks in a list:

```python
# pycomex/functional/experiment.py — Experiment.hook()
self.hook_map[name] = [func]
```

`read_module_metadata()` iterates all `hook_map` entries and assumes they are lists:

```python
for hook, func_list in self.hook_map.items():
    if hook not in self.metadata["hooks"]:
        self.metadata["hooks"][hook] = {
            "name": hook,
            "num": len(func_list),  # <-- TypeError on bare function
        }
```

## Reproduction

The first `read_module_metadata()` call (in `Experiment.__init__`) succeeds because `@experiment.testing` hasn't been registered yet. But when a child experiment calls `Experiment.extend()`, the base module is fully imported first (including its `@experiment.testing` decorator), so by the time `extend()` calls `read_module_metadata()` again, `hook_map["__TESTING__"]` is a bare function.

```python
# base_experiment.py
experiment = Experiment(...)

@experiment.testing
def testing(e):
    e.PARAM = 1

# child_experiment.py
experiment = Experiment.extend("base_experiment.py", ...)  # <-- crashes here
```

## Suggested Fix

Either wrap the function in a list in `Experiment.testing()`:

```python
self.hook_map["__TESTING__"] = [func]  # consistent with hook()
```

Or guard `read_module_metadata()` against non-list entries:

```python
for hook, func_list in self.hook_map.items():
    num = len(func_list) if isinstance(func_list, list) else 1
```

## Workaround

Do not define `@experiment.testing` in child experiments that use `Experiment.extend()`. Set testing parameters as module-level defaults instead.
