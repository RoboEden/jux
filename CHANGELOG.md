# Change Log

## v3.0.0
Major Change:
  - Migrate to `luxai-s2==3.0.0`.

## v2.2.0
Major Change:
  - Migrate to `luxai-s2==2.2.0`.
  - Python 3.7 is no longer supported. Please use python 3.8 or above.
  - Add support for python 3.11.
  - Migrate to jax 0.4.16. But to be compatible with pytorch, we use `jax[cuda11_cudnn82]==0.4.7` by default.

Fix:
 - Class properties work correctly in batched mode now.

## v2.1.1
Major changes:
  - Upgrade luxai_s2 dependency to v2.1.9.
  - Fix batching issue for data with properties. `env.state.board.valid_spawns_mask` correctly works now.

## v2.1.0
Major changes:
 - Implement same game logic as `luxai-s2==2.1.0`. See [Lux-Design-S2/ChangeLog.md](https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/v2.1.0/ChangeLog.md) for details.
 - `jax` is removed from the dependency list, and need to be installed by user manually. This is because `jax` must be compatible with cuDNN, and only the user knows which version of cuDNN is installed on their machine.

Bug fixes:
 - fix a bug that leads to crash when env seed exceed int32 max range.

## v2.0.0
Major changes:
 - Implement same game logic as `luxai-s2==2.0.0` (except map generation). See [Lux-Design-S2/ChangeLog.md](https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/v2.0.0-official-release/ChangeLog.md) for details.

Bug fixes:
 - Fix nan bug for mountain maps in batched mode


## v1.0 (Dec 26, 2022)

Inital release. JUX is guaranteed to implement the same game logic as `luxai2022==1.1.4`, if players' input actions are valid.
