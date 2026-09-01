# VLA82 Manual Teleoperation Launcher Design

## Goal

Provide one command-line launcher that loads any VLA82 selection present in the
installed runtime registry (currently `VLA82-001` through `VLA82-060`) in its
existing RoboCasa / MuJoCo scene, accepts the
standard RoboSuite keyboard controls, and saves an auditable manual episode.

## Scope

The launcher will be `tools/vla82_manual_teleop.py`. It will reuse the
authoritative VLA82 annotation, asset, scene-request, friction, and environment
builders. It will not change an asset, initial placement, friction profile, or
task predicate.

The command line accepts a required `--selection-id`, plus `--seed`,
`--record-dir`, `--camera`, `--max-steps`, sensitivity controls, and an optional
`--record-video` flag. Invalid or unsupported selection IDs fail before a
simulator is created.

## Runtime Flow

1. Load the compiled VLA82 specifications and resolve the requested selection.
2. Build the existing `SceneRequest`, construct the PandaOmron environment, and
   reset it with the requested seed.
3. Create RoboSuite's `Keyboard` controller and show the MuJoCo viewer.
4. On every public simulator step, convert keyboard input to the live
   PandaOmron action vector. The `b` key switches between arm and mobile-base
   control; the same arrow keys then command base X/Y motion and `o`/`p`
   command base yaw.
5. Save each stepped action, timestamp, and scene snapshot metadata. If video
   recording is requested, render the chosen camera to MP4 without changing
   the physics loop.
6. On normal user exit, write `actions.npz`, `episode.json`, `model.xml`, and
   an optional MP4 to a timestamped episode directory.

## Episode Contract

`episode.json` records selection ID, seed, task class, request semantics,
camera, action count, key control description, runtime scene fingerprint, and
friction evidence. `actions.npz` contains the public action vector sequence
and step indices. `model.xml` captures the loaded MuJoCo scene. The episode is
labelled `manual_unverified`; it is not reported as a strict PASS until the
existing predicate evaluator has separately validated it.

## Error Handling

The launcher rejects an unknown selection ID, an empty action stream, missing
camera, unavailable keyboard input, or an invalid video writer with a clear
error. It always closes the environment and finalizes already captured action
data when the user interrupts the session.

## Testing

Pure helper tests cover selection resolution, argument validation, action
recording layout, and metadata construction without opening a MuJoCo GUI. A
help-command check and Python compilation verify that the executable entry
point imports and advertises its options. A GUI session remains a manual
desktop verification because the Codex background session cannot foreground a
Windows MuJoCo viewer.

## Usage

Run this command in a visible Windows PowerShell window, not the Codex
background terminal:

```powershell
cd C:\OpenVLA-Simulator
C:\Users\sjtu101\miniconda3\envs\robocasa\python.exe tools\vla82_manual_teleop.py --selection-id VLA82-017 --record-video
```

Press `b` to toggle arm/base mode. In base mode the arrow keys translate the
PandaOmron base and `o` / `p` rotate it; press `b` again before moving the arm.
Press `q` to finalize and save a non-empty manual capture.
