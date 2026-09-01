# Chapter 5 Task/Object Archive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Archive the objects specified in Chapter 5 sections 5.2.2–5.2.15 without mistaking network-control metadata for VLA training data.

**Architecture:** Produce a manifest from the revised Chapter 5 taxonomy.  Network-controlled sections receive interface-only object folders; VLA sections receive object folders only after a source instruction, RGB clip, and action segment are verified to be from the same episode.

**Tech Stack:** Python, JSON, Parquet, MP4, RoboCasa LeRobot data.

## Global Constraints

- Preserve raw shared RoboCasa chunks; derive object-level clips rather than modifying source chunks.
- Do not include human video, detection images, or metadata-only records as VLA samples.
- Keep network-control records separate from VLA training records.
- Archive only tasks 5.2.2–5.2.15.

---

### Task 1: Build and validate a Chapter 5 archive manifest

**Files:**
- Create: `outputs/chapter5_archive/chapter5_5_2_2_to_5_2_15_manifest.json`
- Create: `outputs/chapter5_archive/chapter5_5_2_2_to_5_2_15_manifest_report.json`

- [ ] Extract the 14 task rows and their 8 objects from `第五章（中期验收版_修订）.docx`.
- [ ] Assert exactly 14 task rows and 112 object rows.
- [ ] Mark 5.2.2–5.2.4 as `network_control_only`; mark remaining tasks as `vla_candidate`.

### Task 2: Match VLA candidates to paired source episodes

**Files:**
- Create: `outputs/chapter5_archive/vla_pair_evidence.json`

- [ ] Search official RoboCasa episode text before any extraction.
- [ ] Require an exact or documented canonical object match plus source RGB and action availability.
- [ ] Record unmatched candidates as `pending_source_evidence`, without producing a training folder.

### Task 3: Derive and archive verified VLA object samples

**Files:**
- Create: `datasets/chapter5_02_*` through `datasets/chapter5_15_*`

- [ ] Preflight destinations and unique episode allocation.
- [ ] Extract matching RGB/action segments to object folders.
- [ ] Write `instruction.json`, `source_manifest.json`, and `conversion_status.json` with each verified sample.

### Task 4: Re-verify and report

**Files:**
- Create: `outputs/chapter5_archive/archive_verification_report.json`

- [ ] Verify each VLA folder has a matching instruction, RGB clip, and action Parquet.
- [ ] Verify network-control folders are excluded from VLA counts.
- [ ] Report verified, pending, and excluded counts per task.
