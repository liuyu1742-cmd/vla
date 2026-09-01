import fs from "node:fs/promises";

const target =
  "C:/OpenVLA-Simulator/tools/run_vla82_openvla_600_augmentation_batch.py";
let source = await fs.readFile(target, "utf8");

const oldImports = `import statistics
import time`;
const newImports = `import statistics
import sys
import time`;
const oldRoot = `from tools.run_vla82_openvla_60_policy_probe import build_instruction


ROOT = Path(__file__).resolve().parents[1]`;
const newRoot = `ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.run_vla82_openvla_60_policy_probe import build_instruction`;

if (!source.includes(oldImports) || !source.includes(oldRoot)) {
  throw new Error("expected entry-point source pattern not found");
}
source = source.replace(oldImports, newImports).replace(oldRoot, newRoot);
await fs.writeFile(target, source, "utf8");
console.log("UPDATED: OpenVLA 600-case direct entry point");
