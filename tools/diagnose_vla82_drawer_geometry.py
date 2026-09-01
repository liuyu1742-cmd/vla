"""Print live MuJoCo drawer collision geometry for one VLA82 scene."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_vla82_full_simulation import _load_compiled_specs, _scene_for_spec
from tools.vla82_full_sim.environment import build_physics_capture_contract, make_environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-id", default="VLA82-006")
    parser.add_argument("--seed", type=int, default=820000)
    args = parser.parse_args()
    spec = next(item for item in _load_compiled_specs() if item.selection_id == args.selection_id)
    scene = _scene_for_spec(spec)
    environment = make_environment(scene, seed=args.seed)
    try:
        environment.reset(seed=args.seed)
        contract = build_physics_capture_contract(environment, scene)
        raw = environment.unwrapped.env
        model, data = raw.sim.model, raw.sim.data
        target_id = next(iter(contract.target_geometries))
        fixture = target_id.split(":", 1)[0]
        print("target", target_id, contract.target_geometries[target_id])
        print("target_geoms", contract.target_geom_names[target_id])
        print("object_geoms", contract.object_geom_names)
        for index in range(int(model.ngeom)):
            name = model.geom_id2name(index) or ""
            if fixture not in name and name not in {
                geom for geoms in contract.object_geom_names.values() for geom in geoms
            }:
                continue
            print(
                name,
                "pos=", data.geom_xpos[index].round(6).tolist(),
                "size=", model.geom_size[index].round(6).tolist(),
                "type=", int(model.geom_type[index]),
                "contype=", int(model.geom_contype[index]),
                "conaffinity=", int(model.geom_conaffinity[index]),
            )
    finally:
        environment.close()


if __name__ == "__main__":
    main()
