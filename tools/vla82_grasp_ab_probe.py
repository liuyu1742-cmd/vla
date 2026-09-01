"""Step-only VLA82-002 gripper hold / horizontal-transit A/B probe."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from tools.run_vla82_full_simulation import _load_compiled_specs, _scene_for_spec
from tools.vla82_full_sim.environment import make_environment, build_physics_capture_contract, scene_runtime_fingerprint
from tools.vla82_full_sim.expert import CleaningPrimitive, _raw_environment, _selected_geom_has_two_finger_contacts
from tools.pick_place_oracle.open_gripper import OpenGripperPickPlaceOracle

OUT=Path("outputs/midterm_testing_vla82/full_simulation_objectwise_8x60/diagnostics/vla82_002_grasp_ab.json")
SEED=2000
def row(raw, action, step):
 m,d=raw.sim.model,raw.sim.data; robot=raw.robots[0]; obj=raw.obj_body_id['obj']; geoms=tuple(raw.objects['obj'].get_obj().worldbody.find_all('geom')) if False else ('obj_collision',)
 fingers=[i for i in range(m.nu) if 'gripper' in (m.actuator_id2name(i) or '').lower()]
 pads=[]; counter=False
 for i in range(d.ncon):
  c=d.contact[i]; a=m.geom_id2name(c.geom1) or ''; b=m.geom_id2name(c.geom2) or ''
  if 'obj' in a or 'obj' in b:
   pads.append({'pair':[a,b],'normal':float(d.efc_force[c.efc_address]) if c.efc_address>=0 else 0.0})
   counter |= 'counter' in a or 'counter' in b
 return {'step':step,'action':np.asarray(action).round(6).tolist(),'qpos':d.qpos.copy().round(8).tolist(),'qvel':d.qvel.copy().round(8).tolist(),'gripper_ctrl':[float(d.ctrl[i]) for i in fingers],'gripper_force':[float(d.actuator_force[i]) for i in fingers],'contacts':pads,'obj_counter':counter,'obj_velocity':d.cvel[obj].copy().round(8).tolist(),'held':bool(raw._check_grasp(robot.gripper['right'],raw.objects['obj'])),'two_pad':_selected_geom_has_two_finger_contacts(raw,('obj_collision',))}
def run(kind):
 spec=next(x for x in _load_compiled_specs() if x.selection_id=='VLA82-002'); scene=_scene_for_spec(spec); env=make_environment(scene,SEED); env.reset(seed=SEED); raw=_raw_environment(env); contract=build_physics_capture_contract(env,scene); prim=CleaningPrimitive(spec.phases,contract); gen=prim.actions(env); held=False
 for _ in range(180):
  _,a=next(gen); env.step(a); held=bool(raw._check_grasp(raw.robots[0].gripper['right'],raw.objects['obj']))
  if held: break
 if not held: raise RuntimeError('failed_pregrasp')
 eef=raw.sim.data.site_xpos[raw.robots[0].eef_site_id['right']].copy(); obj=raw.sim.data.body_xpos[raw.obj_body_id['obj']].copy(); data=[]
 for step in range(20):
  if kind=='B':
   oracle=OpenGripperPickPlaceOracle(obj+np.array((-.05,0,0)),obj,obj,world_to_origin=raw.robots[0].composite_controller.part_controllers['right'].world_to_origin_frame); a=np.zeros(12); a[:7]=oracle._motion(type('S',(),{'eef_position':eef})(),obj+np.array((-.05,0,0)),closed=True).action; a[11]=-1
  else:
   a=np.zeros(12); a[:3]=np.array((-.1,0,0)); a[6]=1; a[11]=-1
  env.step(a); data.append(row(raw,a,step)); eef=raw.sim.data.site_xpos[raw.robots[0].eef_site_id['right']].copy()
 fp=scene_runtime_fingerprint(env,scene); env.close(); return {'kind':kind,'fingerprint':fp['sha256'],'rows':data}
if __name__=='__main__':
 OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps({'seed':SEED,'A':run('A'),'B':run('B')},ensure_ascii=False,indent=2),encoding='utf-8'); print(OUT)
