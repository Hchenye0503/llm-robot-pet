"""OwnerState regression on camera views not used while writing the field definitions.

The definitions were refined on two views (2 m eye level, 0.6 m sitting and looking up), so better scores there do not
show that they generalize. This uses four other views, asks REPEAT times per photo, and reports:
    1. the VLM's own noise: how often the REPEAT answers for one photo agree (consistency)
    2. accuracy per field (all REPEAT answers counted)
Each view has 11 photos (absent + 10 poses), so one photo is ~9 percentage points.

Views (the room is only ~2.1 m deep)
    1 standing, eye level, 1.2 m from the toes, front      2 standing, 1.8 m, front
    3 standing, 1.5 m, 45 deg to the side                  4 sitting, looking up 35 deg, 0.9 m, front
At 1.8 m the duck backs up against the wall for some poses and sees only the wall (present = False on 4 photos).

    python -m eval.owner_state       # needs OPENAI_API_KEY; 4 x 11 x 3 = 132 VLM calls
"""

import math
from collections import Counter

import mujoco
from PIL import Image

from pet.body import sim as S
from pet.body.scheduler import hold, run_steps
from pet.perception import geometry as G
from pet.perception.loop import ask_state
from pet.perception.vision_world import mount_camera, shoot
from pet.planner.schema import SitStand
from pet.world import owner as O

FIELDS = ("present", "posture", "arms", "head", "facing_duck")
VIEWS = [("1 standing 1.2 m front", "stand", 1.2, 0, 0), ("2 standing 1.8 m front", "stand", 1.8, 0, 0),
         ("3 standing 1.5 m side 45", "stand", 1.5, 45, 0), ("4 sitting looking up 0.9 m", "sit", 0.9, 0, -35)]
REPEAT = 3


def toe(name):
    """How far the owner's toes are from the pelvis (owner facing +x)."""
    return 0.565 if name in O.SITTING else 0.92 if name == "lie" else 0.175


def main():
    sim = S.DuckSim(O.T.SCENE, None, spec_edit=O.stage_live, start=(O.SIT_SPOT[0] + 1.5, O.SIT_SPOT[1], 180.0))
    owner = O.Owner(sim, pose="sit", at=O.SIT_SPOT, yaw=0.0)
    for _ in range(50):
        sim.step("walk", S.zeros())
    mount_camera(sim)
    rend = mujoco.Renderer(sim.model, G.H_PX, G.W)
    total = {}
    for view, posture, dist, ang, pitch in VIEWS:
        right, n, same, items = Counter(), 0, 0, 0
        for name in ("absent",) + O.POSES:
            a = math.radians(ang)
            tx, ty = O.SIT_SPOT[0] + toe(name), O.SIT_SPOT[1]
            dx, dy = tx + dist * math.cos(a), ty + dist * math.sin(a)
            yaw = math.degrees(math.atan2(O.SIT_SPOT[1] - dy, O.SIT_SPOT[0] + 0.1 - dx))   # face the owner
            sim.start = (dx, dy, yaw)
            sim.reset()
            owner.xy = O.AWAY if name == "absent" else O.SIT_SPOT
            if name != "absent":
                owner.pose, owner.w = name, 1.0
            for _ in range(50):
                sim.step("walk", S.zeros())
            if posture == "sit":
                run_steps(sim, [SitStand(action="sitstand", posture="sit")], [])
                c = [1.0] + [0.0] * 12
                c[4] = math.radians(pitch)
                hold(sim, ("sitstand", c), 1.5)
            duck = sim.xy_yaw()[:2]
            owner.look = duck
            owner.apply()
            truth = owner.truth(duck)
            img = Image.fromarray(shoot(sim, rend)[0])
            answers = [ask_state(img)[0].model_dump() for _ in range(REPEAT)]
            items += 1
            for f in FIELDS:
                vals = [x[f] for x in answers]
                right[f] += sum(v == truth[f] for v in vals)
                same += len(set(vals)) == 1
            n += REPEAT
            wrong = [f"{f}={Counter(x[f] for x in answers).most_common(1)[0][0]} (true {truth[f]})" for f in FIELDS
                     if Counter(x[f] for x in answers).most_common(1)[0][0] != truth[f]]
            print(f"  {view} {name:<14} {' '.join(wrong) or 'ok'}")
        total[view] = (right, n, same / (items * len(FIELDS)))
        print(f"{view}: " + "  ".join(f"{f} {right[f]}/{n}" for f in FIELDS) + f"   consistency {same / (items * len(FIELDS)):.0%}\n")
    rend.close()
    sim.close()
    print("-- summary (accuracy, all answers counted) --")
    for view, (right, n, cons) in total.items():
        print(f"{view}: " + "  ".join(f"{f} {right[f] / n:.0%}" for f in FIELDS) + f"   consistency {cons:.0%}")


if __name__ == "__main__":
    main()
