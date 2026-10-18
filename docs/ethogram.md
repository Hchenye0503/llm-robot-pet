# Pet Ethogram → Robot Skills

Before writing any planner or training any policy, I asked a simpler question: **what should a robot pet actually do?**
The answer is an *ethogram*: a catalogue of a species' behaviors, each with an **operational definition**
(an observable start, end and threshold, so two people watching the same clip would agree whether it happened).

This document lists 26 dog (and two cat) behaviors from the animal-behavior literature, then maps each one onto what
Microduck's body and policies can do. Everything else in this repo follows from it:
the behavior library the LLM picks from ([`pet/behaviors.py`](../pet/behaviors.py)), the head gestures, and the
movement problems that motivated training a new gait (see [last section](#what-the-language-layer-revealed-about-movement)).

> Thresholds (seconds, meters, degrees) are illustrative values chosen for the robot, not standard values from the literature.

## Classification rule

**One criterion: does the behavior *itself* require switching policies?**

| Class | Meaning |
|---|---|
| ① | Uses one existing policy. Parameters may be fixed or vary over time (keyframes + interpolation, no retraining) |
| ② | The behavior contains a policy switch (e.g. walk → sit/stand, walk → kick) |
| ③ | Needs a newly trained policy |
| ✗ | The body has no matching part → design a substitute expression, then classify the substitute with ①–③ |

Posture changes *between* behaviors (sit → stand) are not part of a behavior: the scheduler inserts them automatically as
preconditions. Without this rule, an earlier draft classified head tilt (#4) and response to petting (#8) differently
even though they have the same structure.

## Posture strategy

Dogs lie down when nothing is happening, but most interactive behaviors are performed standing.

```
Idle (sitting, head can move) ──any event / behavior that needs to move──▶ Standing (interacting)
        ▲                                                                       │
        └──────────── no event for T_idle seconds (hysteresis, 20–30 s) ────────┘
```

- Head-only behaviors (#1, #2, #4, #15 and the substitutes for #23, #25) can be done sitting.
- Standing up is itself a signal to the owner ("I'm interested").
- T_idle is long on purpose: it prevents sit-stand-sit chattering (same idea as a Schmitt trigger).

## Body constraints (verified)

- Microduck has 14 servos, a head camera, an IMU and an 8×8 ToF sensor.
- In simulation the beak is a **passive joint with no actuator**, so there is no grasping.
- No tongue, eyelids or tail → behaviors that rely on them get substitutes or are dropped.
- The walking policy is trained to stand still on a zero command, so "standing" needs no separate policy.

## State interface

```
Owner state = {posture, arms, head, facing the robot, region, event:
               came home | called name | petting | scolding | toy out | leaving | none}
Inner state = {energy E, curiosity C, attachment A, stress S}, each 0–1
```

In the implementation the owner state contains only **visible** actions (posture, arms, head); "may be tired or sad"
is inferred from a visible cue held for 30 s, never read directly from a single photo.

## The table

Evidence: 📗 peer-reviewed study · 📙 practitioner / grey literature · 📕 popular belief, no peer-reviewed source.
Sources are numbered, see [References](#references); † = the source supports the behavior only partly (see its note).
"In this repo": what the LLM can actually select (behavior = moves the body; gesture = head only).

| # | Behavior | Operational definition | Trigger: owner | Trigger: inner | How the robot does it | Class | Evidence | Source | In this repo |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Gaze at owner | Head faces owner's face ≥ 1 s | Looking at it, or name called; near/mid | A ≥ mid | Head yaw = owner bearing, pitch up 20–30°; ends 1 s after owner looks away | ① | 📗 | [1] [2] | `gaze` |
| 2 | Gaze alternation | Target → owner → target ≥ 2× within 10 s | Owner present | C high + unreachable object | Head yaw switches between target and owner 3×, 1 s each | ① | 📗 | [3] | — |
| 3 | Social referencing | Looks at owner within 3 s of facing a new object | Owner present; then owner's mood | C mid, S mid | Head to object 1 s → owner 1–2 s → happy: #21, angry: back off | ① | 📗 | [4] [5] | — |
| 4 | Head tilt | Head rolls > 10° then returns | Name called / spoken to | C ≥ mid | Head roll 15° (max 18°), hold 1–1.5 s; side = personality parameter | ① | 📗 (exploratory) | [6] | gesture `head_tilt` |
| 5 | Greeting | Within 10 s of arrival: approach to < 1 m + high activity | Came home (stronger after longer absence) | A high | Walk to owner → happy head motion + tail-wag substitute | ① | 📗 | [7] [8] | `greet` |
| 6 | Following | Moves with owner, < 2 m, ≥ 3 s | Owner leaving | A high, E ≥ mid | Walk toward owner's position, head on owner; stop when owner stops | ① | 📗 | [9]† | — |
| 7 | Proximity seeking | Stays < 0.5 m for ≥ 10 s | Owner sitting/lying; tired, sad or calm | A high, E low–mid | Walk to 0.4 m → sit → occasional #1, #4 | ② | 📗 | [9] | `stay_close` |
| 8 | Response to petting | Stays close vs. freezes and moves away | Petting | A high → likes; S high or A low → dislikes | Likes: hold still, head tilts up; dislikes: #20 → back off 0.2 m + #15 | ① | 📗 | [10] | — (no touch sensing) |
| 9 | Play bow | Forelimbs bent, chest low, rear high, 1–3 s | Looking at it, pause in play | E high | No front legs → substitute: deep head dip + quick rebound + step back | ✗ → ② | 📗 | [11] [12] [13] | `play_bow` (with #10) |
| 10 | Bounce | Dash in → back off ≥ 2× within 2 s | Looking at it | E high | Forward 0.5 s → back 0.5 s ×2, head down 20° | ① | 📙 | Trainer practice | `play_bow` (with #9) |
| 11 | Object presentation | Carries object to owner and drops it | Owner free, looking | E high | Needs grasp + carry + release; beak has no actuator → substitute #12 | ③ | 📙 | — | — |
| 12 | Object play / kick to owner | Repeatedly pushes or chases object ≥ 5 s | Ball present; kick toward owner if watched | E high, C high | Walk to ball → kick toward owner → #1 | ② | 📗 | [14]† | — (kick not wired to a ball scene) |
| 13 | Rollover | Back on ground, belly up | Play / petting | E high | Substitute: sit down (merged into #22) | ✗ → ① | 📗 (debated) | [15] | via `rest` |
| 14 | Zoomies | Sudden fast circling ≥ 10 s, abrupt stop | Owner not engaging | E very high (after long rest) | Fast walk + random turns + spin-hop, 10–15 s → sudden sit | ② | 📙 / 📕 | — | `zoomies` |
| 15 | Looking away | Head turns away > 30° within 1 s | Owner angry, or stared at up close | S mid–high | Head yaw away 40–60°, hold 1–2 s, slowly return | ① | 📗 | [16] | gesture `shy` |
| 16 | Lip licking | Tongue licks lip/nose, no food | Owner angry | S mid | No tongue; #15 takes over the appeasement role | ✗ | 📗 | [16] [17] | — |
| 17 | Yawning | Mouth fully open ~1 s | Owner yawns / tired | E low or S mid | Needs a mouth actuator | ③ (unconfirmed) | 📗 contagion / 📙 stress | [18] | — |
| 18 | "Guilty look" | Head low + gaze averted + body low + slow approach | Owner angry / scolding | S mid | Slow walk + head pitch −30° + yaw away 20°; may end sitting | ① | 📗 | [19] | `guilty` |
| 19 | Lowered posture | Body and head clearly lower than normal ≥ 2 s | Sudden noise / falling object | S high | Approximation: sit + head fully down; real crouch = ③ | ② (approx.) | 📗 | [20] | — |
| 20 | Freezing | Whole body still ≥ 2 s | Unwanted touch, sudden stimulus | S high | Zero command + head pose frozen 2 s | ① | 📗 | [10] | — |
| 21 | Investigating | Approaches to 0.3 m + head down ≥ 2 s | Owner present (often after #3) | C high, S low | Walk to 0.3 m → head down + small left-right scan 2–3 s | ① | 📗 | [14] | `investigate` |
| 22 | Resting | Lying or sitting still ≥ 30 s | Owner resting, quiet | E low | Sit + head slowly down 20°; stand up when E recovers or name called | ① | 📗 | General | `rest` |
| 23 | Slow blink (cat) | Half-blink / squint sequence | Owner slow-blinks, soft gaze | A high, S low | No eyelids → slow nod facing owner (pitch −10° over 1 s, back 1 s) | ✗ → ① | 📗 | [21] | gesture `nod` |
| 24 | Head rubbing (cat) | Rubs head/cheek on target | Owner sitting, legs nearby | A high | Real contact = ③ (force control, fall risk); substitute: stop at 0.1 m + tilt head toward leg | ③ / ① | 📗 (cat-to-cat) | [22]† | — |
| 25 | Tail wag | Tail swings side to side | Sees owner | A high, E ≥ mid | No tail → "head wag": yaw ±10–15° at ~2 Hz for 1–2 s | ✗ → ① | 📗 | [23] [24] | gesture `wag` |
| 26 | Spinning hop | ≥ 3 consecutive two-foot hops with ≥ 180° accumulated rotation; ends after > 1 s without a two-foot flight; excludes in-place spin, getting up, walking | Came home / toy out | E very high, A high | Switch to the spin-hop policy for a 1–2 s burst → back to walk | ① (sim only) | 📕 | Own RL experiment | inside `zoomies` |

#26 was not taken from the literature. It came from a reward-hacking side effect in an earlier gait experiment: the policy
learned to hop and spin. I wrote an operational definition for it and kept it as a trick.

## What the language layer revealed about movement

Running the behaviors in simulation (LLM → validator → scheduler → policy) exposed problems that the planner could only
work around. They became the requirements for training a new gait.

| # | Problem (measured) | Workaround in the planner | Outcome |
|---|---|---|---|
| 1 | **Can't look up at a standing person**: head command 40° → only 26° | Sit down, then look up | Not a policy problem: the joint reaches 37.9°/40°; it is linkage geometry (57° joint → 36° camera). Solved without training by leaning the body back 15° (`alpha_stand`), now used in `gaze` and `greet` |
| 2 | **Head oscillation lags**: tracks 85% at 0.5 Hz, 77% at 1 Hz, 52% at 2 Hz | Slower, larger gestures | Open |
| 3 | **In-place turns overshoot** ~9° after the stop command | IMU closed loop + stop 12° early | New gait slides +0.2 to +1.6° (left) and +5.6 to +6.4° (right) → lead reduced to 6°; turn error now −4 to 0° |
| 4 | **Dead zone**: no motion below 0.2 m/s forward, 0.3 backward, 43°/s turning | Snap small commands up to the dead-zone edge | **New gait walks at 0.05 m/s.** Backward walking got worse, so the planner turns 180° instead of backing up |
| 5 | **Speed tracking**: original policies reach only 50–58% of a 0.3–0.4 m/s command | Sudden stops and turns to look energetic | New gait: 0.366 m/s at a 0.4 m/s command (randomized physics) |
| 6 | **Can't turn while sitting** (falls over) | Only turn the head while sitting | Open |
| 7 | **Sit/stand policy leans when it takes over a standing pose** | Always stand with the walking policy | Open (worked around) |
| 8 | **No front legs for a play bow** | Head keyframes + walk | Open: a "front low, rear high" posture would need a new task |
| 9 | **Feet catch when standing between the owner's legs** | Stop at 0.3 m, stop on contact | Open |
| 10 | **No touch sensing** | Simulator contact list only | Hardware: bump switch or motor current |
| 11 | **Low, fixed camera** (24 cm, tilted down) | Sit and look up (#1) | Hardware: wider lens |
| 12 | **Recovery after a push**: 14/16 | Abandon the plan after a fall | Open |

Items 3–5 were the targets of the pet-gait training (see [`gait/`](../gait/)). Problems 10 and 11 are hardware, not RL.

## References

DOIs and titles were checked against Crossref. The note under each entry says what the source actually supports;
robot-specific numbers (angles, durations, distances) are always our own choices.

1. Miklósi, Á., Kubinyi, E., Topál, J., Gácsi, M., Virányi, Z., & Csányi, V. (2003). A simple reason for a big difference: wolves do not look back at humans, but dogs do. *Current Biology*, 13(9), 763–766. https://doi.org/10.1016/S0960-9822(03)00263-X
   Dogs, unlike hand-raised wolves, turn to look at the human's face when a problem becomes unsolvable. → #1
2. Nagasawa, M., Mitsui, S., En, S., Ohtani, N., Ohta, M., Sakuma, Y., et al. (2015). Oxytocin-gaze positive loop and the coevolution of human-dog bonds. *Science*, 348(6232), 333–336. https://doi.org/10.1126/science.1261022
   Dog-to-owner gaze raises oxytocin in both; gaze is a bonding behavior. → #1
3. Miklósi, Á., Polgárdi, R., Topál, J., & Csányi, V. (2000). Intentional behaviour in dog-human communication: an experimental analysis of "showing" behaviour in the dog. *Animal Cognition*, 3(3), 159–166. https://doi.org/10.1007/s100710000072
   Dogs alternate gaze between hidden food and the owner ("showing"). → #2
4. Merola, I., Prato-Previde, E., & Marshall-Pescini, S. (2012a). Social referencing in dog-owner dyads? *Animal Cognition*, 15(2), 175–185. https://doi.org/10.1007/s10071-011-0443-0
   Facing a novel object, dogs look at the owner and adjust their approach to the owner's emotional message. → #3
5. Merola, I., Prato-Previde, E., & Marshall-Pescini, S. (2012b). Dogs' social referencing towards owners and strangers. *PLoS ONE*, 7(10), e47653. https://doi.org/10.1371/journal.pone.0047653
   Same paradigm: a positive message leads to more approach, a negative one to slower approach, mainly when the informant is the owner. → #3
6. Sommese, A., Miklósi, Á., Pogány, Á., Temesi, A., Dror, S., & Fugazza, C. (2022). An exploratory analysis of head-tilting in dogs. *Animal Cognition*, 25(3), 701–705. https://doi.org/10.1007/s10071-021-01571-8
   Dogs tilted their heads when hearing meaningful verbal requests, and each dog's tilt side stayed the same across months (hence "fixed side = personality parameter"). → #4
7. Rehn, T., & Keeling, L. J. (2011). The effect of time left alone at home on dog welfare. *Applied Animal Behaviour Science*, 129(2–4), 129–135. https://doi.org/10.1016/j.applanim.2010.11.015
   Greeting was more intense after longer separations. → #5
8. Rehn, T., Handlin, L., Uvnäs-Moberg, K., & Keeling, L. J. (2014). Dogs' endocrine and behavioural responses at reunion are affected by how the human initiates contact. *Physiology & Behavior*, 124, 45–53. https://doi.org/10.1016/j.physbeh.2013.10.009
   Reunion behavior depends on how the human greets the dog. → #5
9. Topál, J., Miklósi, Á., Csányi, V., & Dóka, A. (1998). Attachment behavior in dogs (*Canis familiaris*): A new application of Ainsworth's (1969) Strange Situation Test. *Journal of Comparative Psychology*, 112(3), 219–229. https://doi.org/10.1037/0735-7036.112.3.219
   Adult dogs show attachment to the owner: proximity seeking, physical contact, waiting at the door when the owner leaves. → #7. †#6 (following a leaving owner) is our extrapolation; it was coded explicitly in later Strange Situation studies, not in this one.
10. Kuhne, F., Hößler, J. C., & Struwe, R. (2014). Behavioral and cardiac responses by dogs to physical human–dog contact. *Journal of Veterinary Behavior*, 9(3), 93–97. https://doi.org/10.1016/j.jveb.2014.02.006
    Petting elicits measurable responses, including appeasement gestures, freezing (standing motionless) and withdrawal. → #8, #20
11. Bekoff, M. (1995). Play signals as punctuation: the structure of social play in canids. *Behaviour*, 132(5–6), 419–429. https://doi.org/10.1163/156853995X00649
    The play bow is a stereotyped signal that starts and maintains play. → #9
12. Byosiere, S. E., Espinosa, J., & Smuts, B. (2016). Investigating the function of play bows in adult pet dogs (*Canis lupus familiaris*). *Behavioural Processes*, 125, 106–113. https://doi.org/10.1016/j.beproc.2016.02.007
    Play bows tend to follow pauses in play and help restart it. → #9
13. Mitchell, R. W. (2024). Play bows by dogs in dog-human play. *Interaction Studies*, 25(2), 146–166. https://doi.org/10.1075/is.24016.mit
    Dogs also play-bow to humans, not only to other dogs, which is the case that matters for a robot pet. → #9
14. Kaulfuß, P., & Mills, D. S. (2008). Neophilia in domestic dogs (*Canis familiaris*) and its implication for studies of dog cognition. *Animal Cognition*, 11(3), 553–556. https://doi.org/10.1007/s10071-007-0128-x
    Dogs prefer to approach and interact with novel objects. → #21. †#12 extends this to object play.
15. Norman, K., Pellis, S., Barrett, L., & Henzi, S. P. (2015). Down but not out: Supine postures as facilitators of play in domestic dogs. *Behavioural Processes*, 110, 88–95. https://doi.org/10.1016/j.beproc.2014.09.001
    Rolling onto the back during play is mostly a play tactic, not submission. → #13
16. Firnkes, A., Bartels, A., Bidoli, E., & Erhard, M. (2017). Appeasement signals used by dogs during dog–human communication. *Journal of Veterinary Behavior*, 19, 35–44. https://doi.org/10.1016/j.jveb.2016.12.012
    Looking away and lip licking appear when dogs face threatening human behavior. → #15, #16
17. Albuquerque, N., Guo, K., Wilkinson, A., Resende, B., & Mills, D. S. (2018). Mouth-licking by dogs as a response to emotional stimuli. *Behavioural Processes*, 146, 42–45. https://doi.org/10.1016/j.beproc.2017.11.006
    Dogs lick their mouths more when seeing negative faces. → #16
18. Joly-Mascheroni, R. M., Senju, A., & Shepherd, A. J. (2008). Dogs catch human yawns. *Biology Letters*, 4(5), 446–448. https://doi.org/10.1098/rsbl.2008.0333
    21 of 29 dogs yawned after watching a human yawn. → #17
19. Horowitz, A. (2009). Disambiguating the "guilty look": Salient prompts to a familiar dog behaviour. *Behavioural Processes*, 81(3), 447–452. https://doi.org/10.1016/j.beproc.2009.03.014
    The "guilty look" is triggered by the owner scolding, not by the dog having misbehaved, which is why our trigger is scolding. → #18
20. Beerda, B., Schilder, M. B. H., van Hooff, J. A. R. A. M., de Vries, H. W., & Mol, J. A. (1998). Behavioural, saliva cortisol and heart rate responses to different types of stimuli in dogs. *Applied Animal Behaviour Science*, 58(3–4), 365–381. https://doi.org/10.1016/S0168-1591(97)00145-7
    Sudden stimuli (sound blasts, a falling bag) produce a very low posture. → #19
21. Humphrey, T., Proops, L., Forman, J., Spooner, R., & McComb, K. (2020). The role of cat eye narrowing movements in cat–human communication. *Scientific Reports*, 10, 16503. https://doi.org/10.1038/s41598-020-73426-0
    Cats answer an owner's slow blink with half-blinks and approach more afterwards. → #23
22. Crowell-Davis, S. L., Curtis, T. M., & Knowles, R. J. (2004). Social organization in the cat: A modern understanding. *Journal of Feline Medicine and Surgery*, 6(1), 19–28. https://doi.org/10.1016/j.jfms.2003.09.013
    Describes allorubbing as an affiliative behavior between cats. †#24 (rubbing a human) is our extrapolation.
23. Quaranta, A., Siniscalchi, M., & Vallortigara, G. (2007). Asymmetric tail-wagging responses by dogs to different emotive stimuli. *Current Biology*, 17(6), R199–R201. https://doi.org/10.1016/j.cub.2007.02.008
    Dogs wag more to the right when they see their owner, which is where the right bias comes from. → #25
24. Cafazzo, S., & Natoli, E. (2009). The social function of tail up in the domestic cat (*Felis silvestris catus*). *Behavioural Processes*, 80(1), 60–66. https://doi.org/10.1016/j.beproc.2008.09.008
    Tail up is an affiliative signal shown when approaching. → #25
