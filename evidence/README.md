# Evidence: the LIBERO vs so101-nexus input diff

Backing data for a narrow claim: the photorealism changes were not what moved the success
rate, and the zero-shot failure is not explained by an input-pipeline bug.

MolmoAct2-LIBERO, a different checkpoint fine-tuned for the LIBERO benchmark, scores in the
high 90s there; MolmoAct2-SO100_101 run zero-shot in our sim did not grasp. To check whether
an input-pipeline mismatch could explain that gap, we instrumented both eval loops at the
tensor level and captured what the model actually receives in each:

| file | what it is |
|---|---|
| `libero_prompt_decoded.txt` | the exact decoded text prompt MolmoAct2-LIBERO receives in the LIBERO eval loop |
| `ours_prompt_decoded.txt` | the exact decoded text prompt MolmoAct2-SO100_101 receives in our eval loop |
| `libero_facts.json` | tensor-level facts from the LIBERO run: image shapes/dtypes/value ranges, state vector, norm stats applied, action head config |
| `ours_facts.json` | the same facts from our run |

**What this shows.** The prompts are structurally identical and the vision inputs are
equivalent (same resolution class, same normalization, both plain-OpenGL renders). So the
zero-shot failure is not an input-pipeline artifact, and the photorealism pass, which only
changed the pixels, did not move success.

**What this does not show.** It does not establish that training data is the sole remaining
cause. The two setups differ in many other ways the facts files make explicit: a different
robot, a different task, state dimension (8 vs 6), action dimension (7 vs 6), action scale,
and control mode (delta end-effector pose vs absolute joint pose). The controlled
observation is only that the attempted photorealism changes did not improve this setup.
Isolating the data as the cause would need a matched comparison this evidence does not
provide.
