# project : ONE-SHOT refinement

# One-shot refinement effectively makes an input structure more plausible by utilizing a fine-tuned RosettaFold model. This fine-tuned RosettaFold is based on a model trained for self-conditioning in RF Diffusion. It operates efficiently when the input structure is already reasonably well-formed, further refining it into a more realistic conformation.

# command
 python scripts/refine.py --config-name oneshot inference.input_pdb=input.pdb +inference.output_pdb=output.pdb +inference.shot_num=2

# commits