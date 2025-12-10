# -*- coding: utf-8 -*-
""" grounded_sam_2/download_pretrained_grounding_dino.py """

import os
import subprocess


checkpoints_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "gdino_checkpoints")
subprocess.run("bash download_ckpts.sh".split(), cwd=checkpoints_path, check=True, shell=False)
