# Google Barkour v0 Joystick Policy

## Overview

This folder contains a training script for a flat-terrain joystick policy for the [Barkour v0 Quadruped](https://ai.googleblog.com/2023/05/barkour-benchmarking-animal-level.html) which demonstrates sim2real transfer.

`barkour_joystick.py` contains the environment definition, while the [colab](https://colab.research.google.com/github/google/brax/blob/main/brax/experimental/barkour_v0/barkour_v0_joystick.ipynb) shows how to train the policy.

<p float="left">
  <img src="assets/joystick.gif" width="400">
</p>

## Running the environment

We encourage the usage of the [colab](https://colab.research.google.com/github/google/brax/blob/main/brax/experimental/barkour_v0/barkour_v0_joystick.ipynb) for viewing and training policies. However, the environment can be loaded as follows:

```python
import jax
from jax import numpy as jp

from brax import envs
from brax.experimental.barkour_v0 import barkour_joystick

barkour_env = envs.create('barkour_v0_joystick', backend='generalized')
```

And to step through the environment:

```python
jit_env_reset = jax.jit(barkour_env.reset)
jit_env_step = jax.jit(barkour_env.step)

state = jit_env_reset(jax.random.PRNGKey(0))

rollout = []
for i in range(500):
  act = jp.sin(i / 500) * jp.ones(barkour_env.sys.act_size())
  state = jit_env_step(state, act)
  rollout.append(state)
```

## MJCF Instructions

The MuJoCo config in `assets/barkour_v0_brax.xml` was copied from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/google_barkour_v0). The following edits were made to the MJCF specifically for brax:

* `meshdir` was changed from `assets` to `.`.
* `frictionloss` was removed. `damping` was changed to 0.5239.
* A custom `init_qpos` was added.
* A sphere geom `lowerLegFoot` was added to all feet. All other contacts were turned off.
* The compiler option was changed to `<option timestep="0.002" iterations="40"/>`.
* Non-visual geoms were removed from the torso, to speed up rendering.

## Publications

If you use this work in an academic context, please cite the following publication:

    @misc{caluwaerts2023barkour,
          title={Barkour: Benchmarking Animal-level Agility with Quadruped Robots},
          author={Ken Caluwaerts and Atil Iscen and J. Chase Kew and Wenhao Yu and Tingnan Zhang and Daniel Freeman and Kuang-Huei Lee and Lisa Lee and Stefano Saliceti and Vincent Zhuang and Nathan Batchelor and Steven Bohez and Federico Casarini and Jose Enrique Chen and Omar Cortes and Erwin Coumans and Adil Dostmohamed and Gabriel Dulac-Arnold and Alejandro Escontrela and Erik Frey and Roland Hafner and Deepali Jain and Bauyrjan Jyenis and Yuheng Kuang and Edward Lee and Linda Luu and Ofir Nachum and Ken Oslund and Jason Powell and Diego Reyes and Francesco Romano and Feresteh Sadeghi and Ron Sloat and Baruch Tabanpour and Daniel Zheng and Michael Neunert and Raia Hadsell and Nicolas Heess and Francesco Nori and Jeff Seto and Carolina Parada and Vikas Sindhwani and Vincent Vanhoucke and Jie Tan},
          year={2023},
          eprint={2305.14654},
          archivePrefix={arXiv},
          primaryClass={cs.RO}
    }