from brax import base
from brax.envs.base import PipelineEnv, State
from jax import numpy as jp
from etils import epath
from brax.io import mjcf


class Links(PipelineEnv):
    """Trains 2 links to move."""

    def __init__(self, backend='generalized', **kwargs):
        path = epath.resource_path(
            'brax') / 'envs/assets/2_links.xml'
        sys = mjcf.load(path)
        self._dt = 0.2
        self._reset_count = 0
        self._step_count = 0

        super().__init__(sys=sys, backend=backend, **kwargs)

    def reset(self, rng: jp.ndarray) -> State:
        self._reset_count += 1
        # pipeline_state = base.State(
        #     q=jp.array([1]),
        #     qd=jp.array([1]),
        #     # position transform shift of 0, no rotation
        #     x=base.Transform.create(pos=jp.array([[1, 1]])),
        #     xd=base.Motion.create(vel=jp.array([[1, 1]])),
        #     contact=None
        # )
        # pipeline_state = base.State(
        #     q=jp.zeros(1),
        #     qd=jp.zeros(1),
        #     x=base.Transform.create(pos=jp.zeros(3)),
        #     xd=base.Motion.create(vel=jp.zeros(3)),
        #     contact=None
        # )
        pipeline_state = self.pipeline_init(q=jp.zeros(1), qd=jp.zeros(1))
        obs = self._get_obs(pipeline_state)
        # obs = jp.zeros(2)
        reward, done = jp.array(0.0), jp.array(0.0)
        return State(pipeline_state, obs, reward, done)

    def step(self, state: State, action: jp.ndarray) -> State:
        self._step_count += 1
        # vel = state.pipeline_state.xd.vel + (action > 0) * self._dt
        vel = state.pipeline_state.xd.vel + self._dt
        pos = state.pipeline_state.x.pos + vel * self._dt

        qp = state.pipeline_state.replace(
            x=state.pipeline_state.x.replace(pos=pos),
            xd=state.pipeline_state.xd.replace(vel=vel),
        )
        obs = jp.array([pos[0], vel[0]])
        reward = pos[0]

        return state.replace(pipeline_state=qp, obs=obs, reward=reward)

    def _get_obs(self, pipeline_state: base.State) -> jp.ndarray:
        """Observe body position and velocities."""
        qpos = pipeline_state.q
        qvel = pipeline_state.qd

        if self._exclude_current_positions_from_observation:
            qpos = pipeline_state.q[2:]

        return jp.concatenate([qpos] + [qvel])

    @property
    def reset_count(self):
        return self._reset_count

    @property
    def step_count(self):
        return self._step_count

    @property
    def observation_size(self):
        return 2

    @property
    def action_size(self):
        return 1
