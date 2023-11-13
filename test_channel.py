import numpy as np

from channels.quadriga import QuadrigaChannels

max_number_ues = 100
max_number_basestations = 1
num_available_rbs = np.array([100])

channel_model = QuadrigaChannels(
    max_number_ues, max_number_basestations, num_available_rbs
)
sched_decision = np.ones(
    (max_number_basestations, max_number_ues, num_available_rbs[0])
)
# sched_decision[0, 0, 0 : num_available_rbs[0] : 2] = 1
step_number = 1
episode_number = 1

channel_model.step(
    step_number,
    episode_number,
    np.array(
        [],
    ),
    sched_decision,
)
