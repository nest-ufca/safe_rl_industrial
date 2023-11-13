import matplotlib.pyplot as plt
import numpy as np

from traffics.industrial import IndustrialTraffic

max_number_ues = 100
seed = 10
rng = np.random.default_rng(seed)
traffic_gen = IndustrialTraffic(max_number_ues, rng)

steps = 1000

slice_req = {
    "embb": {
        "number_ues": 20,
        "ue_throughput": 20,  # Mbps
        "latency": 20,  # ms
    },
    "urllc": {
        "number_ues": 20,
        "ue_throughput": 5,  # Mbps
        "latency": 1,
    },
    "mmtc": {
        "number_ues": 60,
        "ue_throughput": 0.1,  # Mbps
        "latency": 1,  # ms
        "active_probability": 0.5,
    },
}

traffic_hist = np.zeros((steps, max_number_ues))
for step in range(steps):
    traffic_hist[step, :] = traffic_gen.step(np.ones(10), slice_req, step, 0)

ues_slice = [
    slice_req["embb"]["number_ues"],
    slice_req["urllc"]["number_ues"],
    slice_req["mmtc"]["number_ues"],
]
plt.figure(figsize=(10, 5))
plt.plot(np.mean(traffic_hist[:, 0 : ues_slice[0]], axis=1), label="eMBB")
plt.plot(
    np.mean(
        traffic_hist[:, ues_slice[0] : ues_slice[0] + ues_slice[1]], axis=1
    ),
    label="URLLC",
)
plt.plot(
    np.mean(
        traffic_hist[
            :,
            ues_slice[0]
            + ues_slice[1] : ues_slice[0]
            + ues_slice[1]
            + ues_slice[2],
        ],
        axis=1,
    ),
    label="mMTC",
)
plt.grid()
plt.legend()
plt.xlabel("Simulation step (n)")
plt.ylabel("Throughput (Mbps)")
plt.show()
