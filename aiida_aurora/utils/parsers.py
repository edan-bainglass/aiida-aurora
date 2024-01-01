import numpy as np
from scipy.integrate import cumtrapz

from aiida.orm import ArrayData


def get_data_from_raw(jsdata) -> dict:
    "Extract raw data from json file."

    if not isinstance(jsdata, dict):
        raise TypeError("jsdata should be a dictionary")

    if len(jsdata["steps"]) > 1:
        raise NotImplementedError("multi-step analysis not implemented.")

    raw_data = jsdata["steps"][0]["data"]

    # extract raw data
    t = np.array([ts["uts"] for ts in raw_data]) - raw_data[0]["uts"]
    Ewe = np.array([ts["raw"]["Ewe"]["n"] for ts in raw_data])
    I = np.array([ts["raw"]["I"]["n"] for ts in raw_data])

    return post_process_data(t, Ewe, I)


def get_data_from_results(array_node) -> dict:
    "Extract data from parsed ArrayData node."

    if not isinstance(array_node, ArrayData):
        raise TypeError("array_node should be an ArrayData")

    # collect data
    t = array_node.get_array("step0_uts")
    t -= t[0]
    Ewe = array_node.get_array("step0_Ewe_n")
    I = array_node.get_array("step0_I_n")

    return post_process_data(t, Ewe, I)


def post_process_data(t: np.ndarray, Ewe: np.ndarray, I: np.ndarray) -> dict:
    """docstring"""

    mask = I != 0  # filter out zero current
    t, Ewe, I = t[mask], Ewe[mask], I[mask]  # [s], [V], [A]

    Q = cumtrapz(I, t, axis=0, initial=0)  # [As]

    # mark half-cycles (including first and last values)
    idx = np.where(np.diff(np.sign(I), prepend=0) != 0)[0]
    idx = np.append(idx, len(I))

    # integrate and store charge/discharge capacities/energies
    cycle_idx, Qc, Qd, Ec, Ed = [], [], [], [], []

    for ii in range(len(idx) - 1):

        i0, ie = idx[ii], idx[ii + 1]

        if ie - i0 < 10:
            continue

        e = np.trapz(Ewe[i0:ie], Q[i0:ie])  # [Ws]

        if (q := np.trapz(I[i0:ie], t[i0:ie])) > 0:
            cycle_idx.append(i0)
            Qc.append(q)
            Ec.append(e)
        else:
            Qd.append(abs(q))
            Ed.append(abs(e))

    return {
        "time": t,
        "Ewe": Ewe,
        "I": I,
        "Q": Q / 3.6,
        "cycle-number": np.arange(len(Qd)),
        "cycle-index": np.array(cycle_idx),
        "Qc": np.array(Qc) / 3.6,  # [mAh]
        "Qd": np.array(Qd) / 3.6,  # [mAh]
        "Ec": np.array(Ec) / 3600,  # [Wh]
        "Ed": np.array(Ed) / 3600,  # [Wh]
    }
