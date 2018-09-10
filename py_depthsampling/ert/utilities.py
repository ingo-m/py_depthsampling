# -*- coding: utf-8 -*-
"""Function of the depth sampling library."""

import numpy as np


def onset(aryErt, varBse, varThr):
    """
    Calculate response onset time.

    Parameters
    ----------
    aryErt : np.array
        Event related time courses. Shape: aryErt[iterations, depth, volumes],
        where iterations corresponds to bootstrap iterations (can be 1 in case
        of no bootsrapping).
    varBse : int
        Time point of first volume after stimulus onset (index in event related
        time course). In other words, the index of the first volume on which
        the stimulus was on.
    varThr : float
        z-threshold for peak finding. Peak is identified if signal is
        above/below `varThr` times mean baseline signal.

    Returns
    -------
    aryFirst : np.array
        Array with indicies of first time point above/below threshold. Shape:
        aryFirst[iterations, depth].

    """
    # Mean (over time) in pre-stimulus period, separately for each depth level.
    aryBseMne = np.mean(aryErt[:, :, :varBse], axis=2, dtype=np.float32)

    # SD (over time) in pre-stimulus period, separately for each depth level.
    aryBseSd = np.std(aryErt[:, :, :varBse], axis=2, dtype=np.float32)

    # z-score interval around mean:
    aryLimUp = np.add(aryBseMne,
                      np.multiply(aryBseSd, varThr, dtype=np.float32),
                      dtype=np.float32)
    aryLimLow = np.subtract(aryBseMne,
                            np.multiply(aryBseSd, varThr, dtype=np.float32),
                            dtype=np.float32)

    # Which timepoints are above threshold?
    aryLgc = np.logical_or(
                           np.greater(aryErt, aryLimUp[:, :, None]),
                           np.less(aryErt, aryLimLow[:, :, None])
                           )

    # Set volumes before baseline to false (avoiding false positives on first
    # and second volume due to uncomplete recovery of signal in the volumes
    # before pre-stimulus baseline):
    aryLgc[:, :, :varBse] = False

    # Find first time point over threshold:
    aryFirst = np.argmax(aryLgc, axis=2)

    return aryFirst
