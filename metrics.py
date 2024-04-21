# -*- mode: python -*-

import numpy as np
from scipy import stats


class Metrics:

    @staticmethod
    def R2(measured, estimated) -> float:
        r_value = stats.linregress(measured, estimated)
        return r_value[2] ** 2

    @staticmethod
    def SLOPE(measured, estimated) -> float:
        return stats.linregress(measured, estimated)[0]

    @staticmethod
    def MAPE(measured, estimated) -> float:
        mape = (abs(estimated - measured) / measured) * 100
        return np.mean(mape)

    @staticmethod
    def RMSE(measured, estimated) -> float:
        return np.sqrt(np.mean((estimated - measured) ** 2))

    @staticmethod
    def MAELOG(measured, estimated) -> float:
        # Seegers (2018)
        logMeasured = np.log10(measured)
        logEstimated = np.log10(estimated)
        mean_ = np.mean(abs(logEstimated - logMeasured))
        return 10 ** mean_

    @staticmethod
    def BIASLOG(measured, estimated) -> float:
        # Seegers (2018)
        logMeasured = np.log10(measured)
        logEstimated = np.log10(estimated)
        mean_ = np.mean(logEstimated - logMeasured)
        return 10 ** mean_

