import pandas as pd
from river import stats, utils
from river.feature_extraction import Agg

from hadoop_optimizer.drl_model.consts.general import ALL_HOSTS
from hadoop_optimizer.river_extensions.timezone_aware_time_rolling import TimezoneAwareTimeRolling


class CustomAgg(Agg):
    def __init__(self, on: str, by: str | list[str] | None,
                 how: stats.base.Univariate | utils.Rolling | utils.TimeRolling):
        """
        :param on: The feature on which to compute the aggregate statistic
        :param by: The feature by which to group the data. All the data is included in the aggregate
        if this is `None`.
        :param how: The statistic to compute. This constructor ensures that this statistic is computed via
         TimezoneAwareTimeRolling.
        """
        super().__init__(on, by, how)
        if not isinstance(how, TimezoneAwareTimeRolling):
            raise ValueError("CustomAgg supports only TimezoneAwareTimeRolling as the 'how' argument")

        # Add the time window to the feature name
        self._feature_name += f"_window_{int(how.period.total_seconds())}_seconds"

    @property
    def state(self) -> pd.Series:
        """
        This function ensures that aggregations time window are computed compared to the current time.
        Also, it ensures no crashes in the case where 'by' is not provided.
        """
        for time_rolling in self._groups.values():
            time_rolling.flush_expired()

        if not self.by:
            return pd.Series(
                (stat.get() for stat in self._groups.values()),
                index=pd.Index([ALL_HOSTS]),
                name=self._feature_name,
            )
        return super().state
