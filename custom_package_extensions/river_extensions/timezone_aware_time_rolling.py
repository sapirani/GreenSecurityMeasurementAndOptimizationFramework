from datetime import timedelta, datetime, timezone

from dateutil.tz import UTC
from river import utils
from river.utils.rolling import Rollable


class TimezoneAwareTimeRolling(utils.TimeRolling):
    def __init__(self, obj: Rollable, period: timedelta):
        super().__init__(obj, period)
        self._latest = datetime(1, 1, 1, tzinfo=UTC)

    def flush_expired(self):
        """
        This function is mostly copied from TimeRolling update function.
        The original implementation is updated only when a new element is inserted.
        This function enables the removal of old events, even though no new elements are inserted.
        """
        # Update latest time marker (this is the difference from the original implementation where they
        # perform computations based on the latest sample arrived instead of the current time).
        self._latest = datetime.now(timezone.utc)

        # Find all expired points (older than now - period)
        i = 0
        for ti, (argsi, kwargsi) in zip(self._timestamps, self._datum):
            if ti > self._latest - self.period:
                break
            self.obj.revert(*argsi, **kwargsi)
            i += 1

        # Remove expired events
        if i > 0:
            self._timestamps = self._timestamps[i:]
            self._datum = self._datum[i:]
