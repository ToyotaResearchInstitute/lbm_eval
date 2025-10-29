import numpy as np


class Trajectory:
    """Trajectory for an generic set of values."""

    # TODO(eric.cousineau): Use the name `rollout` instead?
    def __init__(self, *, ts=[], values=[]):
        assert len(ts) == len(values)
        self._ts = list(ts)
        self._values = list(values)

    @property
    def ts(self):
        return list(self._ts)

    @property
    def values(self):
        return list(self._values)

    def add(self, t, value):
        if len(self) > 0:
            tf = self._ts[-1]
            assert t > tf, f"Must be strictly increasing: {t} <= {tf}"
        self._ts.append(t)
        self._values.append(value)

    def __iadd__(self, other):
        cls = type(self)
        assert isinstance(other, cls), (other, cls)
        if len(self) > 1:
            # TODO(eric.cousineau): Injecting this dt feels sketchy... Use
            # eps() instead?
            dt = np.min(np.diff(self.ts))
            assert dt > 0
            t_offset = self.ts[-1] + dt
        else:
            t_offset = 0.0
        for t, value in other:
            self.add(t + t_offset, value)
        return self

    def __getitem__(self, index):
        out = self.ts[index], self.values[index]
        if isinstance(index, slice):
            return zip(*out)
        else:
            return out

    def __delitem__(self, index):
        del self._ts[index]
        del self._values[index]

    def __len__(self):
        return len(self._ts)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self):
        count = len(self)
        duration = None
        if count > 0:
            duration = self._ts[-1] - self._ts[0]
        cls = type(self)
        return f"<{cls.__name__} len={count} duration={duration}s>"
