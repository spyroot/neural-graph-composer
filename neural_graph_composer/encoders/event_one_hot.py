import abc


class OneHotEncoding:
    """An interface for specifying a one-hot encoding of individual events."""
    __metaclass__ = abc.ABCMeta

    @property
    @abc.abstractmethod
    def num_classes(self):
        """The number of distinct event encodings.
    Returns:
      An int, the range of ints that can be returned by self.encode_event.
    """
        pass

    @property
    @abc.abstractmethod
    def default_event(self):
        """An event value to use as a default.
    Returns:
      The default event value.
    """
        pass

    @abc.abstractmethod
    def encode_event(self, event):
        """Convert from an event value to an encoding integer.
    Args:
      event: An event value to encode.
    Returns:
      An integer representing the encoded event, in range [0, self.num_classes).
    """
        pass

    @abc.abstractmethod
    def decode_event(self, index):
        """Convert from an encoding integer to an event value.
    Args:
      index: The encoding, an integer in the range [0, self.num_classes).
    Returns:
      The decoded event value.
    """
        pass

    @staticmethod
    def event_to_num_steps(self, unused_event):
        """Returns the number of time steps corresponding to an event value.
    This is used for normalization when computing metrics. Subclasses with
    variable step size should override this method.
    Args:
      unused_event: An event value for which to return the number of steps.
    Returns:
      The number of steps corresponding to the given event value, defaulting to
      one.
    """
        return 1
