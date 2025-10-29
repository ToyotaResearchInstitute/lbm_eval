from datetime import datetime, timezone


# TODO(sfeng/naveen): add unit test for these.
def isoformat_from_utc_float_timestamp(timestamp_float: float, tz=None):
    """
    converts a utc float timestamp (unix time) and converts to a datetime in
    timezone `tz`. If `tz` is not specified, the system time zone will be used.
    """

    # get utc datetime
    dt = datetime.fromtimestamp(timestamp_float, timezone.utc)
    return datetime_to_isoformat(dt, tz)


def isoformat_to_utc_float_timestamp(iso_time: str):
    """
    converts a iso formatted string into a utc float timestamp. `iso_time`
    must have time zone information in it.
    """
    dt = datetime.fromisoformat(iso_time)
    assert dt.tzinfo is not None
    return dt.replace(tzinfo=timezone.utc).timestamp()


def datetime_to_isoformat(dt, tz=None):
    """
    converts a datetime into iso format string in microseconds.
    `dt` must have time zone information.
    If `tz` is not specified, the system time zone will be used.
    """
    assert dt.tzinfo is not None
    return dt.astimezone(tz).isoformat(timespec="microseconds")


def get_datetime_now(tz=None):
    """
    returns a datetime obj for current time in timezone `tz`. defaults
    to current time zone if `tz` is none.
    """
    return datetime.now().astimezone(tz)
