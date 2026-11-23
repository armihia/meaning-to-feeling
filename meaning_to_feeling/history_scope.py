from datetime import datetime, timezone


def _completion_time(value):
    if not isinstance(value, str):
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        instant = datetime.fromisoformat(value)
        if instant.tzinfo is not None:
            return instant.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None
    return None


def parse_cutoff(as_of):
    cutoff = _completion_time(as_of)
    if cutoff is None:
        raise ValueError("开读时间需要带时区，例如 2026-11-19T09:00:00+08:00")
    return cutoff


def select_completed_history(value, as_of):
    if as_of is None:
        return value, None
    cutoff = parse_cutoff(as_of)
    history = value.get("history") if isinstance(value, dict) else None
    if not isinstance(history, list) or any(not isinstance(row, dict) for row in history):
        raise ValueError("G输入需要由历史作答对象组成的history数组")
    eligible = []
    after_cutoff = unverified = 0
    for row in history:
        finished = _completion_time(row.get("finished_at"))
        if finished is None:
            unverified += 1
        elif finished > cutoff:
            after_cutoff += 1
        else:
            eligible.append(row)
    return dict(value, history=eligible), dict(
        as_of=cutoff.isoformat(),
        provided_trials=len(history),
        eligible_trials=len(eligible),
        after_cutoff_trials=after_cutoff,
        unverified_completion_trials=unverified,
    )
