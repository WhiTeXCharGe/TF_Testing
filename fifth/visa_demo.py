from typing import List, Tuple, Dict, Iterable

def _indices_of_country(days: Iterable[str], country: str) -> List[int]:
    return [i for i, c in enumerate(days) if c == country]

def _segments_from_days(day_ids: List[int], break_days: int) -> List[Tuple[int, int]]:
    """
    Build stay segments from indices where the employee is IN the country.
    We split only if the count of consecutive OUT-of-country days >= break_days.
    """
    if not day_ids:
        return []
    s = sorted(day_ids)
    segs: List[Tuple[int,int]] = []
    cur_start = s[0]
    prev = s[0]
    for d in s[1:]:
        gap_out_days = (d - prev - 1)  # number of NOT-in-country days between prev and d
        if gap_out_days >= break_days:
            segs.append((cur_start, prev))
            cur_start = d
        prev = d
    segs.append((cur_start, prev))
    return segs

def _fmt_span(seg: Tuple[int, int]) -> int:
    return (seg[1] - seg[0]) + 1

def print_segments(title: str, segs: List[Tuple[int,int]]):
    print(f"{title}:")
    for (a,b) in segs:
        print(f"  stay {a}->{b}, span={_fmt_span((a,b))} days")

def check_visa_presence(days: List[str], visa_limits: Dict[str,int], presence_gap_break_days: int):
    print("\n=== VISA PRESENCE CHECK ===")
    for country, limit in visa_limits.items():
        idxs = _indices_of_country(days, country)
        segs = _segments_from_days(idxs, max(1, presence_gap_break_days))
        print_segments(f"Country {country} (limit {limit})", segs)
        for (a,b) in segs:
            span = _fmt_span((a,b))
            if span > limit:
                print(f"  -> OVERSTAY: {span} > {limit}")

def check_annual_presence(days: List[str], annual_limits: Dict[str,int], annual_break_days: int):
    print("\n=== ANNUAL PRESENCE CHECK ===")
    for country, limit in annual_limits.items():
        idxs = _indices_of_country(days, country)
        segs = _segments_from_days(idxs, max(1, annual_break_days))
        print_segments(f"Country {country} (annual limit {limit})", segs)
        total = sum(_fmt_span(s) for s in segs)
        print(f"  Total span={total}, Limit={limit}")
        if total > limit:
            print(f"  -> ANNUAL OVER: {total} > {limit}")

if __name__ == "__main__":
    # Try input sequences
    seq = ['A','A','A','F','F','A','A','A']   # F = free/Japan
    print("Days:", seq)

    check_visa_presence(seq, {'A': 6, 'B': 7}, presence_gap_break_days=3)
    check_annual_presence(seq, {'A': 20, 'B': 20}, annual_break_days=3)
