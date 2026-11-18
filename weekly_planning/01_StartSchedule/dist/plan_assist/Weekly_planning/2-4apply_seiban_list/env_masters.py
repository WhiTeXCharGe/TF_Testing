"""
Customers, fabs and regions of the EnvConfig (tool 2-4, step B).

resolve_site() finds, or adds, the customer / fab / region of one seiban row and returns the id of its fab.  Nothing is ever
deleted or renamed.  The functions work on the plain `environment` dict (the part under `environment:`) and change it in place:
the caller passes a copy.  Every change and every doubt is a line in `report` ("INFO: ..." for a change, "WARN: ..." for a doubt).

The caller puts weekly_planning/ on the import path (the tool script and the tests do).
"""
from __future__ import annotations

import re

from common.text import match_key

OTHER_REGION, OTHER_CUSTOMER, OTHER_FAB = "r_other", "c_other", "f_other"

# the values the decoder gave to every region it created
REGION_DEFAULTS = {"max_stay_on": 10000, "max_annual_stay": 10000, "stay_off_interval": 3}
TRANSIT_DAYS = 1


def next_id(items, prefix):
    """The next free '<prefix><n>' (highest existing number + 1; 'r_other' and the like are not numbers)."""
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    numbers = [int(m.group(1)) for it in items for m in [pattern.match(str(it.get("id", "")))] if m]
    return f"{prefix}{max(numbers, default=0) + 1}"


def _find_by_name(items, name):
    key = match_key(name)
    return next((it for it in items if match_key(it.get("name")) == key), None)


def _by_id(items, item_id):
    return next((it for it in items if it.get("id") == item_id), None)


# ----------------------------------------------------------------------------- regions

def _add_region(env, region_id, name, report):
    regions = env.setdefault("region_list", [])
    others = [r["id"] for r in regions]
    region = {"id": region_id, "name": name, **REGION_DEFAULTS, "unavailable_dates": []}
    regions.append(region)
    transit = env.setdefault("transite_day_map", [])
    have = {(t.get("from"), t.get("to")) for t in transit}
    for other in others:                                  # both directions between the new region and every existing one
        for a, b in ((region_id, other), (other, region_id)):
            if (a, b) not in have:
                transit.append({"from": a, "to": b, "days": TRANSIT_DAYS})
    report.append(f"INFO: region added: {region_id} {name!r} (transit {TRANSIT_DAYS} day to and from {len(others)} regions)")
    return region


def ensure_region(env, name, report):
    """The region with this name (by name key), added when it does not exist."""
    found = _find_by_name(env.get("region_list") or [], name)
    return found or _add_region(env, next_id(env.get("region_list") or [], "r"), name.strip(), report)


def _other_region(env, report):
    return _by_id(env.get("region_list") or [], OTHER_REGION) or _add_region(env, OTHER_REGION, "Other", report)


# ----------------------------------------------------------------------------- customers and fabs

def ensure_customer(env, name, report):
    found = _find_by_name(env.get("customer_company_list") or [], name)
    if found:
        return found
    customers = env.setdefault("customer_company_list", [])
    customer = {"id": next_id(customers, "c"), "name": name.strip(), "unavailable_dates": []}
    customers.append(customer)
    report.append(f"INFO: customer added: {customer['id']} {customer['name']!r}")
    return customer


def _other_customer(env, report):
    customers = env.setdefault("customer_company_list", [])
    found = _by_id(customers, OTHER_CUSTOMER)
    if found:
        return found
    customers.append({"id": OTHER_CUSTOMER, "name": "OTHER", "unavailable_dates": []})
    report.append(f"INFO: customer added: {OTHER_CUSTOMER} 'OTHER'")
    return customers[-1]


def _other_fab(env, report):
    fabs = env.setdefault("fab_list", [])
    found = _by_id(fabs, OTHER_FAB)
    if found:
        return found
    fabs.append({"id": OTHER_FAB, "name": "Other", "region": _other_region(env, report)["id"],
                 "customer_company": _other_customer(env, report)["id"], "unavailable_dates": []})
    report.append(f"INFO: fab added: {OTHER_FAB} 'Other'")
    return fabs[-1]


def resolve_site(env, customer, fab, region, site_region, report):
    """Fab id for a seiban row.  customer / fab / region are the texts of the row; site_region is the Region the site table
    has for this customer and fab (or None).

      region      the row wins; the site table fills it when the row has none; a difference is a WARN
      customer    matched by name key, added when new; empty = c_other (both empty = f_other)
      fab         matched by customer + fab name, so the same fab name under another customer is another fab;
                  an empty fab name uses the customer name
      existing fab with another region   its region is updated (INFO)
      new fab     region unknown = r_other (WARN)"""
    customer, fab, region = (customer or "").strip(), (fab or "").strip(), (region or "").strip()
    label = " / ".join(x for x in (customer, fab) if x) or "(no customer, no fab)"
    if region and site_region and match_key(region) != match_key(site_region):
        report.append(f"WARN: {label}: Region {region!r} of the row differs from {site_region!r} of the site table; the row is used")
    region_text = region or (site_region or "").strip()

    if not customer and not fab:
        return _other_fab(env, report)["id"]
    region_id = ensure_region(env, region_text, report)["id"] if region_text else None
    owner = ensure_customer(env, customer, report) if customer else _other_customer(env, report)
    fab_name = fab or customer
    fabs = env.setdefault("fab_list", [])
    key = match_key(fab_name)
    found = next((f for f in fabs if f.get("customer_company") == owner["id"] and match_key(f.get("name")) == key), None)
    if found:
        if region_id and found.get("region") != region_id:
            names = {r["id"]: r.get("name") for r in env.get("region_list") or []}
            report.append(f"INFO: fab {found['id']} {found['name']!r} ({owner['name']}): region changed "
                          f"{names.get(found.get('region'), found.get('region'))!r} -> {names.get(region_id, region_id)!r}")
            found["region"] = region_id
        return found["id"]

    if region_id is None:
        region_id = _other_region(env, report)["id"]
        report.append(f"WARN: {label}: no Region in the row or in the site table; the new fab gets {OTHER_REGION}")
    new = {"id": next_id(fabs, "f"), "name": fab_name, "region": region_id, "customer_company": owner["id"],
           "unavailable_dates": []}
    fabs.append(new)
    report.append(f"INFO: fab added: {new['id']} {fab_name!r} (customer {owner['id']}, region {region_id})")
    return new["id"]
