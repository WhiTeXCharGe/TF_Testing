package com.yourorg.scheduler.calendar;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

// ---- Calendar / blackout support ----
public class Calendars {
    public Set<Integer> weekends = new HashSet<>();
    public Map<String, Set<Integer>> fabOff = new HashMap<>();
    public Map<String, Set<Integer>> regionOff = new HashMap<>();
    public Map<String, Set<Integer>> customerOff = new HashMap<>();
    public Map<String, Set<Integer>> workerCompanyOff = new HashMap<>();
    public Map<String, String> fabToRegion = new HashMap<>();
    public Map<String, String> fabToCustomer = new HashMap<>();
    public Map<String, Set<Integer>> workerOffByWid = new HashMap<>();
    public Map<String, Map<String, Integer>> transitDays = new HashMap<>(); // fromRegion -> (toRegion -> days)
    public Map<String, Integer> regionStayMaxOn = new HashMap<>();
    public Map<String, Integer> regionStayOffInterval = new HashMap<>();

    //  annual-stay per region
    public Map<String, Integer> regionAnnualMaxStay = new HashMap<>();

    //  overtime limits per worker company (used per-worker)
    public Map<String, Integer> workerCompanyAnnualOtLimit = new HashMap<>();
    public Map<String, Integer> workerCompanyMonthlyOtLimit = new HashMap<>();

    public int transitDays(String from, String to) {
        if (from == null || to == null || from.equals(to)) return 0;
        return transitDays.getOrDefault(from, Map.of()).getOrDefault(to, 0);
    }
    public String regionOfFab(String fabId) { return fabId == null ? null : fabToRegion.get(fabId); }
    public String customerOfFab(String fabId) { return fabId == null ? null : fabToCustomer.get(fabId); }

    public int maxStayOn(String regionId) { return regionStayMaxOn.getOrDefault(regionId, Integer.MAX_VALUE); }
    public int stayOffInterval(String regionId) { return Math.max(1, regionStayOffInterval.getOrDefault(regionId, 1)); }

    //  helper getters
    public int annualMaxStay(String regionId) {
        return regionAnnualMaxStay.getOrDefault(regionId, Integer.MAX_VALUE);
    }
    public int annualOtLimit(String companyId) {
        return workerCompanyAnnualOtLimit.getOrDefault(companyId, Integer.MAX_VALUE);
    }
    public int monthlyOtLimit(String companyId) {
        return workerCompanyMonthlyOtLimit.getOrDefault(companyId, Integer.MAX_VALUE);
    }

    public boolean isWorkingDay(int dayId, String fabId) {
        if (dayId < 0) return false;
        if (fabId == null) return true;

        if (fabOff.getOrDefault(fabId, Set.of()).contains(dayId)) return false;
        String rid = fabToRegion.get(fabId);
        if (rid != null && regionOff.getOrDefault(rid, Set.of()).contains(dayId)) return false;
        String cid = fabToCustomer.get(fabId);
        if (cid != null && customerOff.getOrDefault(cid, Set.of()).contains(dayId)) return false;
        return true;
    }

    public boolean isWorkerUnavailable(String wid, int dayId) {
        if (wid == null) return false;
        return workerOffByWid.getOrDefault(wid, Set.of()).contains(dayId);
    }
}
