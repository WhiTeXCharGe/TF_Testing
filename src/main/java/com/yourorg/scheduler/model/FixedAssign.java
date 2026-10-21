package com.yourorg.scheduler.model;

import java.util.HashMap;
import java.util.Map;

public class FixedAssign {
    public String module; public String opId; public String factory; public String wid;
    public int startDayId; public int endDayId;
    public Map<Integer,Integer> hoursByDay = new HashMap<>();
    public String phaseId; public int phaseNum;
}
