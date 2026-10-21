package com.yourorg.scheduler.model;

import java.util.List;

public class TaskWindow {
    public String module;
    public String workflowId;
    public String factory;
    public String phaseId;
    public int phaseNum;
    public String opId;
    public int startDayId;
    public int endDayId;
    public List<Integer> allowed;
    public int minHeads;
    public int maxHeads;
    public int workloadDays;
    public int workloadHours = -1; // -1 = compute from workloadDays; >=0 = direct hours
    public int recommendMinHeads;
    public int recommendMaxHeads;
}
