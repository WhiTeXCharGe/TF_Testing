package com.yourorg.scheduler.model;

import java.util.HashMap;
import java.util.Map;

public class InitialAssign {
    public String module;
    public String opId;
    public Map<Integer,Integer> hoursByDay = new HashMap<>();
}
