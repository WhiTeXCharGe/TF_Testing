package com.yourorg.scheduler.model;

import java.util.List;
import java.util.Map;

public class ParsedEnv {
    public Map<String,OpDef> opdef; public List<EmployeeFact> employees;
    public Map<String, EmployeeFact> byWid;
}
