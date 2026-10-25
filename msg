environment:
  workflow_list:
  - id: workflow
    name: Weekend Test
    phase_list:
    - id: p1
      name: Test Phase
      operation_list:
      - id: p1o1
        name: Test Operation
        work_hours: [10]
        workload_hours: 70
        min_worker_num: 1
        max_worker_num: 1
  fab_list:
  - id: f1
    name: Test Fab
    region: r1
    customer_company: c1
  region_list:
  - id: r1
    name: Test Region
    max_stay_on: 90
    max_annual_stay: 270
    stay_off_interval: 3
    unavailable_dates: []
  customer_company_list:
  - id: c1
    name: Test Customer
  worker_company_list:
  - id: wc1
    name: Test Company
    annual_overtime_limit: 360
    monthly_overtime_limit: 40
  transite_day_map: []
  worker_list:
  - id: w1
    name: Test Worker
    worker_company: wc1
    is_manager: true
    skill_map: {p1o1: 5}
    worker_type_by_operation: {p1o1: regular}
    fab_suitability_map:
    - kind: region
      suitability: {r1: 2}
    - kind: customer_company
      suitability: {c1: 2}
    unavailable_dates: []
    affinity: []



schedule:
  plan_range:
    start_date: 2025/09/01
    end_date: 2025/09/07
  workflow_task_list:
  - id: e1
    name: Weekend Test Task
    description:
    workflow: workflow
    fab: f1
    phase_task_list:
    - id: e1p1
      name: Test Phase
      description:
      phase: p1
      start_date: 2025/09/01
      end_date: 2025/09/07
      operation_task_list:
      - id: e1p1o1
        name: Test Operation
        description:
        operation: p1o1
        workload_hours: 70
        color_code: FFE599
  misc_task_list: []
  assignment_list: []
