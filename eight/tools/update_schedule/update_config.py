# update_config.py
# Config for update_schedule.py

# --- Cutoff for Fixed/Flexible + plan_range.start_date ---
# All assignments with start_date < this are Fixed, others Flexible.
CUTOFF_DATE_STR = "2025/10/01"   # format: YYYY/MM/DD

# --- File paths ---
# You can change these if you want to use different files.
ENV_PATH = "EnvConfig.yaml"
SCHEDULE_IN_PATH = "Schedule.yaml"
SCHEDULE_OUT_PATH = "Schedule.yaml"   # same as input = overwrite

# --- Target counts after update ---
# Total workers you want in EnvConfig.environment.worker_list
WORKER_NUM = 120

# Total equipment modules you want in Schedule.schedule.workflow_task_list
# e1..e{EQ_NUM} will exist after update (existing are kept, new ones appended).
EQ_NUM = 50

# --- Plan range extra days ---
# plan_range.end_date = max(last_module_end) + PLAN_RANGE_EXTRA_DAYS
PLAN_RANGE_EXTRA_DAYS = 3

# --- Random seeds (change to reshuffle) ---
# Worker extension (skills, manager flag, worker_company)
ENV_SEED = 100
# Module extension (start dates, fab choice, normal/vip pattern)
MODULE_SEED = 200

EQ_EVAL_DAYS = 1