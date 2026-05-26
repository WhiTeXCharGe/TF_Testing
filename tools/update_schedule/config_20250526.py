WORKER_NUM = 1500
EQ_PER_DAYS = 3.7
EQ_PER_DAYS_SIGMA = 1.5
EQ_NUM = 365

skill_level_list = (1, 2, 3, 4, 5)
skill_level_weights = (0.05, 0.1, 0.5, 0.25, 0.1)

region_suitability_list = (0, 1, 2)
region_suitability_weights = (0.03, 0.6, 0.37)

customer_suitability_list = (0, 1, 2)
customer_suitability_weights = (0.03, 0.6, 0.37)

# (phase_total_days, [workload_days_per_operation]) — days format; converted to hours when writing
normal_worklength = [
    (15, [30, 20, 20]),
    (12, [30, 25]),
    (6, [15]),
    (7, [12, 8]),
]

vip_worklength = [
    (12, [30, 20, 20]),
    (10, [30, 25]),
    (6, [15]),
    (8, [12, 8]),
]

manager_rate = 0.7

is_skip_weekend = True

# --- worker company definitions ---
# 8 companies × ~187 workers each; meaningful for company-level constraints
worker_company_definitions = [
    {"id": "wc1", "name": "AAA"},
    {"id": "wc2", "name": "BBB"},
    {"id": "wc3", "name": "CCC"},
    {"id": "wc4", "name": "DDD"},
    {"id": "wc5", "name": "EEE"},
    {"id": "wc6", "name": "FFF"},
    {"id": "wc7", "name": "GGG"},
    {"id": "wc8", "name": "HHH"},
]

# --- operation worker count ---
operation_worker_min = 2
operation_worker_max = 3

# --- region definitions ---
# id is auto-assigned r1, r2, ... based on list order
region_definitions = [
    {
        "name": "America",
        "max_stay_on": 80,
        "max_annual_stay": 240,
        "stay_off_interval": 3,
        "weekly_weekdays": ["sat", "sun"],
        "single_days": [],
    },
    {
        "name": "Germany",
        "max_stay_on": 80,
        "max_annual_stay": 240,
        "stay_off_interval": 3,
        "weekly_weekdays": ["sat", "sun"],
        "single_days": [],
    },
    {
        "name": "Taiwan",
        "max_stay_on": 80,
        "max_annual_stay": 240,
        "stay_off_interval": 3,
        "weekly_weekdays": ["sat", "sun"],
        "single_days": [],
    },
]

# --- transit days between regions ---
transit_day_options = [1]
transit_day_weights = [1.0]

# --- worker company affinity tags ---
worker_company_tag_weight = 2

# --- worker_type_by_operation ---
worker_type_regular_chance = 0.8

# --- affinity tags ---
# 86 groups × avg 3.5 workers = ~300 memberships → ~20% of 1500 workers hold at least one tag
affinity_group_num = 86
affinity_weight_options = (-2, -1, 1, 2)
affinity_weight_chances = (0.15, 0.25, 0.40, 0.20)
affinity_group_size = (2, 5)

# --- worker unavailable dates ---
unavailable_date_range_start = "2025/09/01"
unavailable_date_range_end = "2026/03/31"
unavailable_max_dates = 6
unavailable_count_weights = (0.50, 0.20, 0.12, 0.08, 0.05, 0.03, 0.02)

# --- workload format ---
workload_format = "hours"
workload_units = 8

# --- recommended workers per operation task ---
recommends_worker_enabled = False
recommends_worker_options = [(1, 1), (2, 2), (2, 3), (3, 3)]
recommends_worker_weights = (0.10, 0.30, 0.40, 0.20)
