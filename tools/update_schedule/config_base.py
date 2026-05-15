WORKER_NUM = 400
EQ_PER_DAYS = 2.5
EQ_PER_DAYS_SIGMA = 2.5
EQ_NUM = 300

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

worker_company_num = 2

is_skip_weekend = True

# --- operation worker count ---
# min/max workers required per operation in the workflow definition
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
# All region pairs (r_i → r_j, i ≠ j) are auto-generated.
# Day count is randomly selected from options with the given weights.
transit_day_options = [1]
transit_day_weights = [1.0]

# --- worker company affinity tags ---
# One tag (wct1, wct2, ...) is created per worker company; all workers in that company receive it.
worker_company_tag_weight = 2

# --- worker_type_by_operation ---
# Probability that a skilled operation is assigned "regular" (remainder is "spot")
worker_type_regular_chance = 0.8

# --- affinity tags ---
affinity_group_num = 25
# Possible weight values for each affinity tag and their selection probabilities
affinity_weight_options = (-2, -1, 1, 2)
affinity_weight_chances = (0.15, 0.25, 0.40, 0.20)
# Number of workers assigned to each affinity tag (inclusive min/max)
affinity_group_size = (2, 5)

# --- worker unavailable dates ---
unavailable_date_range_start = "2025/09/01"
unavailable_date_range_end = "2026/03/31"
# Index = number of unavailable dates assigned; lower index = more common
unavailable_max_dates = 6
unavailable_count_weights = (0.50, 0.20, 0.12, 0.08, 0.05, 0.03, 0.02)

# --- workload format ---
# "hours": write workload_hours = workload_days * workload_units
# "days":  write workload_days as-is
workload_format = "hours"
workload_units = 8   # hours per working day

# --- recommended workers per operation task ---
recommends_worker_enabled = False
recommends_worker_options = [(1, 1), (2, 2), (2, 3), (3, 3)]
recommends_worker_weights = (0.10, 0.30, 0.40, 0.20)
