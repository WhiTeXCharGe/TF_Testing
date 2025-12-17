WORKER_NUM = 400
EQ_PER_DAYS = 2.5
EQ_PER_DAYS_SIGMA = 2.5
EQ_NUM = 30

skill_level_list = (1,2,3,4,5)
skill_level_weights = (0.05, 0.1, 0.5, 0.25, 0.1)

region_suitability_list = (0, 1, 2)
region_suitability_weights = (0.1, 0.6, 0.3)

customer_suitability_list = (0, 1, 2)
customer_suitability_weights = (0.1, 0.6, 0.3)

normal_worklength = [
    (15, [30, 20, 20]),
    (12, [ 30, 25]),
    (6, [15]),
    (7, [12, 8]),
]

vip_worklength = [
    (12, [30, 20, 20]),
    (10, [30, 25]),
    (6, [15]),
    (8, [12, 8]),
]

manager_rate = 0.5

worker_company_num = 2

is_skip_weekend = True