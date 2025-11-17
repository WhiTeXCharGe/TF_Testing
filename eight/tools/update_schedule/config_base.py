WORKER_NUM = 80
EQ_PER_DAYS = 0.5
EQ_PER_DAYS_SIGMA = 0.5
EQ_NUM = 30

skill_level_list = (1,2,3,4,5)
skill_level_weights = (0.05, 0.1, 0.5, 0.25, 0.1)

normal_worklength = [
    (15, [30, 20, 20]),
    (10, [ 30, 25]),
    (5, [15]),
    (10, [12, 8]),
]

vip_worklength = [
    (8, [30, 20, 20]),
    (6, [30, 25]),
    (5, [15]),
    (8, [12, 8]),
]

manager_rate = 0.3

worker_company_num = 2

is_skip_weekend = False