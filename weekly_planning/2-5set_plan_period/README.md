# Setting the plan period

The tool reads Schedule and EnvConfig files and writes the Schedule with a calculated
plan period. The start date is the later of the earliest workflow phase start and
today + 7 calendar days. The end date is the later of the latest workflow phase
end + 30 calendar days and the plan start + 30 calendar days. If there are no
phases, the start date is today + 7 calendar days and the end date is the plan
start + 30 calendar days.

An assignment with any work date before the calculated start date becomes `Fixed`
as a whole; assignment rows are not split. EnvConfig is copied byte-for-byte.

Specify the input and output file paths with the required arguments
`--input-schedule`, `--input-env-config`, `--output-schedule`, and
`--output-env-config`. Use `--plan-start YYYY/MM/DD` to override the calculated
start date without changing phase dates. An output-prefix option is not provided
because output paths are specified individually. Report output is outside the
current design scope.

Python callers can use `set_plan_period.set_plan_period(input_schedule,
input_env_config, output_schedule, output_env_config, plan_start=None)`.
The four path arguments accept strings or `pathlib.Path` objects; `plan_start`
is an optional `datetime.date`.

Missing or invalid phase dates, a missing or empty work-date list, invalid work
dates, and conflicts between assignment periods and work dates are input format
errors. Reversed phase or assignment date ranges, missing required Schedule keys,
invalid `plan_flexibility` values, and overlapping input/output paths are also
rejected before writing either output.
