# 🎯 Project Goal and Objective

## 1. One-line Goal
To develop an **Employee Scheduling System using Timefold (Python)** that can automatically assign employees to operations and dates based on skills, capacity, and constraints provided by the customer (“SP”).

---

## 2. Business Objective (Customer View)
The purpose of this project is to help **SP** (the scheduling provider / client) automatically generate optimized employee schedules from YAML input files.

SP will define:
- Available employees, their skills, and manager status.  
- Factory workflows, operation windows, and workload requirements.  

The system will then:
1. Read the provided YAML configuration (`EnvConfig.yaml` and `Schedule.yaml`).
2. Optimize schedules using Timefold’s solver engine.
3. Output a new YAML file with filled assignments.

This helps SP:
- Reduce manual scheduling time.
- Avoid overworking or cross-factory conflicts.
- Ensure every team block has at least one manager.
- Achieve fair and balanced workload among employees.

---

## 3. Project Objective (Developer View)
- Implement a **two-pass architecture** for scheduling:
  - **Pass 1:** Determine block start dates, duration, and crew size.
  - **Pass 2:** Assign qualified employees to those blocks.  
- Ensure hard and soft constraints are enforced:
  - Respect factory windows, heads limits, workloads, and skill levels.
  - Balance total working hours and avoid overtime.
- Maintain modular, YAML-driven design for flexibility with new inputs.
- Keep code easily testable with mock data and Excel exports for result visualization.

---

## 4. Current Stakeholders
| Role                         | Description                                                   |
| ---------------------------- | ------------------------------------------------------------- |
| **SP (Scheduling Provider)** | Provides workflow data and expected scheduling problems.      |
| **Developer (__)**           | Implements solver logic, YAML model, and export modules.      |
| **Customer / End-User**      | Uses generated schedules to manage actual workforce planning. |

---

## 5. Long-Term Vision
- Integrate a web dashboard for schedule visualization.
- Add predictive modules (AI-assisted planning).
- Enable multi-day optimization with automatic manager allocation per block-day.
- Eventually generalize this system for other scheduling scenarios (factory, shift, project).

---

## 6. Reference Keywords
`Timefold`, `Employee Scheduling`, `Python`, `Optimization`, `Manager Logic`, `Two-Pass Solver`, `EnvConfig.yaml`, `Schedule.yaml`
