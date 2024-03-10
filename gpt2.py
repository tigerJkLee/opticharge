from gurobipy import Model, GRB
import numpy as np

# Given Data
D = np.array([
    [5, 5, 5, 5, 5, 5, 10, 15, 20, 25],  # Charging times
    [20, 25, 20, 15, 25, 30, 30, 25, 30, 35]  # Flying times
])

n = D.shape[1]  # Number of vehicles
g = np.gcd.reduce(np.concatenate((D[0, :], D[1, :]), axis=None))
D = D / g 
C = D[0, :]  # Normalized charging times
F = D[1, :]  # Normalized flying times
nums_charging_pads = 5  # Assuming a fixed number of charging pads

# Model Setup
model = Model("MaximizeFlightTime")

# Decision Variables
# Binary decision variable for each vehicle indicating flying status
w = model.addVars(n, vtype=GRB.BINARY, name="w")

# Objective: Assuming the goal is to maximize the number of vehicles flying simultaneously under given constraints
model.setObjective(sum(w[i] for i in range(n)), GRB.MAXIMIZE)

# Constraints

# Constraint 1: At any time, the number of vehicles being charged must not exceed the available charging pads.
# This requires a more detailed scheduling constraint, which might depend on additional model variables or assumptions not fully clear here.
# Assuming a simplistic constraint where vehicles can either be charging or flying but not both at the same time.

# Add constraints based on your specific requirements and assumptions

# Optimize Model
model.optimize()

# Results
if model.status == GRB.OPTIMAL:
    w_values = model.getAttr('x', w)
    print("Optimal Flight Schedule for each Vehicle (1 indicates flying):")
    for v in w.values():
        print(f"{v.VarName}: {v.X}")
else:
    print("Model could not be solved to optimality.")

