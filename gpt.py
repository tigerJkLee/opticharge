from gurobipy import Model, GRB
import numpy as np

# Problem Data
D = np.array([[5, 5, 5, 5, 5, 5, 10, 15, 20, 25],  # Charging times
              [20, 25, 20, 15, 25, 30, 30, 25, 30, 35]])  # Flying times

n = D.shape[1]  # Number of vehicles/operations
g = np.gcd.reduce(np.concatenate((D[0, :], D[1, :]), axis=None))  # Greatest common divisor
D = D / g  # Normalize times by gcd
C = D[0, :]  # Normalized charging times
F = D[1, :]  # Normalized flying times
T = D.sum(axis=0)  # Total operation times (charging + flying)
t = np.lcm.reduce(T.astype(int))  # Least common multiple for operation cycles
nums = int(sum(T))  # Sum of all operation times

# Initialization
model = Model("MaximizeFlightTime")

# Define decision variables for flying times
flight_times = model.addVars(n, vtype=GRB.INTEGER, lb=0, ub=F, name="flight_times")

# Objective: Maximize the total flight time
model.setObjective(sum(flight_times[i] for i in range(n)), GRB.MAXIMIZE)

# Constraints ensuring each vehicle's total operation time does not exceed its cycle
for i in range(n):
    model.addConstr(flight_times[i] + C[i] <= T[i], name=f"op_cycle_{i}")

# Optimize model
model.optimize()

# Results
if model.status == GRB.OPTIMAL:
    optimal_flight_times = model.getAttr('x', flight_times)
    total_flight_time = sum(optimal_flight_times[i] for i in range(n))
    print(f'Maximum Total Flight Time: {total_flight_time}')
    print("Solution Status:", model.status)
    print("Optimal Flight Times for Each Vehicle:", optimal_flight_times)
else:
    print('Model could not be solved to optimality.')
