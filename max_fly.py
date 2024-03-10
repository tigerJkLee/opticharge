from gurobipy import*
import numpy as np

D = np.array([[5, 5, 5, 5, 5, 5, 10, 15, 20, 25], # charging
              [20, 25, 20, 15, 25, 30, 30, 25, 30, 35]]) # flying
m = 5 # number of charging pads
n = D.shape[1] # number of vehicles
g = np.gcd.reduce(np.concatenate((D[0, :], D[1, :]), axis=None)) # 5 최대공약수
D = D / g 
C = D[0, :]
T = D.sum(axis=0)
t = np.lcm.reduce(T.astype(int)) # 840 최소공배수
nums = int(sum(T)) # 71 총 운행시간 합계
T_max = int(max(T)) 


A = np.zeros((n, nums + 1)) # 10 x 72
p_matrix = []
q_matrix = []

k = 0

for i in range(n):
    A[i, k:k + int(T[i])] = np.ones(int(T[i]))  # equality
    k = k + int(T[i])

    p_i = np.zeros(int(T[i]))
    q_i = np.ones(int(T[i]))
    p_i[0:int(C[i])] = np.ones(int(C[i]))
    q_i[0:int(C[i])] = np.zeros(int(C[i]))
    p_matrix.append(p_i)
    q_matrix.append(q_i)

A_bar = np.zeros((t, nums))  # inequality
K_bar = np.zeros((t, nums))

for i in range(t):
    k = 0
    for j in range(n):
        A_bar[i, k:k + int(T[j])] = np.roll(p_matrix[j], shift=i, axis=0)
        K_bar[i, k:k + int(T[j])] = np.roll(q_matrix[j], shift=i, axis=0)
        k = k + int(T[j])
    # A_bar[i, -1] = -1

b = np.ones(n) # vehicle 수만큼
b_bar = np.zeros(t) # 

lb = np.zeros(nums)
ub = np.ones(nums)
# ub[-1] = np.inf
# lbofw = np.zeros((t,n))
# ubofw = np.ones((t,n))
lbofw = np.zeros(n)
ubofw = np.ones(n)


intcon = list(range(nums + 1))
c = np.zeros((nums + 1))
c[-1] = 1

# Create a new model
model = Model("mip")

# Define decision variables
r = model.addVars(nums, vtype=GRB.INTEGER, lb=lb, ub=ub, name="r") # r_0
w = model.addVars(n, vtype=GRB.INTEGER, lb=lbofw, ub=ubofw, name="w") # w

w_matrix = []
for i in range(n):
    w_matrix.extend([w[i]] * int(T[i]))  # Convert T[i] to an integer, in case it's a float

# w_nums 

objective = quicksum(K_bar[i, j] * r[j] for i in range(t) for j in range(nums))
model.setObjective(objective, GRB.MAXIMIZE)

# # Add constraints
for i in range(len(b)):
    model.addConstr(quicksum(K_bar[i, j] * r[j] for j in range(nums))) <= w_matrix[i]

for i in range(len(b)):
    model.addConstr(quicksum(c[i] * r[i] for i in range(len(c))) <= m)

for i in range(len(b_bar)):
    temp = np.zeros((1, nums))
    k = 0
    for j in range(n):
        temp[0, k:k + int(T[j])] = np.roll(p_matrix[j], shift=i, axis=0)
        k = k + int(T[j])
    model.addConstr(quicksum(temp[0, j] * r[j] for j in range(nums)) <= m)
    
# for i in range(t):
#     for j in range(n):
#         # Assume you are adding a constraint that involves the (i, j)-th element of the repeated matrix
#         model.addConstr(some_expression == w[j])



# Optimize model
model.optimize()

# Get results
if model.status == GRB.OPTIMAL:
    # x_values = model.getAttr('w', w)
    x_values = {v.varName: v.x for v in w.values()}
    fval = model.objVal
    print(f'Optimal Objective Value(max flight time): {fval}')
    print(f'Solution Status: {model.status}')
    print("x_values is: ", x_values)
else:
    print('Model could not be solved to optimality.')