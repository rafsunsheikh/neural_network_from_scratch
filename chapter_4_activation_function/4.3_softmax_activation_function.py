dense_outputs = [4.8, 1.21, 2.385]

E = 2.71828182846

exp_values = []

for i in dense_outputs:
    exp_values.append(E ** i)

exp_sum = sum(exp_values)

norm_exp = []
for i in exp_values:
    norm_exp.append(i / exp_sum)

print(norm_exp)