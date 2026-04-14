from NSGA2 import basin_primitive_f
import numpy as np

values = []
for x1 in np.linspace(-1,1,1000):
    for x2 in np.linspace(-1,1,1000):
        if abs(x1)+abs(x2) <= 1:
            values.append(basin_primitive_f(x1,x2))

min_val = min(values)
max_val = max(values)

print(f"min: {min_val}, max: {max_val}")