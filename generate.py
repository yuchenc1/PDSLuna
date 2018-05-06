import numpy as np

def generate_view(cube):
    generated_cubes = [cube]

    for i in range(3):
        generated_cubes.append(np.rot90(cube, i+1, (0,1)))
        generated_cubes.append(np.rot90(cube, i+1, (0,2)))
        generated_cubes.append(np.rot90(cube, i+1, (1,2)))

    return generated_cubes

x = np.array([[[0,1],[2,3]],[[4,5],[6,7]]])
tmp = generate_view(x)
print(len(tmp))