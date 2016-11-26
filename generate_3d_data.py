import numpy as np
import matplotlib.pyplot as plt

size = 5
dimen = 2


def random_sphere():
    center = np.random.randint(0, high=size, size=3, dtype='int')
    print('center: ' + str(center))
    max_radius = size
    for i in range(len(center)):
        max_radius = min(max_radius, min(center[i] + 1, size - center[i]))
    print('max radius: ' + str(max_radius))
    radius = np.random.randint(1, high=max_radius+1, dtype='int')
    print('radius: ' + str(radius))
    return center, radius


def generate_data():
    data = np.zeros((size, size, size))
    for i in range(1):
        c, r = random_sphere()
        cx, cy, cz = c[0], c[1], c[2]

        y, x = np.ogrid[-cx: size - cx, -cy: size - cy]
        mask = x ** 2 + y ** 2 <= r ** 2
        data[cz][mask] += 1

        for j in range(1, r + 1):
            mask = x**2 + y**2 <= (r-j)**2
            if cz - j >= 0:
                data[cz - j][mask] += 1
            if cz + j < size:
                data[cz + j][mask] += 1
    return data


def main():
    data = generate_data()
    print(data)


if __name__ == '__main__':
    main()
