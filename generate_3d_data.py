import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from mpl_toolkits.mplot3d import Axes3D

size = 50


def random_sphere():
    center = np.random.randint(0, size, size=3, dtype='int')
    max_radius = size
    for i in range(len(center)):
        max_radius = min(max_radius, min(center[i] + 1, size - center[i]))
    radius = np.random.randint(1, max_radius+1, dtype='int')
    weight = np.random.randint(1, size, dtype='int')
    return center, radius, weight


def generate_data():
    data = np.zeros((size, size, size))
    for i in range(size):
        c, r, w = random_sphere()
        cx, cy, cz = c[0], c[1], c[2]

        y, x = np.ogrid[-cx: size - cx, -cy: size - cy]
        mask = x ** 2 + y ** 2 <= r ** 2
        data[cz][mask] += w

        for j in range(1, r + 1):
            mask = x**2 + y**2 <= (r-j)**2
            if cz - j >= 0:
                data[cz - j][mask] += w
            if cz + j < size:
                data[cz + j][mask] += w
    return data


def generate_attention():
    a = np.zeros((size, size, size))
    c, r, w = random_sphere()
    cx, cy, cz = c[0], c[1], c[2]

    y, x = np.ogrid[-cx: size - cx, -cy: size - cy]
    mask = x ** 2 + y ** 2 <= r ** 2
    a[cz][mask] += 1

    for j in range(1, r + 1):
        mask = x**2 + y**2 <= (r-j)**2
        if cz - j >= 0:
            a[cz - j][mask] += 1
        if cz + j < size:
            a[cz + j][mask] += 1
    return a


def plot(data, attention):
    data_x = []
    data_y = []
    data_z = []
    att_x = []
    att_y = []
    att_z = []
    for i in range(size):
        for j in range(size):
            for k in range(size):
                if data[i, j, k] > 0:
                    data_x.append(i)
                    data_y.append(j)
                    data_z.append(k)
                if attention[i, j, k] > 0:
                    att_x.append(i)
                    att_y.append(j)
                    att_z.append(k)
    weights = [v for v in data.flatten() if v != 0]
    weights /= max(weights)

    att = [v for v in attention.flatten() if v != 0]
    att /= max(att)

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(111, projection='3d')
    colmap = cm.ScalarMappable(cmap=cm.Blues)

    colmap.set_array(weights)
    ax.scatter(data_x, data_y, data_z, c=cm.Blues(weights), marker='x')

    colmap.set_array(att)
    ax.scatter(att_x, att_y, att_z, c=cm.Reds(att), marker='x')

    fig.colorbar(colmap)

    plt.xlim([0, size])
    plt.ylim([0, size])
    ax.set_zlim([0, size])

    plt.show()


def main():
    data = generate_data()
    attention = generate_attention()
    plot(data, attention)


if __name__ == '__main__':
    main()
