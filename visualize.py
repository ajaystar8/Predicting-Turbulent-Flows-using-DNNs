import matplotlib.pyplot as plt
from config import *


def magnitude_vs_time_plot(actual_data, predicted_data):
    actual_sol = actual_data
    predicted_sol = predicted_data

    fig1 = plt.figure(figsize=(8, 8))

    time_points = np.linspace(0, 4000, 4001)

    a1, a2, a3, a4, a5, a6, a7, a8, a9 = [
        1, 0.07066, -0.07076, 0.0996, 0, 0, 0, 0, 0]

    title = rf"$a_1 = {a1}, a_2 = {a2}, a_3 = {a3}, a_4 = {a4}, a_5 = {a5}, a_6 = {a6}, a_7 = {a7}, a_8 = {a8}, \
    a_9 = {a9}$"

    fig1.suptitle(title)

    plt.subplot(5, 2, 1)
    plt.plot(time_points, actual_sol[:, 0], '--b', label=r"Actual $a_1$")
    plt.plot(time_points, predicted_sol[:, 0], '-r', label=r"Predicted $a_1$")
    plt.legend()

    plt.subplot(5, 2, 2)
    plt.plot(time_points, actual_sol[:, 1], '--b', label=r"Actual $a_2$")
    plt.plot(time_points, predicted_sol[:, 1], '-r', label=r"Predicted $a_2$")
    plt.legend()

    plt.subplot(5, 2, 3)
    plt.plot(time_points, actual_sol[:, 2], '--b', label=r"Actual $a_3$")
    plt.plot(time_points, predicted_sol[:, 2], '-r', label=r"Predicted $a_3$")
    plt.legend()

    plt.subplot(5, 2, 4)
    plt.plot(time_points, actual_sol[:, 3], '--b', label=r"Actual $a_4$")
    plt.plot(time_points, predicted_sol[:, 3], '-r', label=r"Predicted $a_4$")
    plt.legend()

    plt.subplot(5, 2, 5)
    plt.plot(time_points, actual_sol[:, 4], '--b', label=r"Actual $a_5$")
    plt.plot(time_points, predicted_sol[:, 4], '-r', label=r"Predicted $a_5$")
    plt.legend()

    plt.subplot(5, 2, 6)
    plt.plot(time_points, actual_sol[:, 5], '--b', label=r"Actual $a_6$")
    plt.plot(time_points, predicted_sol[:, 5], '-r', label=r"Predicted $a_6$")
    plt.legend()

    plt.subplot(5, 2, 7)
    plt.plot(time_points, actual_sol[:, 6], '--b', label=r"Actual $a_7$")
    plt.plot(time_points, predicted_sol[:, 6], '-r', label=r"Predicted $a_7$")
    plt.legend()

    plt.subplot(5, 2, 8)
    plt.plot(time_points, actual_sol[:, 7], '--b', label=r"Actual $a_8$")
    plt.plot(time_points, predicted_sol[:, 7], '-r', label=r"Predicted $a_8$")
    plt.legend()

    plt.subplot(5, 2, 9)
    plt.plot(time_points, actual_sol[:, 8], '--b', label=r"Actual $a_9$")
    plt.plot(time_points, predicted_sol[:, 8], '-r', label=r"Predicted $a_9$")
    plt.legend()

    plt.show()


def uxbar_vs_y_plot(U_X_pred, U_X_true):
    ux_bar_pred = np.zeros(nx)
    ux_bar_true = np.zeros(nx)

    for point in range(ny):
        ux_bar_pred[point] = np.sum(U_X_pred[:, point, :], axis=(0, 1))
        ux_bar_true[point] = np.sum(U_X_true[:, point, :], axis=(0, 1))
    ux_bar_pred *= (1 / (nx * nz))
    ux_bar_true *= (1 / (nx * nz))

    plt.plot(ux_bar_pred, y, 'red', label='ux_bar_pred', linewidth=2)
    plt.plot(ux_bar_true, y, linestyle='dotted',
             color='blue', label='ux_bar_true', linewidth=2)
    plt.title('ux_bar vs. y')
    plt.xlabel('u̅')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def quiver_contour_plots_xavg(ux, uy, uz, t):
    Z, Y = np.meshgrid(z, y)
    plt.xlabel("$z$")
    plt.ylabel("$y$")

    ux_avg = np.mean(ux, axis=0).reshape(ny, nz)
    uy_avg = np.mean(uy, axis=0).reshape(ny, nz)
    uz_avg = np.mean(uz, axis=0).reshape(ny, nz)

    contour = plt.contourf(Z, Y, ux_avg, cmap='jet')
    plt.colorbar(contour, label='ux_avg')

    plt.quiver(Z, Y, uz_avg, uy_avg, color='k')

    plt.title(r"$t = {0}$".format(t), fontsize=15)
    plt.show()
