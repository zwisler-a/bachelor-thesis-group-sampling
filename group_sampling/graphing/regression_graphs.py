from matplotlib import pyplot as plt


def plot_linear_regression(x_test, y_test, y_predict, title):
    # Plot outputs
    plt.title(title)
    plt.scatter(x_test, y_test, color="black")
    plt.plot(x_test, y_predict, color="blue", linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()
