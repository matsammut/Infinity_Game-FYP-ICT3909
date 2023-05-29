import matplotlib.pyplot as plt


def plot_2hr():
    x_values = [500, 500, 500, 1000, 1000, 1000, 1500, 1500, 1500, 2000, 2000, 2000, 2500, 2500, 2500, 3000, 3000, 3000]
    y_values = [0, 1, 0, 2, 2, 1, 4, 5, 4, 6, 4, 5, 6, 7, 6, 7, 7, 8]

    # Create scatter plot
    plt.scatter(x_values, y_values)

    # Set axis labels and title
    plt.xlabel('Grabs')
    plt.ylabel('Clusters')
    plt.title('2hr Screen Grabs')

    # Set axis limits
    plt.xlim(0, 2500)
    plt.ylim(0, 15)

    # Show plot
    plt.show()


def plot_1_5hr():
    x_values = [500, 500, 500, 1000, 1000, 1000, 1500, 1500, 1500, 2000, 2000, 2000, 2500, 2500, 2500, 3000, 3000, 3000]
    y_values = [1, 1, 0, 1, 1, 2, 4, 5, 3, 3, 4, 5, 5, 5, 6, 7, 7, 6]

    # Create scatter plot
    plt.scatter(x_values, y_values)

    # Set axis labels and title
    plt.xlabel('Grabs')
    plt.ylabel('Clusters')
    plt.title('1hr 30min Screen Grabs')

    # Set axis limits
    plt.xlim(0, 2500)
    plt.ylim(0, 15)

    # Show plot
    plt.show()


def plot_1hr():
    x_values = [500, 500, 500, 1000, 1000, 1000, 1500, 1500, 1500, 2000, 2000, 2000, 2500, 2500, 2500, 3000, 3000, 3000]
    y_values = [0, 0, 1, 2, 2, 3, 4, 3, 4, 3, 3, 3, 6, 4, 5, 8, 7, 8]

    # Create scatter plot
    plt.scatter(x_values, y_values)

    # Set axis labels and title
    plt.xlabel('Grabs')
    plt.ylabel('Clusters')
    plt.title('1hr Screen Grabs')

    # Set axis limits
    plt.xlim(0, 2500)
    plt.ylim(0, 15)

    # Show plot
    plt.show()


def plot_5hr():
    x_values = [500, 500, 500, 1000, 1000, 1000, 1500, 1500, 1500, 2000, 2000, 2000, 2500, 2500, 2500, 3000, 3000, 3000]
    y_values = [0, 0, 0, 2, 2, 1, 3, 3, 4, 5, 4, 5, 6, 6, 5, 6, 6, 7]

    # Create scatter plot
    plt.scatter(x_values, y_values)

    # Set axis labels and title
    plt.xlabel('Grabs')
    plt.ylabel('Clusters')
    plt.title('30min Screen Grabs')

    # Set axis limits
    plt.xlim(0, 2500)
    plt.ylim(0, 15)

    # Show plot
    plt.show()


plot_5hr()
plot_1hr()
plot_1_5hr()
plot_2hr()
