import matplotlib.pyplot as plt


def make_plot():
    # Sample data
    x = [1, 2, 3, 4, 5]
    y = [2, 3, 5, 7, 11]

    # Create a plot
    plt.plot(x, y)

    # Add title and labels
    plt.title('Sample Plot')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Show the plot
    plt.show()
