import matplotlib.pyplot as plt

def create_comparison_graphic_with(input_y, predicted_y):

    if(len(input_y) != len(predicted_y)):
        print("Both dataframes should have the same amount of elements")
        return

    plt.plot(input_y)
    plt.plot(predicted_y)
    plt.legend(labels=['input_y', 'predicted_y'])
    plt.show()


def plot_loss(loss):
    plt.plot(loss)
    plt.legend(labels=['loss'])
    plt.show()
