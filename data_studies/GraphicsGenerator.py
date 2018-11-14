import matplotlib.pyplot as plt

def create_comparison_graphic_with(input_y, predicted_y):

    if(len(input_y) != len(predicted_y)):
        print("Both dataframes should have the same amount of elements")

    plt.plot(range(0, len(input_y)), input_y, predicted_y)
    plt.show()