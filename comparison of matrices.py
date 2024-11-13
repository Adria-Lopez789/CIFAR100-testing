import numpy as np
import math
import matplotlib.pyplot as plt
from numpy.core.arrayprint import set_printoptions
from sklearn.metrics import ConfusionMatrixDisplay

class_spread_matrix = []
total_spread_array = []
total_accuracy_array = []
badValues = []

for epoch in range(10):
    for batch in range(391):
        matrix = np.load('matrices_numbers_CI_lrIncrease/confusion_matrix_' + str(epoch) + '_' + str(batch) + '.npy')

        diagonalValues = 0
        totalValues = 0
        np.set_printoptions(threshold=np.inf,linewidth=155)
        #print(matrix)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[0]):
                if(i==j):
                    diagonalValues += matrix[i,j]
                totalValues += matrix[i,j]
        accuracy = diagonalValues/totalValues

        badValuesRow = []
        for i in range(matrix.shape[0]):
            badValuesSingle = 0
            for j in range(matrix.shape[0]):
                if (i != j):
                    badValuesSingle += matrix[i, j]
            badValuesRow.append(badValuesSingle)
        badValues.append(badValuesRow)

        #Now calculate the spread
        spread = [0]*matrix.shape[0]
        for i in range(matrix.shape[0]):
            row_sum = np.sum(matrix[i, :])
            for j in range(matrix.shape[0]):
                if(i!=j):
                    probability = matrix[i,j]/row_sum
                    if(probability != 0):
                        spread[i] = spread[i] + (probability*math.log(probability))
        total_spread = -sum(spread)

        total_spread_array.append(total_spread)
        total_accuracy_array.append(accuracy)
        class_spread_matrix.append(spread)
        print("matrix calculated-- epoch:" + str(epoch) + " batch:" + str(batch))
    transposed_matrix = np.array(badValues).T
    for i, column in enumerate(transposed_matrix):
        #if(column[0] == 100 and i<25):
        if(i == 43 or i == 45):
            plt.plot(column, marker='o')
            plt.ylim(0, 100)
            plt.title("Class comparison 43,45 overtime")
            plt.xlabel("mini Batches")
            plt.ylabel("Incorrect assessments")
            plt.savefig('CI_lrIncrease/Class_' + 'dw' + '_plot.png')

    print(1)

"""
    np_class_spread_matrix = np.array(class_spread_matrix)

    plt.figure(figsize=(10, 8))  # Set the figure size
    plt.imshow(np_class_spread_matrix, cmap='viridis', aspect='auto')  # Use the viridis colormap
    plt.colorbar()  # Add a colorbar to indicate the value scale
    plt.title("Heatmap of Large Matrix Spread")
    plt.xlabel("Spread of each class")
    plt.ylabel("Batch")
    plt.savefig('spread_matrix.png', dpi=1000)
    plt.show()  # Display the plot


    plt.plot(np.array(total_spread_array))
    plt.title('Spread change over batches(conf_mat)')
    plt.xlabel('Batch')
    plt.ylabel('Spread')
    plt.grid(True)  # Optional: adds a grid for better readability
    plt.savefig('spread_array_plot.png')
    plt.show()

    plt.plot(np.array(total_accuracy_array))
    plt.title('Accuracy change over batches(conf_mat)')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    plt.grid(True)  # Optional: adds a grid for better readability
    plt.savefig('accuray_array_plot.png')
    plt.show()
"""




