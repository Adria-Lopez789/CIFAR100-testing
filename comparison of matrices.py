import numpy as np
import math
import matplotlib.pyplot as plt
class_spread_matrix = []
total_spread_array = []
total_accuracy_array = []
for batch in range(622):
    matrix = np.load('matrices_numbers/confusion_matrix_' + str(batch) + '.npy')

    diagonalValues = 0
    totalValues = 0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            if(i==j):
                diagonalValues += matrix[i,j]
            totalValues += matrix[i,j]
    accuracy = diagonalValues/totalValues

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



