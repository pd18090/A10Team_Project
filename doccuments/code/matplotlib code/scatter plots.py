import matplotlib.pyplot as plt

x = [1,2,3,4,5,6,7,8,9,10] ## values for x axis
y = [7,3,4,4,6,6,7,8,2,5] ## values for y axis

plt.scatter(x,y, label = 'Scatter Graph', colour = 'r', marker = '*', s = 50) ## calls on the .scatter function to plot a scatter graph
                                                                              ## uses the x and y lists for the values to be plotted
                                                                              ## label and colour functions are being called again to properly represent the graph
                                                                              ## the marker function can be changed to other preset chars and the size can be change with the func 's'

plt.xlabel('X Axis') ## naming the x axis
plt.ylabel('Y Axis') ## naming the y axis
plt.title('Example Bar Chart') ## naming the title of the barchart window
plt.legend() ## this represents the key for the data being graphically shown
plt.show() ## this displays the figure
