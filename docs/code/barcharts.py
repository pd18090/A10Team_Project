import matplotlib.pyplot as plt

x = [2, 4, 6, 8, 10] ## list for the bar
y = [1, 3, 5, 7, 9] ## list for the bar

x2 = [1,3,5,7,9] ## second list for the bar
y2 = [3,4,6,7,8] ## second list for the bar

plt.bar(x, y, label = "Bar 1", color = 'r') ## the 1st bar chart, which uses the x and y lists for the values.
                                             ## the label and colour function are also called to customise the first barchart
plt.bar(x2, y2, label = "Bar 2", color = 'b') ## the 2nd bar chart, which uses the x2 and y2 lists for the values.
                                               ## the label and colour function are called again

plt.xlabel('X Axis') ## naming the x axis
plt.ylabel('Y Axis') ## naming the y axis
plt.title('Example Bar Chart') ## naming the title of the barchart window
plt.legend() ## this represents the key for the data being graphically shown
plt.show() ## this displays the figure
