import matplotlib.pyplot as plt

populationAges = [22,55,23,26,88,55,34,74,35,76,6,3,66,33,65,43,20,49,69] ## list of values to plot

## idAges = [age_range for age_range in range(len(populationAges))] ## var create a list of the values of populationAges, e.g. (0 = 22, 1 = 55, etc..)

bins = [0,10,20,30,40,50,60,70,80] ## a container for the values to be held in between
                                                  ## the amount of bins = the amount of bars in the chart
                                                  ## helps condense data

plt.hist(populationAges, bins, label = 'Values', histtype = 'bar', rwidth = 0.8) ## calls on the .hist function to plot a histogram
                                                               ## the histogram uses the populationAges and bins for the values and their ranges
                                                               ## while histtype specifies what style of histogram will be displayed
                                                               ## rwidth set quite low to not take up too much space in this small example of a chart


plt.xlabel('X Axis') ## naming the x axis
plt.ylabel('Y Axis') ## naming the y axis
plt.title('Example Bar Chart') ## naming the title of the barchart window
plt.legend() ## this represents the key for the data being graphically shown
plt.show() ## this displays the figure
