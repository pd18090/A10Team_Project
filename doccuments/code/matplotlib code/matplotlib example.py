import matplotlib.pyplot as plt ## module to plot graphs
import matplotlib.animation as animation ## to allow graph animations

graph = plt.figure() ## defining the figure with empty parameters
axis1 = graph.add_subplot(1,1,1) ## defining the sole axis for now, with a 1x1 size, with only
                                 ## 1 chart

def animate(i): ## function for the visualisation of the graph
    readData = open("sample data.txt", "r").read() ## var for opening and reading data set,
                                                   ## which is the sample data text file
    arrayData = readData.split('\n') ## the text file is read line by line by splitting the lines
                                     ## and storing it in the arrayData var
    x_Axis_array = [] ## empty array for x axis
    y_Axis_array = [] ## empty array for y axis
    for lineData in arrayData: ## for loop checking every number in arrayData
        if len(lineData) > 1: ## if the length of the current number its checking is more than 1
            x,y = lineData.split(',') ## splits the number with a comma (,)
                                      ## and stores it in two separate vars (x value and y value)
            x_Axis_array.append(int(x)) ## appends the x values into the empty x axis array
            y_Axis_array.append(int(y)) ## appends the y values into the empty y axis array
    axis1.clear() ## clears the subplots created before
    axis1.plot(x_Axis_array, y_Axis_array) ## plots the new x and y axis' that have been assigned
                                           ## new values (from x,y)

animateGraph = animation.FuncAnimation(graph, animate, interval = 1000) ## calling on the figure that's been drawn,
                                                                        ## the function you want it pass through,
                                                                        ## and how often we refresh the graph (1000ms)
                            
plt.show() ## show the graph we've defined
