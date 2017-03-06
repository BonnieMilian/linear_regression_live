from numpy import *
import numpy as np
# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, points):
    #initialize it at 0
    totalError = 0
    #for every point
    for i in range(0, len(points)):
        #get the x value
        x = points[i, 0]
        #get the y value
        y = points[i, 1]
        #get the difference, square it, add it to the total
        totalError += (y - (m * x + b)) ** 2
    
    #get the average
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
    #stating points for our gradients
    b_gradient = 0
    m_gradient = 0

    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    
    #update our b and m values using our partial  derivatives
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
     #starting b and m
    b = starting_b
    m = starting_m

    #gradient descent
    for i in range(num_iterations):
        #update b and m with the new more accurate b and m by performing
        #this is gradient step
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

def train(initial_b, initial_m, num_iterations, learning_rate, points):
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    return [b, m]

def plot(bs, ms, points, colors):  
    try:
        import  matplotlib.pyplot as plt
        plt.plot(points[:,0], points[:,1], 'ro', color='black')
        
        for i in range(0, len(colors)):
            plt.plot(points[:,0], ms[i]*points[:,0] + bs[i], color=colors[i], linewidth=3)
        plt.show()
    except:
        pass
    return

def run():
    #Step 1 - collect our data
    points = genfromtxt("data.csv", delimiter=",")
    
    #Step 2 - define our hyperparameters
    #how fast should our model converge?
    learning_rate = 0.0001
    #y = mx + b
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 1 #how much train the model

    b_s = array([]) #b in each iteration
    m_s = array([]) #m in each iteration
    colors = array(['crimson','red','orange','yellow','limegreen','green','teal','blue'])

    #Step 3 - train our model
    print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points))
    print "Running..."
    
    for i in range(0, len(colors)):
        [b, m] = train(initial_b, initial_m, num_iterations, learning_rate, points)
        b_s = np.append(b_s, [b])
        m_s = np.append(m_s, [m])
        print "After {0} iterations b = {1}, m = {2}, error = {3}, FRM {4}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points), i + 1)
        num_iterations += 1
    
    plot(b_s, m_s, points, colors)
    
if __name__ == '__main__':
    run()
