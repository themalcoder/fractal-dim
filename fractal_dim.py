import os
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import linregress

# ## The Given Equation
# 
# ## $$ \frac{d^2 x}{d t^2} = -\gamma \frac{d x}{d t} + 2 a x - 4 b x^3 + F_0 \cos(\omega t) \qquad \gamma, a, b > 0 $$ 

# So first let's convert this equation into two-first order Differentrial Equations.
# 
# Let $$ x_1 = x $$
# $$ x_2 = x' = x_1' $$
# $$ x_2' = x'' $$
# 
# Then we have,
# 
# ### $$ x_1' = x_2 $$
# ### $$ x_2' = -\gamma x_2 + 2 a x_1 - 4 b x_1^3 + F_0 \cos(\omega t) $$
# 
# Now these two will be the equations that we will be playing with.
# 
# 
# Now we create a "model" for these equations

def middle(x, a, b):
    result = (2 * a * x) - (4 * b * x ** 3)
    return result

def system(X, t, a, b, gamma, F0, omega):
    ''' This will return the derivatives of dx/dt and d^2 x / dt^2 '''

    x1, x2 = X
    
    x2_prime = -(gamma * x2) + middle(x1, a, b) + (F0 * np.cos(omega * t))

    # x2 ---> xdot 
    # and x2_prime ---> xdotdot
    
    return [x2, x2_prime]

# Created a model for the system. Let's solve it.

def solve_duffing(model, args_dict):
    x0 = args_dict['x0']
    t = args_dict['t']
    a = args_dict['a']
    b = args_dict['b']
    gamma = args_dict['gamma']
    F0 = args_dict['F0']
    omega = args_dict['omega']

    args = (a, b, gamma, F0, omega)

    sol = odeint(model, x0, t, args=args)

    x = sol[:, 0]
    x_prime = sol[:, 1]

    return [x, x_prime]

def strange_attractor(x, x_prime, t, omega, h):
    t_period = (2 * np.pi / omega)
    strange = {'x': [], 'x_prime': []}

    k = 1
    for i in range(len(t)):
        if np.abs(t[i] - k * t_period) < h:
            strange['x'].append(x[i])
            strange['x_prime'].append(x_prime[i])
            k += 1
            
    return strange

def plot_graph(x, y, p_range = None, args_dict = None, scatter = False, show_grid = True, save_as=None):
    label = args_dict['label']
    xlabel = args_dict['xlabel']
    ylabel = args_dict['ylabel']
    title = args_dict['title']
    
    if scatter is False:
        if p_range is None:
            ''' If xrange and yrange is not given, then plot with every point'''
            plt.plot(x, y, label=label)
        else:
            r1, r2 = p_range[0], p_range[1]
            plt.plot(x[r1:r2], y[r1:r2], label=label) # if we want to plot a subsection of t_series (x axis quantity)
    else:
        plt.scatter(x, y, s=7, lw=0, c=sb.color_palette('tab10')[0], label=label)
    
    plt.title(title)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if show_grid:
        plt.grid()

    if save_as is not None:
        plt.savefig(save_as)
    # plt.show()


## Now to calculate the fractal dimension...
def create_boxes_coordinates(data, divisions):
    ''' This will return the boxes coordinates. 
    Let's assume the grid is square and divisions is actually the number by which each side is divided by.
    So, if divisions is 2, the number of boxes will be 4 and number of coordianates will be 9 and so on. '''

    no_of_coordinates = divisions + 1
    
    # print(f"Grid will have {divisions ** 2} boxes and {no_of_coordinates ** 2} sets of co-ordinates")
    x_data, y_data = data

    x_lim = [np.min(x_data) - 1 , np.max(x_data) + 1]
    y_lim = [np.min(y_data) - 1, np.max(y_data) + 1]


    lim = [np.round(x_lim), np.round(np.round(y_lim))]

    grid_size = (np.min(lim[0]), np.max(lim[1]))
    
    box_dims_x = np.linspace(grid_size[0], grid_size[1], no_of_coordinates)
    box_dims_y = box_dims_x
    
    boxes_list = [] # coordinates of the boxes

    for i in box_dims_x:
        temp_coords = []
        for j in box_dims_y:
            temp_coords = [j, i]
            boxes_list.append(temp_coords)

    return boxes_list

def create_boxes(boxes_data, divisions):
    ''' Returns the boxes as an array of 4 coordinates corresponding to them.
    boxes_data is the data obtained after running create_boxes_coordinates function
    '''
    Offset = divisions + 1
    boxes = []
    for idx in range(divisions ** 2):
        temp_data = [boxes_data[0 + idx], boxes_data[1 + idx], boxes_data[Offset  + idx], boxes_data[(Offset + 1) + idx]]
        boxes.append(temp_data)
    
    return boxes

def point_in_box(point, box):
    x1, y1 = box[0]
    x2, y2 = box[3]
    x, y = point

    if (x1 <= x and x <= x2):
        if (y1 <= y and y <= y2):
            return True
    return False

def fractal_dimension(data, divisions):
    x, y = data
    box_coords = create_boxes_coordinates(data=data, divisions=divisions)
    boxes = create_boxes(boxes_data=box_coords, divisions=divisions)

    points = [[x[i], y[i]] for i in range(len(x))]

    counter = 0

    for box in boxes:
        for point in points:
            if point_in_box(point=point, box=box):
                counter += 1
                break
    
    return counter


def main(args, t):
    for k in args.keys():
        print(f'\n\n[+] Solving for the set-{k + 1}')
        print("------------------------------------")
        x0 = args[k][0]
        a = args[k][1]
        b = args[k][2]
        gamma = args[k][3]
        F0 = args[k][4]
        omega = args[k][5]

        if not os.path.exists('graphs/'):
            os.mkdir('graphs')
        if not os.path.exists(f'graphs/set-{k + 1}'):
            os.mkdir(f'graphs/set-{k + 1}')
        

        args_dict = {'x0': x0, 't': t, 'a': a, 'b': b, 'gamma': gamma, 'F0': F0, 'omega': omega}
        x, x_prime = solve_duffing(system, args_dict=args_dict)

        title = r"Position vs time"
        label = r"$x(t)$ vs $t$"
        xlabel = r"$t$"
        ylabel = r"$x(t)$"
        plot_dict = {'label': label, 'xlabel': xlabel, 'ylabel': ylabel, 'title': title}
        save_as_base = f'graphs/set-{k + 1}'
        plot_graph(t, x, p_range=[1000, 2000], args_dict=plot_dict, save_as=f'{save_as_base}/position-vs-time.png')
        plt.clf()
        print(f'[+] Graph saved in {save_as_base}/')

        title = r"Phase space"
        label = r"$\dot{x}$ vs $x$"
        xlabel = r"$x$"
        ylabel = r"$\dot{x}$"
        plot_dict = {'label': label, 'xlabel': xlabel, 'ylabel': ylabel, 'title': title}
        plot_graph(x, x_prime, p_range = [2000, 4000], args_dict=plot_dict, save_as=f'{save_as_base}/phase-space.png')
        plt.clf()
        print(f'[+] Graph saved in {save_as_base}/')

        strange_dict = strange_attractor(x, x_prime, t, args_dict['omega'], h=0.1)

        title = r"Poincare Plot (Phase space at T = $\frac{2\pi N}{\omega}$)"
        label = r"$\dot{x}$ vs $x$"
        xlabel = r"$x$"
        ylabel = r"$\dot{x}$"
        plot_dict = {'label': label, 'xlabel': xlabel, 'ylabel': ylabel, 'title': title}
        plot_graph(strange_dict['x'], strange_dict['x_prime'], args_dict=plot_dict, scatter=True, show_grid=False, save_as=f'{save_as_base}/poincare.png')
        plt.clf()
        print(f'[+] Graph saved in {save_as_base}/')

        data = [strange_dict['x'], strange_dict['x_prime']]
        divisions_list = [2, 4, 5, 8, 10, 16, 20, 32, 40, 64, 80, 100, 200]

        counts = []
        for divisions in divisions_list:
            count = fractal_dimension(data=data, divisions=divisions)
            counts.append(count)

        x = np.log(divisions_list)
        y = np.log(counts)

        linear_regression = linregress(x, y)
        slope = linear_regression[0]

        print(f"\n\nFractal Dimension (for set-{k+1}) = {slope}")
        print('-----------------------------------')



if __name__ == '__main__':
    x0_list = [[0.25, 0], [0.65, 0], [0.4, 0], [0.5, 0.5]]
    a_list = [0.5, 0.25, 0.5, 0.75]
    b_list = [0.25, 0.5, 1 / 16, 1 / 8]
    gamma_list = [1.24, 0.4, 0.2, 0.1]
    F0_list = [3.25, 2.25, 2.05, 2.0]
    omega_list = [3.7, 2.3, 2.5, 2.25]

    conditions_dict = {}
    conditions_dict = {k: [x0_list[k], a_list[k], b_list[k], gamma_list[k], F0_list[k], omega_list[k]] for k in range(len(x0_list))}
    
    t_max = 3000
    h = 1e-1
    t = np.arange(0, t_max, h)

    main(args=conditions_dict, t=t)