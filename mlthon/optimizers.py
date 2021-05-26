def GDT(derivative, objective, max_iter, learning_rate=0.01, momentum=0.0, nesterov=False):
    change_x, x_evals = [], []

    change = 0.0
    x = 1.0

    for i in range(max_iter):
        if nesterov == True:
            new_change = learning_rate*derivative(x + momentum * change) + momentum * change
        
        else:
            new_change = learning_rate*derivative(x) + momentum * change

        x = x - new_change

        change = new_change
        x_eval = objective(x)
        change_x.append(x)
        x_evals.append(x_eval)
        print(f'>{i} f({x}) = {x_eval}')

    return x, change_x, x_evals