def GD_StepWise_bis(Theta_, alpha_, x_, y_, IterMax_, StepLim_):
    Iterractions = 0
    tic = time.process_time()
    for i in range(IterMax_):
        Theta_old_ = Theta_
        
        temp = gd_iterration(Theta_,alpha_,x_,y_)
        index = np.argmax(temp)
        
        Theta_[index] = temp[index]
        if abs(Theta_[index]-Theta_old_[index]) < StepLim_:
            Iterractions = Iterractions+i
            break
    toc = time.process_time() -tic
    return np.append(Theta_,[Iterractions,alpha_,toc])



def gd_iterration(Theta_,alpha_,x_,y_):
        return Theta_ - alpha_ * cost_func_grad(Theta_,x_,y_)
    
    
    
def GD_StepWise(Theta_, alpha_, x_, y_, IterMax_, StepLim_):
    Iterractions = IterMax_
    tic = time.process_time()
    for k in range(len(Theta_)):
        for i in range(IterMax_):
            Theta_old_ = Theta_
            Theta_[k] = gd_iterration(Theta_,alpha_,x_,y_)[k]
            if cost_func(Theta_,x_,y_)>cost_func(Theta_old_,x_,y_):
                Iterractions = Iterractions+i+1
                break
    toc = time.process_time() -tic
    return np.append(Theta_,[Iterractions,alpha_,toc])

def GD(Theta_, alpha_, x_, y_, IterMax_, StepLim_):
    Iterractions = IterMax_
    tic = time.process_time()
    for i in range(IterMax_):
        Theta_old_ = Theta_
        Theta_ = gd_iterration(Theta_,alpha_,x_,y_)
        if np.abs(cost_func(Theta_,x_,y_)-cost_func(Theta_old_,x_,y_)) < StepLim_:
            Iterractions = i
            break
    toc = time.process_time() -tic
    return np.append(Theta_,[Iterractions,alpha_,toc])

def GD_Work_Alpha(alpha_):
    return GD(np.array([0,0,0,0,0,0]), alpha_, shared_x, shared_y, shared_IterMax, shared_StepLim)

def init_worker_alpha(x_,y_,IterMax_,StepLim_):
    global shared_x
    global shared_y
    global shared_IterMax
    global shared_StepLim
    
    shared_x = x_
    shared_y = y_
    shared_IterMax = IterMax_
    shared_StepLim = StepLim_