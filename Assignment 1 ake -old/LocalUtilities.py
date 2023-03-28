def Triplotlog(X,Y1,Y2,Y1_Name,Y2_Name):
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots()
    ax1.set_title("Two Curves on Two Axes")

    ax1.set_xlabel("x")
    ax1.set_ylabel(Y1_Name, color="blue")
    ax1.plot(X, Y1, color="blue")
    ax1.set_xscale('log')

    ax2 = ax1.twinx()
    ax2.set_ylabel(Y2_Name, color="red")
    ax2.plot(X, Y2, color="red")
    ax2.set_xscale('log')
    plt.show()
    
def Triplot(X,Y1,Y2,Y1_Name,Y2_Name):
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots()
    ax1.set_title("Two Curves on Two Axes")

    ax1.set_xlabel("x")
    ax1.set_ylabel(Y1_Name, color="blue")
    ax1.plot(X, Y1, color="blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel(Y2_Name, color="red")
    ax2.plot(X, Y2, color="red")
    plt.show()
    