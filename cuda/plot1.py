import matplotlib.pyplot as plt
import sys
import os

def main(args):
    #path = os.path.join('D:/famaf/tesis/mediciones/', args[1])
	# temp_mx_my_mz_mo05hp002f050-00.dat
    path = args[1]
    x = int(args[2])*3
    y  = int(args[3])*3
    print(str(x) + " " + str(type(x)))
    print(str(y) + " " + str(type(y)))
    print(path)
    print(os.path.isfile(path))
    print(os.path.exists(path))
    f = open(path)
    ones = []
    fourths = []
    f.readline()
    for line in f:
        #print(line.split(' '))
        ones.append(float(line.split(' ')[x]))
        #print(str(line.split(' ')[y]) + ".")
        #sys.stdout.flush()
        fourths.append(float(line.split(' ')[y]))
    #print(ones)
    #print(fourths)

    plt.plot(ones, fourths,'r')
    plt.grid()
    plt.yscale('linear')
    plt.xscale('linear')
    plt.xlabel('T')
    plt.ylabel('M')
    #plt.title('Rabbits vs Foxes')
    #plt.savefig('RvF.png')
    plt.show()

if __name__ == "__main__":
   main(sys.argv)
