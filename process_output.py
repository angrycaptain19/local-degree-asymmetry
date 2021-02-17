import matplotlib.pyplot as plt

filename = "out_ba_100000_b.txt"
x_range = range(50, 100000, 50)

if __name__ == "__main__":
    metric_type = filename.split('.')[0].split('_')[-1]

    f = open(filename)
    lines = f.readlines()
    # Temporary
    processed_values = [0 for x in lines[1].split(' ')]
    data_count = 0
    for line in lines:
        if line.strip() and not (line.startswith(">")):
            data_count += 1
            values = line.split(' ')
            for i in range(len(values)):
                processed_values[i] += float(values[i])
    
    for i in range(len(processed_values)):
        processed_values[i] /= data_count

    f_out = open("proc_" + filename, "w")
    f_out.write("t\t" + metric_type + "(t)\n")
    for i in range(len(processed_values)):
        f_out.write(str(x_range[i]) + "\t" + str(processed_values[i]) + "\n")

    f_out.close()
    f.close()
    plt.plot(x_range, processed_values)
    #plt.show()
    
    #print(len(processed_values))
    #print(processed_values)
        