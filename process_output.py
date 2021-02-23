import matplotlib.pyplot as plt

filenames = ["output/out_ba_25000_3_1000_s.txt"]

# this code averages result for nodes and produces proc_ file that contains LaTeX Tikzpicture-compatible data 
# for visualising friendship index dynamics

if __name__ == "__main__":
    for filename in filenames:
        x_range = range(1000, int(filename.split('_')[2]), 50)
        metric_type = filename.split('.')[-2].split('_')[-1]

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
        for i in range(len(x_range)):
            f_out.write(str(x_range[i]) + "\t" + str(processed_values[i]) + "\n")

        f_out.close()
        f.close()
