import csv
import matplotlib.pyplot as plt
'''
Reference book: Data Mining: Theories, Algorithms and Examples by Nong Ye (Example 2.1)

Fitting a simple linear regression model to the space shuttle O-rings dataset
with the least squares method using the below equation:
                         y = B0 + B1x + e
where y is the target variable representing Number of O-rings with stress
and x is the attribute variable representing Launch Temperature in Farenheit 
e represents random noise

Find estimated values for parameters B0 and B1 
that minimizes the sum of squared errors (SSE) between observed and estimated target values as follows:
B1 = sum((x_i - mean_x)(y_i - mean_y))/sum((x_i - mean_x)^2) for i=1:n
B0 = y_mean - B1*x_mean

NB: The difference in the values of parameters B0 and B1 from that of the book are caused by rounding in the calculation
'''
x = [] # Launch Temperature
y = [] # Number of O-rings with stress

# Read the CSV file
with open('C:\Data Mining\datasets\Space shuttle O-ring erosion.csv', 'r') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)  # Read the header row
    for row in csv_reader:
        #print(row)  # Print each data row
        x.append(int(row[2]))
        y.append(int(row[5]))

# Calculate x_mean (avg_temp) and y_mean (avg_rings)
avg_temp = (sum(x)/len(x))
avg_rings = (sum(y)/len(y))

sum1 = 0
sum2 = 0

for i in range(len(x)):
    sum1 = sum1 + ((x[i] - avg_temp) * (y[i] - avg_rings))
    sum2 = sum2 + ((x[i] - avg_temp) * (x[i] - avg_temp))

B1 = sum1/sum2
B0 = avg_rings - B1 * avg_temp

print("The linear regression model is: ")
print(f"y = {B0:.2f} {B1:.2f}x + e")

# Calculate estimated values of y using B0 and B1
x_est = [i for i in range(53,82)]
y_est = [B0 + (i * B1) for i in x_est] 

plt.scatter(x, y)
plt.plot(x_est, y_est)

# Add labels and title
plt.xlabel('Launch Temperature (F)')
plt.ylabel('O-rings with stress')
plt.title('Space shuttle O-rings')

# Show the plot
plt.show()