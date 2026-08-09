
x_input = [0.1, 0.5, 0.2]
weight = [0.4, 0.3, 0.6]
threshold = 0.5

def activation(weighted_sum):
    if weighted_sum > threshold:
        return 1
    else:
        return 0

def perceptron():
    weighted_sum = 0

    for i in range(len(x_input)):
        weighted_sum += x_input[i] * weight[i]
        print(weighted_sum)

    return activation(weighted_sum)

output = perceptron()
print("Result: "+str(output))