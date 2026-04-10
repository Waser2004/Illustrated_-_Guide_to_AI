def main(x: list[float], y:list[float], learning_rate = 0.1, loss="mse"):
    expected_weight = 2
    expected_bias   = 1

    weight = 1
    bias   = 0

    iteration_counter = 0
    while (abs(weight - expected_weight) > 0.001 and abs(bias - expected_bias) > 0.001 and iteration_counter < 1000):
        input_value = x[iteration_counter % len(x)]
        label       = y[iteration_counter % len(y)]

        prediction = input_value * weight + bias

        if loss == "mse":
            weight -= input_value * 2 * (prediction - label) * learning_rate
            bias   -= 2 * (prediction - label) * learning_rate
        elif loss == "absl":
            weight -= input_value * (1 if prediction > label else -1) * learning_rate
            bias   -= (1 if prediction > label else -1) * learning_rate

        iteration_counter += 1
    
    print(f"Final weight: {weight:.4f}, Final bias: {bias:.4f}")
    print(f"Expected weight: {expected_weight}, Expected bias: {expected_bias}")
    print(f"Total iterations: {iteration_counter}")


if __name__ == "__main__":
    x = [1, 2, 3, 4]
    y = [3, 5, 7, 9]

    # optimal learning rate for the least amount of iterations with mse seems to be 0.07 with 184 iterations
    # with absl the model does not converge to the optimal solution, but gets close.
    main(x, y, learning_rate=0.07, loss="mse")