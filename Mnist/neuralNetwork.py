import numpy as np
import scipy.special
import matplotlib.pyplot as plt


# 神经网络编程基础训练
class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 网络架构
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        # 权重矩阵
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        # 权重矩阵
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        # 激活函数
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        self.who += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)),
                                     np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), np.transpose(inputs))

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


if __name__ == '__main__':
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    learning_rate = 0.1
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    epochs = 3
    print("开始训练,世代为: " + str(epochs))
    #  优化权重
    for e in range(epochs):
        for record in training_data_list:
            all_values = record.split(',')
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)
    print("训练完成")

    # 测试数据集
    test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    all_values = test_data_list[0].split(',')
    img = np.asfarray(all_values[1:]).reshape((28, 28))
    plt.imshow(img, cmap='Greys', interpolation='None')
    plt.show()
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    print(outputs)

    # # test the neural network
    #
    # # scorecard for how well the network performs, initially empty
    # scorecard = []
    #
    # # go through all the records in the test data set
    # for record in test_data_list:
    #     # split the record by the ',' commas
    #     all_values = record.split(',')
    #     # correct answer is first value
    #     correct_label = int(all_values[0])
    #     # scale and shift the inputs
    #     inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    #     # query the network
    #     outputs = n.query(inputs)
    #     # the index of the highest value corresponds to the label
    #     label = np.argmax(outputs)
    #     # append correct or incorrect to list
    #     if (label == correct_label):
    #         # network's answer matches correct answer, add 1 to scorecard
    #         scorecard.append(1)
    #     else:
    #         # network's answer doesn't match correct answer, add 0 to scorecard
    #         scorecard.append(0)
    #
    # scorecard_array = np.asarray(scorecard)
    # print("performance = ", scorecard_array.sum() / scorecard_array.size)
