import numpy as np

def Generate_Batches(data, num_epochs, batch_size, shuffle=True):
    data = np.array(data)
    data_size = len(data)

    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            # Create a generator and return it to run batch job
            yield shuffled_data[start_index:end_index]