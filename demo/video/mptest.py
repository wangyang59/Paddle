import multiprocessing as mp
import random
import string, time

random.seed(123)

# Define an output queue
output = mp.Queue(10)


# define a example function
def rand_string(length, output, x):
    """ Generates a random string of numbers, lower- and uppercase chars. """
    rand_str = ''.join(
        random.choice(string.ascii_lowercase + string.ascii_uppercase +
                      string.digits) for i in range(length))

    time.sleep(2)
    output.put(rand_str)


# Setup a list of processes that we want to run
processes = [
    mp.Process(
        target=rand_string, args=(5, output, x)) for x in range(4)
]

# Run processes
for p in processes:
    p.start()


def provider(output):
    for x in range(4):
        yield output.get()


# Get process results from the output queue
generator = provider(output)

for item in generator:
    time.sleep(3)
    print(item)

# Exit the completed processes
for p in processes:
    p.join()
