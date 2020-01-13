import numpy as np
import helper

# d0dcbcd6cd9243a38882210c6355078f.jpg

# TODO:
#   - Convert image to a square
#       - Add whitespace (zeros) instead of squishing, to give the model hints about which is the long side
#   - Resize image to some fixed square value
#   - Train to identify pieces of paper from open books
#       - Don't do anything with open books, yet. Those will need additional processing
#   - Train to identify rotation of found pieces of paper

# TODO (maybe):
#   - Concatenate all four rotations of image into one square image

a = np.array([
    [1,2,3],
    [4,5,6],
])
b = np.array([
    [1,2],
    [3,4],
    [5,6]
])
c = np.array([
    [1,2],
    [3,4]
])

print(helper.pad2DArrayWithZeros(a))
print(helper.pad2DArrayWithZeros(b))
print(helper.pad2DArrayWithZeros(c))
