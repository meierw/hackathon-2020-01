import numpy as np

# d0dcbcd6cd9243a38882210c6355078f.jpg
# TODO: 
#   - Concatenate all four rotations of image into one square image
#   - Equalize image sizes
#   - Train to identify pieces of paper from open books
#       - Don't do anything with open books, yet. Those will need additional processing
#   - Train to identify rotation of found pieces of paper
a = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9],
])
print(a)
