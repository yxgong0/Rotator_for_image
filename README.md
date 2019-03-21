# image_rotation
A method to rotate images and the annotations on them for data augmentation.

# Requirements
Python3

OpenCV3.4

# Details
This method will rotate the input images and extend the edge to ensure all the pixels of the initial image will not be abandoned.
That means the procedure will look like this:![1](https://github.com/Alpaca07/imgae_rotation/blob/master/examples/sketch1.png)

We are not doing this work:![2](https://github.com/Alpaca07/imgae_rotation/blob/master/examples/sketch2.png)

The main method is rotate(image, points, rects, rotation_angle) in which:

‘image’ is the image needed to be rotated, which should be loaded by cv2.imread() function

‘points’ is a list of some point annotations on the image formatted as tuples (x,y). It can be empty while the output will also be empty

‘rects’ is a list of some rectangles on the image formatted as tuples (x,y,width,height). It can also be empty

‘rotation_angle’ can be an integer or a tuple with length 2

-When it is an integer, the inputs will be rotated rotation_angle degrees

-When it is a tuple with length 2, the rotation angle will be generated randomly with a value between [rotation_angle[0], rotation_angle[1]]


It is worth noting that the method is not suitable for rotations with a large angle as there are accumulative errors while calculating floating numbers.
