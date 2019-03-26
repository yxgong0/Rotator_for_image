# image_rotation
A method to rotate images and the annotations on them for data augmentation.

Requirements
-------
Python3

OpenCV3.4

Numpy1.15

Details
-------
This method will rotate the input images and extend the edge to ensure all the pixels of the initial image will not be abandoned.
That means the procedure will look like this:![1](https://github.com/Alpaca07/imgae_rotation/blob/master/examples/sketch1.png)

We are not doing this work:![2](https://github.com/Alpaca07/imgae_rotation/blob/master/examples/sketch2.png)

You need to build a rotater for one image with its annotations. The parameters means:

‘image’ is the image needed to be rotated, which should be loaded by cv2.imread() function

‘points’ is a list of some point annotations on the image formatted as tuples (x,y). It can be empty while the output will be None. An example of the parameter is [(113, 127), (156, 127), (156, 170)]

‘rects’ is a list of some rectangles on the image formatted as tuples (x,y,width,height). It can be empty while the output will be None. An example of the parameter is [(117, 188, 108, 57), (118, 189, 109, 58)]

'np_rotated_rects' is some rotated rectangles on the image formated as the type of numpy.array. Each row represents a rectangle and the columns represents x_min, y_min, x_max, y_max and rotation angle respectively. It can be empty while the output will be None.

'cv_rotated_rects' is a list of rotated rectangles formatted as those in opencv. Each element has three subelements, in which the first tuple means the center of the rectangle, the second tuple includes width and height, and the third number is the rotation angle. It can be empty while the output will be None. The structure is ((center_x, center_y), (width, height), angle)

'quadrilaterals' is a list of some quadrilaterals. One quadrilateral is a tuple with 8 numbers, which means the 4 vertexes of it. The first two numbers represents the x and y of the first vertex and the rest may be deduced by analogy. It can be empty while the output will be None.

'polygons' is a list of polygons. Each element contains some tuples which represents the coordinates of the vertixes. It can be empty while the output will be None.

Tips: While using this class, you can transform numpy.ndarray to list with .tolist() or transform list to numpy.ndarray with numpy.array(list).

Then a rotater of an image has been created. You can invoke the method rotate(rotation_angle) to rotate the image and its annotations.

‘rotation_angle’ can be an integer or a tuple with length 2

-When it is an integer, the inputs will be rotated rotation_angle degrees

-When it is a tuple with length 2, the rotation angle will be generated randomly with a value between [rotation_angle[0], rotation_angle[1]]

The method will return a dictionary of all of the data. The key of each data is the same as the name of the parameters of the initializer. For instance, returned_value['image'] is the rotated image, returned_value['points'] is a list of the points after rotation.


It is worth noting that the method is not suitable for rotations with a large angle as there are accumulative errors while calculating floating numbers.
