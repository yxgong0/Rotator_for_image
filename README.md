# imgae_rotation
A method to rotate images and the annotations on them for data augmentation.

This method will rotate the input images and extend the edge to ensure all the pixels of the initial image will not be abandoned.

The main method is rotate(image, points, rects, rotation_angle) in which:
‘image’ is the image needed to be rotated, which should be loaded by cv2.imread() function
‘points’ is a list of some point annotations on the image formatted as tuples (x,y). It can be empty while the output will also be empty
‘rects’ is a list of some rectangles on the image formatted as tuples (x,y,width,height). It can also be empty.
‘rotation_angle’ can be an integer or a tuple with length 2. 
-When it is an integer, the inputs will be rotated rotation_angle degrees
-When it is a tuple with length 2, the rotation angle will be generated randomly with a value between [rotation_angle[0], rotation_angle[1]]

It is worth noting that the method is not suitable for rotations with a large angle for there is accumulative errors while calculating floating numbers.
