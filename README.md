# Image Classification

#### Assignment 4 of CSCI B551 Elements of Artificial Intelligence by Prof. David Crandall<br />

Training and testing data is of the format:-<br />

*photo_id correct_orientation r11 g11 b11 r12 g12 b12 ...<br />

where:
• photo id is a photo ID for the image. <br />
• correct orientation is 0, 90, 180, or 270. Note that some small percentage of these labels may be
wrong because of noise; this is just a fact of life when dealing with data from real-world sources.<br />
• r11 refers to the red pixel value at row 1 column 1, r12 refers to red pixel at row 1 column 2, etc.,
each in the range 0-255<br /><br />


For training, the program should be run like this:<br />
*./orient.py train train_file.txt model_file.txt [model]<br />
where [model] is one of "nearest", "adaboost", "forest", or "best"
<br /><br />
For testing, the program should be run like this:<br />
*./orient.py test test_file.txt model_file.txt [model]<br />
where [model] is again one of nearest, adaboost, forest, or best.

