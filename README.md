# ocr++(an update to existing old ocrs)
The code is to convert the image to speech. An image is processed and segmented to identify the characters in the image. Then the characters are combined to form words and save it as a text file. This text file is converted to speech. We have divided the project into four sub parts : image is pre-processed, segmented to extract the images of characters, then characters are recognized and combined , then the text is translated then converted into speech.
# For language translation(english to french) I achieved an accuracy of 94% while 96.3% for ocr character recognition

# Technical terms:

Image to text

•	Convolution 2d

•	Max Pooling

•	Activation Function (Tanh, Relu, Sigmoid, Leaky Relu)

•	Flatten

•	 Dropout

Character Segmentation
•	C# (.exe)

Language Translation 

•	LSTM
•	GRU
•	Bi-directional RNN
•	Embedding layer
•	Encoder and Decoder

Text to Speech

•	GTTs

•	PYgame    

# We use two different datasets:

Language translation: https://machinelearningmastery.com/prepare-french-english-dataset-machine-translation/
Image to text: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/

