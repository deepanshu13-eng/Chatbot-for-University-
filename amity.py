import nltk                                                     #Importing the nltk file. This is a suite of libraries and programs for symbolic and statistical natural language processing (NLP) for English written in the Python programming language. 
from nltk.stem.lancaster import LancasterStemmer                #Importing some more libraries from nltk to do stemming in the program.
stemmer = LancasterStemmer()                                    # We are making (stemmer) equal to LancasterStemmer(). If we will do this then we dont have to write LancasterStemmer() every time in the program and we can just write stemmer instead of LancasterStemmer(). 
import numpy                                                    # Now we will Import numpy to do stuff's related to array's. It also has functions for working in domain of linear algebra, fourier transform, and matrices.   
import tflearn                                                  # Now we will Import tflearn. TFlearn is a modular and transparent deep learning library built on top of Tensorflow. It was designed to provide a higher-level API to TensorFlow in order to facilitate and speed-up experimentations, while remaining fully transparent and compatible with it.
import tensorflow                                               # Now we will import TensorFlow. TensorFlow is a Python library for fast numerical computing created and released by Google. TensorFlow is an end-to-end open source platform for machine learning. It is a foundation library that can be used to create Deep Learning models directly or by using wrapper libraries that simplify the process built on top of TensorFlow. 
import random                                                   # Now we will import random library. This module contains a number of functions that use random numbers. It can output random numbers, select a random item from a list, and reorder lists randomly. The randomly reordered lists can be output inline, or as various types of ordered and unordered lists.
import json                                                     # Now we will import json library. The json library can parse JSON from strings or files. The library parses JSON into a Python dictionary or list. It can also convert Python dictionaries or lists into JSON strings.
import pickle                                                   # Now ww will import pickle library. Python pickle module is used for serializing and de-serializing python object structures. The process to converts any kind of python objects (list, dict, etc.) into byte streams (0s and 1s) is called pickling or serialization or flattening or marshalling.
import time                                                     # Now we will import time library. We will use this library to give some delay time, if required in the program.
import pyttsx3                                                  # Now we will import pyttsx3 library. This library is used to convert text data into speech. We will use this library to make our chat bot to speak in front of the user.
import tkinter as  tk                                           # Now we will import tkinter library. This library is very usefull for making Graphical User Interface.
from PIL import Image, ImageTk                                  # Now we will import PIL library. We will use this library to work with any kind of Image stuff's.                



engine = pyttsx3.init('sapi5')                                  # Now we will set up voice settings for our chatbot. In these lines of code we are telling what should be the speech type, what should be the speed to read a text, etc..           
voices = engine.getProperty('voices')
rate = engine.getProperty('rate')
engine.setProperty('rate', rate-20)
engine.setProperty('voice',voices[0].id)


def speak(audio):                                               # Here we are defining a speak function which will help us when we want our chat bot to speak something. We will directly apply this speak function where we want our chatbot to speak something.
	engine.say(audio)
	engine.runAndWait()

with open("project.json") as file:                              # Here we are importing our project.json file which contain's all the data about all the question's, its answer's which our chatbot needs to speak. 
	data=json.load(file)                                        # Here we are loading that file and then making it equal to data. 
print(data["intents"])                                          # Here we are printing all the intents.

try:
	with open("data2.pickle","rb") as f:                        # When we use to train our data by using all the data in project.json file then we use to store all the data in a pickle file called data2.
		words,labels,training,output = pickle.load(f)

except:
	words = []                                                   # Creating a list called words. 
	labels = []                                                  # Creating a list called labels.
	docs_x= []                                                   # Creating a list called docs_x.
	docs_y = []                                                  # Creating a list called docs_y.
	for intent in data["intents"]:
		for pattern in intent["patterns"]:
			wrds= nltk.word_tokenize(pattern)
			words.extend(wrds)
			docs_x.append(wrds)
			docs_y.append(intent["tag"])
		if intent["tag"] not in labels:
			labels.append(intent["tag"])
	words = [stemmer.stem(w.lower()) for w in words if w !="?"]
	words = sorted(list(set(words)))
	labels = sorted(labels) 
	training = []
	output = []
	out_empty = [0 for _ in range(len(labels))]
	for x,doc in enumerate(docs_x):
		bag = []
		wrds = [stemmer.stem(w) for w in doc]
		for w in words:
			if w in wrds:
				bag.append(1)
			else:
				bag.append(0)
		output_row = out_empty[:]
		output_row[labels.index(docs_y[x])] = 1
		training.append(bag)
		output.append(output_row)
	training = numpy.array(training)
	output = numpy.array(output)

	with open("data2.pickle","wb") as f:
		pickle.dump((words,labels,training,output),f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net= tflearn.fully_connected(net,8)
net= tflearn.fully_connected(net,8)
net= tflearn.fully_connected(net,len(output[0]), activation="softmax")
net= tflearn.regression(net)
model = tflearn.DNN(net)



model.fit(training,output,n_epoch=1000,batch_size=8,show_metric=True) 
model.save("model2.tflearn")

time.sleep(2)

def bag_of_words(s, words):
	bag = [0 for _ in range(len(words))]

	s_words= nltk.word_tokenize(s)
	s_words= [stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i,w in enumerate(words):
			if w== se:
				bag[i] = 1
	return numpy.array(bag)


time.sleep(2)                                                                              # We have given a sleep time because we want our program to run smoothly without any errors. 


root = tk.Tk()                                                                             # Now this (root) will help us to run the gui.
take = tk.StringVar()                                                                      # We have declared take as a StringVar in Tkinter.

def chat():                                                                                # Here we have defined a chat function, that will take the queary and will check for the answer of that question, by applying some algorithms.

	print("start talking with the bot (type quit to stop)!")                                
	
	

	while True:
		inp = take.get()                                                                    # Here the input queary will come in while loop
		print(inp)                                                                          
		#inp= input("you:")
		if inp.lower() == "q":                                                              # if the user will enter (q) then the program will end up and will get closed.
			speak("Good bye. I had a great time with you. Please do call me again to serve you.")
			break
		results = model.predict([bag_of_words(inp, words)])                                 # Here is some of the algoritms that will chech the correct answer for the queary.
		results_index = numpy.argmax(results)
		tag = labels[results_index]

		
		for tg in data["intents"]:
			if tg['tag'] == tag:
				responses = tg['responses']
		print(random.choice(responses))                                                     # Here we are printing the correct answer that our chatbot has got for a particular queary.
		time.sleep(3)
		say = random.choice(responses)                                                       
		speak(say)                                                                          # Here we are making the chatbot to speak the answer.
		inp = ''                                                                            # Now we are removing the input queary the that the user have entered in the starting. 
		break 


		
if __name__ == '__main__':

	


	root.title("AMITY CHAT BOT")                                                      # Creating a Graphical User Interface and giving it a title (Amity Chat Bot)  
	root.geometry("2000x2000")                                                        # Specifying the geometry of the Graphical User Interface 
	bg = tk.PhotoImage(file = "C:\\Users\\Madhav\\Desktop\\amity.png")                # Opening up a photo in our Graphical User Interface from the file location. 
	label1 = tk.Label( root, image = bg)                                              # Creating a labal and putting that pic in that labal which we have fetched from the file.
	label1.place(x = 180, y = 156)                                                    # Setting up location of the pic in our Graphical User Interface.
	root.configure(bg = 'yellow')                                                     # Setting up background colour of the Graphical User Interface.
	tk.Label(root)                                                                    
	
	tk.Label(root, text = "                                                                                                        Amy chatbot",font = ('Ink free',20,'bold'),fg = "red",bg = "yellow").grid( row= 1,column = 0, sticky ='w')                       # Creating heading of the Graphical User Interface.
	tk.Label(root, text = "                                                                                                 Amity University Gurgaon",font = ('Ink free',20,'bold'),fg = "red",bg = "yellow").grid( row= 2,column = 0, sticky ='w')                 # Creating the sub heading of the GUI.
	tk.Label(root, text = "*************************************************************************************",font = ('Ink free',20,'bold'),fg = "yellow",bg = "yellow").grid( row= 3,column = 0, sticky ='w')                                                  # We want to give some space between the heading's and the queary taking parts, so thats why I have printed some stars which will help us to provide a gap. There is no other option other than this if we are using Tkinter library. 
	tk.Label(root , text =  "                                                                                              Sir please enter your query : ",font = ('Ink free',20,'bold'),fg = "red",bg = "yellow", ).grid( row= 4,column = 0,sticky ='w' )          # Before the queary box, there will be a text written that will let the user know that you have to enter the queary here.
	name= tk.Entry(root,textvariable = take, font=('calibre',20,'normal'), width = 30).grid( row= 4,column = 1,sticky ='w' )                                                                                                                                        # With the help of this program line a queary box will appear that will let the user to input its queary. 
	sub_btn=tk.Button(root,text = 'chat', command = chat, font=('calibre',20,'normal')).grid( row= 5,column = 1,sticky ='w' )                                                                                                                                       # After the user will enter his/ her queary then a chat button will be there, he or she have to click the chat button, so that the chatbot will answer that queary of the user.

root.mainloop()   
		


