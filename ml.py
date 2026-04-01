import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset (handwritten digits 0-9)
(X_train,y_train),(X_test,y_test)=mnist.load_data()
# Reduce dataset size to 10,000 images for faster processing
X_train=X_train[:10000]
y_train=y_train[:10000]

# Flatten 28x28 images into a 1D array of 784 pixels and normalize to range [0, 1]
X_train=X_train.reshape(-1,784)/255.0
X_test=X_test.reshape(-1,784)/255.0
# Initialize StandardScaler to normalize features to mean=0 and variance=1
scaler=StandardScaler()
# Fit the scaler on training data and transform it
X_train_scaled=scaler.fit_transform(X_train)
# Transform test data using the same training parameters
X_test_scaled=scaler.transform(X_test)

# Define a dictionary of traditional Machine Learning models to compare
models={"Logistic Regression":LogisticRegression(max_iter=500),"Random Forest":RandomForestClassifier(n_estimators=50),"KNN":KNeighborsClassifier()}
# Dictionary to store accuracy results for final comparison
results={}
print("\n===== MACHINE LEARNING MODELS =====")
# Loop through each model to train and evaluate
for name,model in models.items():
    # Train the current model
    model.fit(X_train_scaled,y_train)
    # Predict the classes for the test set
    y_pred=model.predict(X_test_scaled)
    # Calculate the accuracy percentage
    acc=accuracy_score(y_test,y_pred)
    results[name]=acc
    print(f"\n{name}")
    print("Accuracy:",acc)
    # Print precision, recall, and f1-score for each digit
    print("Classification Report:\n",classification_report(y_test,y_pred))
    # Create and display a confusion matrix heatmap
    cm=confusion_matrix(y_test,y_pred)
    disp=ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title(name)
    plt.show()

print("\n===== NEURAL NETWORK =====")
# Convert integer labels to one-hot encoded vectors (e.g., 3 becomes [0,0,0,1,0,0,0,0,0,0])
y_train_nn=to_categorical(y_train)
y_test_nn=to_categorical(y_test)
# Initialize a Sequential Neural Network model
model=Sequential()
# Add input layer with 128 neurons and ReLU activation
model.add(Dense(128,activation='relu',input_shape=(784,)))
# Add a second hidden layer with 64 neurons
model.add(Dense(64,activation='relu'))
# Add output layer with 10 neurons and Softmax to get probabilities for digits 0-9
model.add(Dense(10,activation='softmax'))
# Compile the model with Adam optimizer and Cross-Entropy loss
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Train the Neural Network for 10 epochs
history=model.fit(X_train,y_train_nn,epochs=10,batch_size=32,validation_split=0.1)
# Evaluate final performance on the test set
loss,acc=model.evaluate(X_test,y_test_nn)
print("Neural Network Accuracy:",acc)
# Get probability predictions for test images
y_pred_nn=model.predict(X_test)
# Convert probabilities back to the digit with the highest score
y_pred_nn=np.argmax(y_pred_nn,axis=1)
print("Classification Report:\n",classification_report(y_test,y_pred_nn))
# Display confusion matrix for the Neural Network
cm=confusion_matrix(y_test,y_pred_nn)
disp=ConfusionMatrixDisplay(cm)
disp.plot()
plt.title("Neural Network Confusion Matrix")
plt.show()

# Plot the training and validation accuracy over time
plt.plot(history.history['accuracy'],label='Train Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
plt.legend()
plt.title("Accuracy Graph")
plt.show()
# Plot the training and validation loss over time
plt.plot(history.history['loss'],label='Train Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.legend()
plt.title("Loss Graph")
plt.show()
# Store NN result and create a bar chart comparing all models
results["Neural Network"]=acc
plt.bar(results.keys(),results.values())
plt.title("Model Comparison")
plt.xticks(rotation=45)
plt.show()

# Display the first 5 test images and their predicted labels
plt.figure(figsize=(10,5))
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(X_test[i].reshape(28,28),cmap='gray')
    plt.title(f"Pred: {y_pred_nn[i]}")
    plt.axis('off')
plt.show()

# Define function to process user-drawn images to match MNIST format
def preprocess_drawn_digit(img):
    # Threshold the image to make it binary black and white
    _,binary=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    # Find the outlines of the drawn digit
    contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # If canvas is empty, return a blank 28x28 image
    if not contours:
        return np.zeros((28,28),dtype=np.float32)
    # Isolate the largest drawn shape
    largest_contour=max(contours,key=cv2.contourArea)
    x,y,w,h=cv2.boundingRect(largest_contour)
    digit=binary[y:y+h,x:x+w]
    # Calculate padding to make the digit square
    size=max(w,h)
    pad_x=(size-w)//2
    pad_y=(size-h)//2
    padded=cv2.copyMakeBorder(digit,pad_y,size-h-pad_y,pad_x,size-w-pad_x,cv2.BORDER_CONSTANT,value=0)
    # Resize to 20x20 and place inside a 28x28 canvas (MNIST style)
    resized=cv2.resize(padded,(20,20))
    canvas_28=np.zeros((28,28),dtype=np.float32)
    canvas_28[4:24,4:24]=resized/255.0
    return canvas_28

print("\n===== DRAW DIGIT =====")
# Create a 400x400 black canvas for the user to draw on
canvas=np.zeros((400,400),dtype=np.uint8)
drawing=False
# Function to handle mouse drawing logic
def draw(event,x,y,flags,param):
    global drawing
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Draw a white circle with radius 5 at the mouse position
            cv2.circle(canvas,(x,y),5,255,-1)
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False

# Setup OpenCV window and link the mouse callback function
cv2.namedWindow("Draw Digit")
cv2.setMouseCallback("Draw Digit",draw)
# Keep the drawing window open until user quits
while True:
    cv2.imshow("Draw Digit",canvas)
    key=cv2.waitKey(1)&0xFF
    # If 'p' is pressed, predict the drawn digit
    if key==ord('p'):
        preprocessed=preprocess_drawn_digit(canvas)
        img_flat=preprocessed.flatten().reshape(1,784)
        prediction=model.predict(img_flat)
        digit=np.argmax(prediction)
        print("Predicted Digit:",digit)
        # Show the 28x28 image the model actually analyzed
        plt.imshow(preprocessed,cmap='gray')
        plt.title(f"Predicted: {digit}")
        plt.show()
    # If 'c' is pressed, clear the canvas
    elif key==ord('c'):
        canvas=np.zeros((400,400),dtype=np.uint8)
    # If 'q' is pressed, close the application
    elif key==ord('q'):
        break
cv2.destroyAllWindows()