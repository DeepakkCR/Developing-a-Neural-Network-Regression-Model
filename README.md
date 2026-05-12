# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY

The objective of this experiment is to design, implement, and evaluate a Deep Learning–based Neural Network regression model to predict a continuous output variable from a given set of input features. The task is to preprocess the data, construct a neural network regression architecture, train the model using backpropagation and gradient descent, and evaluate its performance using appropriate regression metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score.

## Neural Network Model
Include the neural network model diagram.
<img width="1095" height="745" alt="542714748-b2141eb2-ad6c-4f3d-8419-c8126c9732b7" src="https://github.com/user-attachments/assets/a10c57a4-bf02-4043-9a23-10da995e3d1c" />
## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:

### Register Number:

```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1= nn.Linear(1,8)
        self.fc2= nn.Linear(8,10)
        self.fc3= nn.Linear(10,1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}

  def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# Initialize the Model, Loss Function, and Optimizer



def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range (epochs):
      optimizer.zero_grad()
      loss = criterion(ai_brain(X_train),y_train)
      loss.backward()
      optimizer.step()
      ai_brain.history['loss'].append(loss.item())
      if epoch % 200 == 0:
          print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

```

### Dataset Information
<img width="706" height="320" alt="image" src="https://github.com/user-attachments/assets/08492d50-776f-44f8-beba-910bf6fe9792" />


### OUTPUT

### Training Loss Vs Iteration Plot
<img width="868" height="648" alt="image" src="https://github.com/user-attachments/assets/a43cdacb-4c8d-4e42-b4d8-933e4d32b958" />

### New Sample Data Prediction
<img width="944" height="150" alt="image" src="https://github.com/user-attachments/assets/8ab6debd-b0e8-4f41-ba02-84926fdbf964" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
