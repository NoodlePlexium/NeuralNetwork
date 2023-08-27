import numpy as np
import random

class NeuralNetwork:
    def __init__(self, structure) -> None:
        self.structure=structure
        self.layers=[]
        self.weights={}

        self.GenerateNetwork()

    def GenerateNetwork(self):
        # Generate Layers
        for i in range(len(self.structure)):
            neurons=[]
            biases=[]
            for j in range(self.structure[i]):
                neurons.append(0)
                biases.append(round(random.uniform(-10,10),2))  
            self.layers.append([neurons,biases])
            self.layerCount=len(self.structure)       

        # Generate Weight Datastructure
        self.weightCount=0
        for layer in range(1,len(self.structure)):
            for outNeuron in range(self.structure[layer]):
                for inNeuron in range(self.structure[layer-1]):
                    self.weights[str(layer)+"_"+str(inNeuron)+"_"+str(outNeuron)]=round(random.uniform(-10,10),2)
                    self.weightCount+=1 

    def FeedForward(self):
        for i in range(1, self.layerCount):
            for outNeuron in range(self.layerSize(i)):
                sum=0
                for inNeuron in range(self.layerSize(i-1)):
                    sum+=self.getWeight(i,inNeuron,outNeuron)*self.neurons(i-1)[inNeuron]       
                self.neurons(i)[outNeuron]=self.sigmoid(sum+self.biases(i)[outNeuron])  

    def BackPropagate(self, batch, learnRate):

        weightChanges={}
        biasChanges=[]

        for i in range(self.layerCount):
            biasChanges.append([0]*self.layerSize(i)) 

        for i in range(1, self.layerCount):  
            for outNeuron in range(self.layerSize(i)):
                for inNeuron in range(self.layerSize(i-1)):
                    weightChanges[str(i)+"_"+str(inNeuron)+"_"+str(outNeuron)]=0 

        correct=0                             

        for trainingExample in range(batch['length']):

            self.setInputs(batch['input'][trainingExample])
            self.FeedForward()

            ansIndex=np.argmax(batch['input'][trainingExample])
            predictedIndex=np.argmax(self.neurons(self.layerCount-1))
            if predictedIndex==ansIndex:
                correct+=1

            nodeValues=[]
            for i in range(self.layerCount-1,0,-1):
                # End Layer
                if i==self.layerCount-1:
                    for outNeuron in range(self.layerSize(i)):
                        a = self.neurons(i)[outNeuron]
                        nodeValues.append(self.NeuronCostDerivative(a,batch['answer'][trainingExample][outNeuron])*self.SigmoidDerivative(a))

                        # Calculate Weight Change 
                        for inNeuron in range(self.layerSize(i-1)):
                            newValue = learnRate*self.neurons(i-1)[inNeuron]*nodeValues[outNeuron]
                            change=(newValue-self.getWeight(i,inNeuron,outNeuron))/batch['length'] 
                            weightChanges[str(i)+"_"+str(inNeuron)+"_"+str(outNeuron)]-=change

                        # Calculate Bias Change
                        newBias=learnRate*nodeValues[outNeuron]
                        change=(newBias-self.biases(i)[outNeuron])/batch['length'] 
                        biasChanges[i][outNeuron]-=change

                # Every Other Layer            
                else:
                    newNodeValues=[]
                    for outNeuron in range(self.layerSize(i)):
                        outWeightsTotal=0
                        weightedInput=0

                        # Calculate New Node Values
                        for nextNeuron in range(self.layerSize(i+1)):
                            outWeightsTotal+=self.getWeight(i+1,outNeuron,nextNeuron)*nodeValues[nextNeuron]
                        for inNeuron in range(self.layerSize((i-1))):
                            weightedInput+=self.getWeight(i,inNeuron,outNeuron)
                        newNodeValues.append(outWeightsTotal*self.SigmoidDerivative(weightedInput)) 

                        # Calculate Bias Change
                        newBias=learnRate*newNodeValues[outNeuron]
                        change=(newBias-self.biases(i)[outNeuron])/batch['length'] 
                        biasChanges[i][outNeuron]-=change

                        # Calculate Weight Change 
                        for inNeuron in range(self.layerSize((i-1))):
                            newValue = learnRate*(newNodeValues[outNeuron]*self.neurons(i-1)[inNeuron])
                            change = (newValue-self.getWeight(i,inNeuron,outNeuron))/batch['length'] 
                            weightChanges[str(i)+"_"+str(inNeuron)+"_"+str(outNeuron)]-=change
                    nodeValues=newNodeValues
                    
        # Update all Weights and Biases
        self.weights=weightChanges
        self.updateBiases(biasChanges) 
        return correct/batch['length']   

    def sigmoid(self,x):
        x = np.clip(x, -500, 500 )    
        return 1/(1 + np.exp(-x))
    
    def SigmoidDerivative(self,x):
        activation = self.sigmoid(x)
        return activation * (1-activation) 

    def NeuronCost(self,predicted, expected):
        error=predicted-expected
        return error*error

    def NeuronCostDerivative(self,weightedInput, expected):
        return 2 * (weightedInput-expected)
      

    def getWeight(self, outLayer, inNeuron, outNeuron):        
        return self.weights[str(outLayer)+"_"+str(inNeuron)+"_"+str(outNeuron)]  
    
    def setWeight(self, outLayer, inNeuron, outNeuron, value):        
        self.weights[str(outLayer)+"_"+str(inNeuron)+"_"+str(outNeuron)] = value  

    def setInputs(self, inputArray):
        self.layers[0][0]=inputArray    

    def neurons(self,layer):
        return self.layers[layer][0] 

    def biases(self,layer):
        return self.layers[layer][1]  
    
    def updateBiases(self,newValues):
        for layer in range(self.layerCount):
            self.layers[layer][1]=newValues[layer]
    
    def layerSize(self,layer):
        return self.structure[layer]
    
    def printNeurons(self,layer):
        print("\n")
        print(self.layers[layer][0])

    def printPrediction(self):
        output=self.layers[self.layerCount-1][0]
        predictionIndex=np.argmax(output)
        print(f"Prediction: {predictionIndex+1} {round(output[predictionIndex],2)}% Certain")
  



# Create a training set
Batch={'input':[],'answer':[],'length':0}
for i in range(1000):

    nums=[]
    for j in range(2):
        nums.append(random.uniform(0,100))

    trainingPoint=nums
    Batch['input'].append(trainingPoint)  
    Batch['length']+=1

    ansIndex=np.argmax(nums)   
    ans=[0]*2
    ans[ansIndex]=1
    Batch['answer'].append(ans)  

network = NeuralNetwork([2,8,6,4,2])

# Train the network
for iteration in range(10):
    print(network.BackPropagate(Batch,0.5))

while True:
    print("\n")
    nump1=float(input("Type number 1: "))
    nump2=float(input("Type number 2: "))
    network.setInputs([nump1,nump2])
    network.FeedForward()
    network.printPrediction()
