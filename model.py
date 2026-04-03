import torch 
import torch.nn as nn
import torch.nn.functional as F

class NoteClassifierCNN(nn.Module):

    def __init__(self, n_mels=64, n_frames=345, n_classes=12):
        super(NoteClassifierCNN, self).__init__()

        ##CONV block 1
        self.conv1=nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.bn1= nn.BatchNorm2d(32)
        self.pool1= nn.MaxPool2d(kernel_size=2,stride=2)

        #Conv Block 2
        self.conv2= nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        #parameter count= 64x(32x3x3+1)=18,496

        self.bn2= nn.BatchNorm2d(64)
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=2)

        #compute flattened size
        #
        self._flat_size= 64*(n_mels//4)*(n_frames//4)
        print(f"Flattened size after conv layers: {self._flat_size:,}")

        #fully connected layers

        self.dropout= nn.Dropout(p=0.5)

        self.fc1=nn.Linear(self._flat_size, 128)
        #collapses the entire spatial feature volume into 128 summary values.

        self.fc2= nn.Linear(128,n_classes)
        #maps 128 summary values to n_classes raw scores(logits).


    def forward(self, x):

        #block 1

        x= self.conv1(x) 
        x= self.bn1(x)
        x= F.relu(x)
        x= self.pool1(x)

        #block 2
        x = self.conv2(x)         
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)       

        #Flatten
        x= x.view(x.size(0),-1)

        #Dense
        x=self.dropout(x)
        x= self.fc1(x)
        x= F.relu(x)
        x= self.fc2(x)

        return x
    
    def predict(self,x):
        self.eval()
        with torch.no_grad():
            logits= self.forward(x)
            probs= F.softmax(logits, dim=1)
            predicted_class=torch.argmax(probs, dim=1)
        return probs, predicted_class
    

#print shape at every layer
def print_model_summary(model, n_mels=64, n_frames=345):
    print("=" * 52)
    print("Layer-by-layer shape trace")
    print("=" * 52)
 
    x = torch.zeros(1, 1, n_mels, n_frames)
    print(f"Input:              {tuple(x.shape)}")
 
    x = model.conv1(x);  print(f"After conv1:        {tuple(x.shape)}")
    x = model.bn1(x);    x = F.relu(x)
    x = model.pool1(x);  print(f"After pool1:        {tuple(x.shape)}")
 
    x = model.conv2(x);  print(f"After conv2:        {tuple(x.shape)}")
    x = model.bn2(x);    x = F.relu(x)
    x = model.pool2(x);  print(f"After pool2:        {tuple(x.shape)}")
 
    x = x.view(x.size(0), -1)
    print(f"After flatten:      {tuple(x.shape)}")
 
    x = model.fc1(x);    x = F.relu(x)
    print(f"After fc1:          {tuple(x.shape)}")
 
    x = model.fc2(x)
    print(f"After fc2 (logits): {tuple(x.shape)}")
 
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 52)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")

if __name__=="__main__":
    
    model = NoteClassifierCNN(n_mels=64, n_frames=345, n_classes=12)
    print_model_summary(model,n_mels=64,n_frames=345)

    print("\nOutput of this module:")
    