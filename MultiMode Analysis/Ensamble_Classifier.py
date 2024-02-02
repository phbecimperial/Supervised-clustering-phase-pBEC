import torch

class Ensamble():
    def __init__(self, model, classes, learning_rate, 
                 optimizer: torch.optim.Optimizer, criterion, device, **Model_arguments) -> None:

        self.model_list = [model(**Model_arguments, num_classes = 2).to(device) for i in range(classes)]
        self.optimizers = [torch.optim.Adam(i.parameters(), lr = learning_rate) for i in self.model_list]
        self.criterions = [criterion for i in enumerate(self.model_list)]
        self.losses = None

    def __call__(self, inputs):
        outputs = []
        for i, _ in enumerate(self.model_list):

            outputs.append(self.model_list[i](inputs))
        
        return torch.stack(outputs)
    
    def to(self, *args):
        for i in self.model_list:
            i.to()


    def calculate_losses(self,outputs,labels):
        if not self.losses:
            losses = []
            for i, _ in enumerate(self.model_list):
                losses.append(self.criterions[i](torch.select(outputs,0,i), torch.select(labels,1,i)))
            self.losses = losses
        
        else:
            for i, _ in enumerate(self.model_list):
                self.losses[i] = self.criterions[i](torch.select(outputs,0,i), torch.select(labels,1,i))

        return self.losses

    def loss_step(self, outputs, inputs):

        self.calculate_losses(outputs,inputs)

        for i, _ in enumerate(zip(self.losses, self.optimizers)):
            self.optimizers[i].zero_grad()
            self.losses[i].backward()
            self.optimizers[i].step()
            del outputs, inputs
        return self.losses
        
    def train(self):
        for i, mod in enumerate(self.model_list):
            self.model_list[i].train()

    def eval(self):
        for i in self.model_list:
            self.model_list[i].eval()

    def pred(self, input, dim = 2):
        preds = []
        for i in enumerate(self.model_list):
            _, pred = torch.max(i(input), dim)
            preds.append(pred)
        
        return torch.stack(preds)