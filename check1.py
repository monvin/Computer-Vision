import torch
def predict_labels(model_output: torch.tensor) -> torch.tensor:
  '''
  Predicts the labels from the output of the model.

  Args:
  -   model_output: the model output [Dim: (N, 15)]
  Returns:
  -   predicted_labels: the output labels [Dim: (N,)]
  '''
  predicted_labels = None
  ############################################################################
  # Student code begin
  ############################################################################
  predicted_labels = torch.unsqueeze(torch.argmax(model_output[0]),0)
  #predicted_labels = torch.cat((predicted_labels,predicted_labels),0)
  print(model_output.shape)
  print(predicted_labels.shape)
  if(model_output.shape[0]!=1):
    for i in range(1,model_output.shape[0]):
      temp = torch.argmax(model_output[i])
      print(temp)
      print(type(temp))
      temp=torch.unsqueeze(torch.tensor(temp),0)
      print(temp)
      print(type(temp))
      temp=torch.cat((predicted_labels,temp),0)
      print(temp)
      print(type(temp))
    predicted_labels=temp
  assert(predicted_labels.shape[0]==model_output.shape[0])
  ############################################################################
  # Student code end
  ############################################################################
  #print(predicted_labels)
  print(predicted_labels.shape)
  return predicted_labels