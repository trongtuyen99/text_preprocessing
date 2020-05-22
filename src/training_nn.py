import torch
from torch.autograd import Variable
from torchsummary import summary
from time import time

model = Softmax(24888, 20)
if torch.cuda.is_available():
    model.cuda()

criterion = torch.nn.CrossEntropyLoss()
lr = 0.005
optimizer = torch.optim.Adam(model.parameters(), lr)

epochs = 31
from time import time
t1 = time()
for epoch in range(epochs):
  if torch.cuda.is_available():
    X = Variable(torch.from_numpy(X_train_tfidf.toarray()).float().cuda())
    Y = Variable(torch.from_numpy(Y_train_transform).cuda())
  else:
    X = Variable(torch.from_numpy(X_train_tfidf.toarray()).float())
    Y = Variable(torch.from_numpy(Y_train_transform))
  optimizer.zero_grad()

  output = model(X)
  loss = criterion(output, Y)
  loss.backward()
  optimizer.step()
  if epoch % 2 == 0:
    print(f"loss after {epoch} epochs: {loss}")
    pred1 = torch.argmax(model(torch.from_numpy(X_train_tfidf.toarray()).float().cuda()), axis=1)
    s1 = sum(pred1 == torch.from_numpy(Y_train_transform).cuda())
    print(f"true train: {s1}")

    pred2 = torch.argmax(model(torch.from_numpy(X_test_tfidf.toarray()).float().cuda()), axis=1)
    s2 = sum(pred2 == torch.from_numpy(Y_test_transform).cuda())
    print(f"true test: {s2}")

print(f"training {epochs} epochs done after {time()-t1} s")
