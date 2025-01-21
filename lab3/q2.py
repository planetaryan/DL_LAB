import torch

x=torch.tensor([2.0,4.0])
y=torch.tensor([20.0,40.0])

w=torch.tensor([1.0], requires_grad=True)
b=torch.tensor([1.0], requires_grad=True)

learning_rate=torch.tensor(0.001)

for epoch in range(2):
    loss=0.0

    for j in range(len(x)):
        y_p=w*x[j]+b
        loss+=(y_p-y[j])**2

    loss/=len(x)

    print(f'Loss in Epoch {epoch} is {loss}')

    loss.backward()

    with torch.no_grad():

        print(f'Epoch {epoch}: w.grad = {w.grad} and b.grad = {b.grad}')
        w-=learning_rate*w.grad
        b-=learning_rate*b.grad

    w.grad.zero_()
    b.grad.zero_()

    print(f'The parameters are w={w}, b={b} and loss={loss.item()}')
    print()

