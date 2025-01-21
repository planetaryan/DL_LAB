import torch
import matplotlib.pyplot as plt

x = torch.tensor( [12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9, 18.8, 20.3, 22.4,
19.4, 15.5, 16.7, 17.3, 18.4, 19.2, 17.4, 19.5, 19.7, 21.2])

y = torch.tensor( [11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9, 15.6, 16.4, 17.7, 19.6,
16.9, 14.0, 14.6, 15.1, 16.1, 16.8, 15.2, 17.0, 17.2, 18.6])

b=torch.tensor([1.0],requires_grad=True)
w=torch.tensor([1.0],requires_grad=True)
print(f'The parameters are {w} and {b}')

learning_rate=torch.tensor(0.001)

loss_list=[]

for epoch in range(100):
    loss=0.0
    for j in range(len(x)):
        y_p=(w*x[j])+b
        loss+=(y_p-y[j])**2
    
    loss=loss/len(x)

    loss_list.append(loss.item())

    loss.backward()

    with torch.no_grad():
        w-=learning_rate*w.grad
        b-=learning_rate*b.grad
    
    w.grad.zero_()
    b.grad.zero_()

    print(f'The parameters are w={w}, b={b} and loss={loss.item()}')
plt.plot(loss_list)
plt.show()

