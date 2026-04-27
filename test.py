import numpy as np
from NeuralCart import Builder

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)

model, loss_fn, optimizer = Builder.build_from_json("model_config.json")

print(model)
model.summary(input_shape=(4, 2))

for epoch in range(10000):
    pred = model(X)
    loss = loss_fn(pred, y)

    dout = loss_fn.backward()
    model.backward(dout)

    optimizer.step()
    optimizer.zero_grad()

    if epoch % 1000 == 0:
        print(f"epoch {epoch}, loss: {loss:.6f}")

print("final prediction:")
print(model(X))