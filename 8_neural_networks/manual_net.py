step = 0.0001
accuracy = 1e-12

x = 10
y = 20
a1 = x
w1 = 1
a2 = w1 * a1
w2 = 3
a3 = w2 * a2

while abs(a3 - y) > accuracy:
    print(w1, w2, w1*w2)
    w2, w1 = w2 - step * 2 * (a3 - y) * a2, w1 - step * 2 * (a3 - y) * w2 * a1
    a2 = w1 * a1
    a3 = w2 * a2
