import control as c
import numpy as np

ci = 2 / np.sqrt(13)
w = np.sqrt(13)
Kq = -24
T02 = 1.4
V = 160
s = c.tf([1, 0], [1])
Hq = Kq * (1 + T02 * s) / (s ** 2 + 2 * ci * w * s + w ** 2)
Htheta = Hq / s
Hgamma = Kq / s / (s ** 2 + 2 * ci * w * s + w ** 2)
Hh = Hgamma * V / s
H = c.tf([[Hq.num[0][0], Htheta.num[0][0]], [Hgamma.num[0][0], Hh.num[0][0]]],
         [[Hq.den[0][0], Htheta.den[0][0]], [Hgamma.den[0][0], Hh.den[0][0]]])
sys1 = c.ss(H)
sys1.D = np.array([[1, 2], [3, 4]])  # Changes it to a non-zero input matrix D

H = c.tf(sys1)  # Gives a tf with nice unrounded residual components

H2 = c.tf([1, -3, 0, 0], [1, 1e-13, 7, 0, 0, 0])  # to test minreal things

print(H)