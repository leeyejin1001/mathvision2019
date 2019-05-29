from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np
from matplotlib import cm


fig = plt.figure(figsize = plt.figaspect(0.7))

#--- First subplot

ax = fig.add_subplot(111,projection = '3d')
X = np.arange(-1.5,1.5, 0.05)
Y = np.arange(-1.5,1.5, 0.05)
X,Y = np.meshgrid(X,Y)
Z = (X + Y) * (X*Y + (X*(Y*Y)))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

#X = 1
#Y = 1.5
ZGrad = (2*X*Y + (2*X*Y**2) + Y**2 + Y**3, X**2 + (2*X**2*Y)+2*X*Y + (3*X*Y**2))

criticalPoints = np.array(([0, 0], [0, -1], [1, -1], [3 / 8, -3 / 4]))


for i in range (0,4,1):
    x = criticalPoints[i][0]
    y = criticalPoints[i][1]
    ZHessian = np.array(((2 * y + 2 * y ** 2, 2 * x + 4 * x * y + 2 * y + 3 * y ** 2),
                         (2 * x + 4 * x * y + 2 * y + 3 * y ** 2, 2 * x ** 2 + 2 * x + 6 * x * y)))
    w, v = np.linalg.eig(ZHessian)
    print("\n x,y: ",x,y)
    print("ZHessian  :\n", ZHessian)
    print("\n ZHessian eig v : \n", v)

for i in range(0,4,1):
    ax.plot([criticalPoints[i][0]], [criticalPoints[i][1]], 'ro')



#print("ZGrad: ", ZGrad)
surf = ax.plot_surface(X,Y,Z, rstride = 1, cstride = 1, cmap = cm.jet, edgecolor = 'none' )

# ax.set_zlim(-2,20)

fig.colorbar(surf, shrink = 0.5, aspect = 10)

ax.set_title('z = (x+y)(xy+xy^2)')

plt.show()