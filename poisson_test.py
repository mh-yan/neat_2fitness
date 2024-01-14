# show pictures in the cells right away
from dolfin import *
# this is not always good. 
import matplotlib.pyplot as plt
# matplotlib is for plotting.

mesh = UnitSquareMesh(16, 16)

V = FunctionSpace(mesh, 'Lagrange', 2)
V.dim() 

u = TrialFunction(V)
v = TestFunction(V)
u_ = Function(V)

f = Constant(-6.0)
# form=inner(grad(u_),grad(v))*dx-f*v*dx
form=(div(grad(u_))+f)*v*dx

u0 = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

def u0_boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u0, u0_boundary)

solve(form==0, u_, bc)