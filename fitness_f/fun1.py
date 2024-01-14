from dolfin import *
# from mshr import *
import matplotlib.pyplot as plt
import utils.read_mesh as read_mesh
import numpy as np
def fenics_fitness(points,outside_tri):
    # set_log_level(LogLevel.WARNING)
    set_log_active(False)
    # L, H = 5, 2.5
    points*=0.125
    mesh = read_mesh.getmesh(points,outside_tri)
    mesh.init()
    def lateral_sides(x, on_boundary):
        return (near(x[0], -0.250)) and (x[1]<0.0) and on_boundary

    def end_side(x, on_boundary):
        return (near(x[0], 0.250)) and on_boundary
    def bottom(x, on_boundary):
        return near(x[1], -0.125) and on_boundary
    def top(x, on_boundary):
        return near(x[1], 0.125) and on_boundary

    VT = FunctionSpace(mesh, "CG", 1)
    T_, dT = TestFunction(VT), TrialFunction(VT)
    Delta_T = Function(VT, name="Temperature increase")
    aT = dot(grad(dT), grad(T_))*dx
    LT = Constant(0)*T_*dx

    bcT = [DirichletBC(VT, Constant(100.), bottom),
           DirichletBC(VT, Constant(0.), top),
           DirichletBC(VT, Constant(0.), lateral_sides)]
    solve(aT == LT, Delta_T, bcT)
    # plt.figure()
    # p = plot(Delta_T)
    # plt.colorbar(p)
    # plt.show()
    # plt.savefig("./1.png")

    E = Constant(6.9e10)
    nu = Constant(0.3)
    mu = E/2/(1+nu)
    lmbda = E*nu/(1+nu)/(1-2*nu)
    alpha = Constant(23.21e-6)
    bottomm = AutoSubDomain(lambda x: near(x[1], -0.125))
    g = Constant((6e5, 2e7))
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    bottomm.mark(boundaries, 1)
    dsb = ds(subdomain_data=boundaries)

    def eps(v):
        return sym(grad(v))
    def sigma(v, dT):
        return (lmbda*tr(eps(v))- alpha*(3*lmbda+2*mu)*dT)*Identity(2) + 2.0*mu*eps(v)

    Vu = VectorFunctionSpace(mesh, 'CG', 2)
    du = TrialFunction(Vu)
    u_ = TestFunction(Vu)
    Wint = inner(sigma(du, Delta_T), eps(u_))*dx
    aM = lhs(Wint)
    LM = rhs(Wint) + inner(g, u_)*dsb(1)

    bcu = [DirichletBC(Vu, Constant((0., 0.)), top),
           DirichletBC(Vu, Constant((0., 0.)), end_side)]

    u = Function(Vu, name="Displacement")

    solve(aM == LM, u, bcu)

    # plt.figure()
    # p1 = plot(1e3*u[1],title="Vertical displacement [mm]")
    # plt.colorbar(p1)
    # plt.show()
    # plt.figure()
    # p = plot(sigma(u, Delta_T)[0, 0],title="Horizontal stress [MPa]")
    # plt.colorbar(p)
    # plt.savefig("./3.png")

    # ===========
    s = sigma(u,Delta_T) - (1./3)*tr(sigma(u,Delta_T))*Identity(2)
    von_Mises = sqrt(abs(3./2*inner(s, s)))
    V = FunctionSpace(mesh, 'DG', 1)
    von_Mises = project(von_Mises, V)
    # p2=plot(von_Mises, title='Stress intensity')
    # plt.colorbar(p2)
    # plt.savefig("./Mises.png")
    max_vm_stress = max(von_Mises.vector()[:])
    # print(max_vm_stress)



    strain_energy_density = 0.5 * inner(sigma(u,0), eps(u)) # Integrate strain energy density over the domain
    # total_strain_energy = assemble(inner(g, u)*dsb(1))
    total_strain_energy = assemble(strain_energy_density * dx)
    if np.isnan(total_strain_energy):
        total_strain_energy=1e11
    if np.isnan(max_vm_stress):
        max_vm_stress=1e11
    # plot(project(strain_energy_density,V),title='strain_energy_density')
    # print(total_strain_energy)
    # plt.show()
    # plt.savefig("./5.png")

    return max_vm_stress,total_strain_energy
