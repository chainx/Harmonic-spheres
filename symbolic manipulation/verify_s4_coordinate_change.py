import sympy as sp
from sympy.diffgeom import CoordSystem, Manifold, Patch

def simp(expr):
    return sp.cancel(sp.together(sp.simplify(expr)))

# w = u + iv, z = x + iy
u, v, x, y = sp.symbols("u v x y", real=True)

M = Manifold("M", 4)
P = Patch("P", M)
C = CoordSystem("C", P, [u, v, x, y])

uf, vf, xf, yf = C.coord_functions()

def differential(expr):
    return sp.Matrix([bv.rcall(expr) for bv in C.base_vectors()])

# ===============   Constructing metric on HP^1 in S^2 x D^2 coords   ================

abs_w_sq = uf**2 + vf**2
abs_z_sq = xf**2 + yf**2

D = 1 + abs_w_sq * abs_z_sq
# D = 1+|w|^2+|z|^2 is real and positive, so re/im split directly

# a = (1-|z|^2)/(1+|w|^2+|z|^2) * w
a_re = (1 - abs_z_sq) * uf / D
a_im = (1 - abs_z_sq) * vf / D
da_re = differential(a_re)
da_im = differential(a_im)

# b = (1+|w|^2)/(1+|w|^2+|z|^2) * z
b_re = (1 + abs_w_sq) * xf / D
b_im = (1 + abs_w_sq) * yf / D
db_re = differential(b_re)
db_im = differential(b_im)

denom = (1 + a_re**2 + a_im**2 + b_re**2 + b_im**2) ** 2

dadā = da_re * da_re.T + da_im * da_im.T
dbdƃ = db_re * db_re.T + db_im * db_im.T
g_HP1 = (4 / denom) * (dadā + dbdƃ)

# ===============   Comparing to S^2 x D^2 metric   ================

g_w = 4 * ((1 - abs_z_sq) / (1 + abs_z_sq)) ** 2 / (1 + abs_w_sq) ** 2
g_z = 4 / (1 + abs_z_sq) ** 2
# In (dw,dw̄,dz,dz̄) coords, the metric is block diagonal with off-diagonal blocks
# In (du,dv,dx,dy) coords, the metric is diagonal
g_S2D2 = sp.diag(g_w, g_w, g_z, g_z)

diff = (g_HP1 - g_S2D2).applyfunc(simp)
assert diff==sp.zeros(4,4), diff
