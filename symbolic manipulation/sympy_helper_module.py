import sympy as sp
from sympy.core.function import AppliedUndef
from functools import cache

def simp(expr):
    return sp.simplify(sp.together(sp.factor(expr)))

def impose_relations(matrix, relations):
    if relations is None:
        return matrix
    substitutions = relation_substitutions(relations)
    return groebner_reduce_matrix(matrix.applyfunc(lambda entry: simp(entry.subs(substitutions))), relations, substitutions)

def numerator(expr):
    return sp.together(expr).as_numer_denom()[0]

def algebraic_atoms(expr):
    atoms = set(expr.atoms(sp.Derivative))
    atoms.update(expr.atoms(AppliedUndef))
    atoms.update(expr.atoms(sp.conjugate))
    return atoms

def differential_atoms(expr):
    return set(expr.atoms(sp.Derivative))

def derivative_order(atom):
    return len(atom.variables)

@cache
def relation_substitutions(relations):
    substitutions = {}
    polynomials = [numerator(relation) for relation in relations]
    for relation in polynomials:
        atoms = list(differential_atoms(relation))
        if len(atoms) == 1:
            atom = atoms[0]
            coefficient = relation.coeff(atom)
            if coefficient != 0 and sp.expand(relation - coefficient * atom) == 0:
                substitutions[atom] = 0
    for relation in polynomials:
        relation = numerator(relation.subs(substitutions))
        atoms = sorted(differential_atoms(relation), key=lambda atom: (derivative_order(atom), str(atom)), reverse=True)
        if len(atoms) > 8:
            continue
        for atom in atoms:
            coefficient = relation.coeff(atom)
            value = simp(-(relation - coefficient * atom) / coefficient) if coefficient != 0 else None
            if derivative_order(atom) > 1 and coefficient != 0 and not value.has(atom):
                substitutions[atom] = value
                break
    return substitutions

def related_polynomials(target, polynomials):
    target_atoms = differential_atoms(target)
    related = []
    changed = True
    while changed:
        changed = False
        for polynomial in polynomials:
            polynomial_atoms = differential_atoms(polynomial)
            if polynomial not in related and polynomial_atoms & target_atoms:
                related.append(polynomial)
                target_atoms.update(polynomial_atoms)
                changed = True
    return related

def groebner_reduce_matrix(matrix, relations, substitutions):
    return matrix.applyfunc(lambda entry: groebner_reduce(entry, relations, substitutions))

def groebner_reduce(expr, relations, substitutions):
    target = numerator(expr)
    if target == 0:
        return 0
    polynomials = [numerator(relation.subs(substitutions)) for relation in relations if relation != 0]
    polynomials = [polynomial for polynomial in related_polynomials(target, polynomials) if polynomial != 0]
    if not polynomials:
        return simp(expr)
    atoms = set().union(*[algebraic_atoms(polynomial) for polynomial in polynomials], algebraic_atoms(target))
    if not atoms:
        return simp(expr)
    atoms = sorted(atoms, key=str)
    replacements = {atom: sp.Symbol(f"X_{index}") for index, atom in enumerate(atoms)}
    inverse_replacements = {symbol: atom for atom, symbol in replacements.items()}
    generators = list(replacements.values())
    polynomial_basis = sp.groebner(
        [polynomial.xreplace(replacements) for polynomial in polynomials],
        *generators,
        order="grevlex",
    )
    _, remainder = polynomial_basis.reduce(target.xreplace(replacements))
    return simp(remainder.xreplace(inverse_replacements))
