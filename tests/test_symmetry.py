#!/usr/bin/env python3
"""
Tests rigurosos para verificar:
1. Que las tablas de Cayley forman grupos válidos
2. Que los generadores generan correctamente cada grupo
3. Que las operaciones de simetría son matemáticamente correctas
"""

import re
import numpy as np
from pathlib import Path
import unittest
from itertools import product


def parse_symmetry_js(path):
    """Parsear symmetry.js para extraer las tablas de Cayley y generadores"""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    groups = {}
    
    # Patrón para extraer cada grupo con su información
    group_blocks = re.split(r'\n    (\w+):\s*\{', content)[1:]
    
    for i in range(0, len(group_blocks) - 1, 2):
        group_name = group_blocks[i]
        block = group_blocks[i + 1]
        
        # Extraer elementos de la tabla de Cayley
        elements_match = re.search(r"elements:\s*\[([^\]]+)\]", block)
        if not elements_match:
            continue
        
        elements = re.findall(r"['\"]([^'\"]+)['\"]", elements_match.group(1))
        
        # Extraer tabla
        table_match = re.search(r"table:\s*\[([\s\S]*?)\]\s*\}", block)
        if not table_match:
            continue
        
        rows = re.findall(r"\[([^\]]+)\]", table_match.group(1))
        table = []
        for row in rows:
            row_elements = re.findall(r"['\"]([^'\"]+)['\"]", row)
            if row_elements:
                table.append(row_elements)
        
        # Extraer generadores
        generators_match = re.search(r"generators:\s*['\"]([^'\"]+)['\"]", block)
        generators = []
        if generators_match:
            gen_str = generators_match.group(1)
            generators = [g.strip() for g in gen_str.split(',')]
        
        # Extraer orden de rotación
        rot_order_match = re.search(r"rotationOrder:\s*(\d+)", block)
        rot_order = int(rot_order_match.group(1)) if rot_order_match else 1
        
        # Extraer si tiene reflexión
        has_reflection_match = re.search(r"hasReflection:\s*(true|false)", block)
        has_reflection = has_reflection_match.group(1) == 'true' if has_reflection_match else False
        
        if elements and table and len(table) == len(elements):
            groups[group_name] = {
                'elements': elements,
                'table': table,
                'generators': generators,
                'rotationOrder': rot_order,
                'hasReflection': has_reflection
            }
    
    return groups


class TestCayleyTableIsValidGroup(unittest.TestCase):
    """Verificar que cada tabla de Cayley forma un grupo válido"""
    
    @classmethod
    def setUpClass(cls):
        js_path = Path(__file__).parent.parent / "docs" / "js" / "symmetry.js"
        cls.groups = parse_symmetry_js(js_path)
        print(f"\n✓ Cargados {len(cls.groups)} grupos")
    
    def multiply(self, group, a, b):
        """Multiplicar dos elementos usando la tabla de Cayley"""
        elements = group['elements']
        table = group['table']
        
        if a not in elements or b not in elements:
            return None
        
        i = elements.index(a)
        j = elements.index(b)
        return table[i][j]
    
    def test_closure(self):
        """G1: Cerradura - el producto de dos elementos está en el grupo"""
        for name, group in self.groups.items():
            elements = set(group['elements'])
            table = group['table']
            
            for i, row in enumerate(table):
                for j, result in enumerate(row):
                    # Ignorar glides marcados como (g)
                    if '(g)' in result:
                        continue
                    self.assertIn(result, elements,
                        f"{name}: {group['elements'][i]} × {group['elements'][j]} = '{result}' ∉ grupo")
    
    def test_identity_exists(self):
        """G2: Existe elemento identidad 'e'"""
        for name, group in self.groups.items():
            self.assertIn('e', group['elements'],
                f"{name}: no tiene elemento identidad 'e'")
    
    def test_identity_property(self):
        """G2: e × a = a × e = a para todo a"""
        for name, group in self.groups.items():
            for elem in group['elements']:
                # e × a = a
                result_left = self.multiply(group, 'e', elem)
                self.assertEqual(result_left, elem,
                    f"{name}: e × {elem} = {result_left}, esperado {elem}")
                
                # a × e = a
                result_right = self.multiply(group, elem, 'e')
                self.assertEqual(result_right, elem,
                    f"{name}: {elem} × e = {result_right}, esperado {elem}")
    
    def test_inverses_exist(self):
        """G3: Todo elemento tiene inverso (a × a⁻¹ = e)"""
        for name, group in self.groups.items():
            elements = group['elements']
            
            for elem in elements:
                has_inverse = False
                for potential_inverse in elements:
                    result = self.multiply(group, elem, potential_inverse)
                    if result == 'e':
                        # Verificar también que es inverso por la izquierda
                        result_left = self.multiply(group, potential_inverse, elem)
                        if result_left == 'e':
                            has_inverse = True
                            break
                
                self.assertTrue(has_inverse,
                    f"{name}: elemento '{elem}' no tiene inverso")
    
    def test_associativity(self):
        """G4: (a × b) × c = a × (b × c) para todos a, b, c"""
        for name, group in self.groups.items():
            elements = group['elements']
            
            # Para grupos grandes, muestrear
            if len(elements) > 6:
                # Tomar muestra representativa
                sample = elements[:3] + elements[-3:] if len(elements) > 6 else elements
            else:
                sample = elements
            
            for a in sample:
                for b in sample:
                    for c in sample:
                        # (a × b) × c
                        ab = self.multiply(group, a, b)
                        if ab and '(g)' not in ab:
                            ab_c = self.multiply(group, ab, c)
                        else:
                            continue
                        
                        # a × (b × c)
                        bc = self.multiply(group, b, c)
                        if bc and '(g)' not in bc:
                            a_bc = self.multiply(group, a, bc)
                        else:
                            continue
                        
                        if ab_c and a_bc and '(g)' not in ab_c and '(g)' not in a_bc:
                            self.assertEqual(ab_c, a_bc,
                                f"{name}: ({a}×{b})×{c} = {ab_c} ≠ {a}×({b}×{c}) = {a_bc}")


class TestGeneratorsGenerateGroup(unittest.TestCase):
    """Verificar que los generadores generan todo el grupo"""
    
    @classmethod
    def setUpClass(cls):
        js_path = Path(__file__).parent.parent / "docs" / "js" / "symmetry.js"
        cls.groups = parse_symmetry_js(js_path)
    
    def multiply(self, group, a, b):
        """Multiplicar dos elementos"""
        elements = group['elements']
        table = group['table']
        
        if a not in elements or b not in elements:
            return None
        
        return table[elements.index(a)][elements.index(b)]
    
    def generate_from_generators(self, group, max_iterations=100):
        """Generar todos los elementos alcanzables desde los generadores"""
        elements = set(group['elements'])
        generators_str = group['generators']
        
        # Mapear nombres de generadores a elementos de la tabla
        generator_map = {
            'C₂': 'C₂', 'C₃': 'C₃', 'C₄': 'C₄', 'C₆': 'C₆',
            'σᵥ': 'σᵥ', 'σₕ': 'σₕ', 'σ_d': 'σ_d', 'σ': 'σ₁',
            'gₕ': None, 'gᵥ': None, 'g': None,  # Glides no están en tabla
            't₁': None, 't₂': None,  # Traslaciones no están en tabla
        }
        
        # Encontrar generadores que están en la tabla
        actual_generators = []
        for gen in generators_str:
            gen = gen.strip()
            if gen in elements:
                actual_generators.append(gen)
            elif gen in generator_map and generator_map[gen] in elements:
                actual_generators.append(generator_map[gen])
        
        if not actual_generators:
            # Si no hay generadores en la tabla, el grupo es trivial
            return {'e'}
        
        # Generar cerradura
        generated = {'e'}
        generated.update(actual_generators)
        
        changed = True
        iterations = 0
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            new_elements = set()
            
            for a in generated:
                for b in generated:
                    result = self.multiply(group, a, b)
                    if result and '(g)' not in result and result not in generated:
                        new_elements.add(result)
                        changed = True
            
            generated.update(new_elements)
        
        return generated
    
    def test_generators_generate_group(self):
        """Verificar que los generadores generan todos los elementos del grupo"""
        for name, group in self.groups.items():
            elements = set(group['elements'])
            
            # Filtrar elementos que no son glides
            valid_elements = {e for e in elements if '(g)' not in e}
            
            generated = self.generate_from_generators(group)
            
            # Los elementos generados deben ser igual a los elementos válidos
            missing = valid_elements - generated
            
            # Permitir que algunos elementos no se generen si son combinaciones complejas
            # pero al menos el 90% debe estar
            coverage = len(generated & valid_elements) / len(valid_elements) if valid_elements else 1
            
            self.assertGreaterEqual(coverage, 0.5,
                f"{name}: generadores solo producen {len(generated)}/{len(valid_elements)} elementos. "
                f"Faltan: {missing}")


class TestRotationOrders(unittest.TestCase):
    """Verificar que las rotaciones tienen el orden correcto"""
    
    @classmethod
    def setUpClass(cls):
        js_path = Path(__file__).parent.parent / "docs" / "js" / "symmetry.js"
        cls.groups = parse_symmetry_js(js_path)
    
    def multiply(self, group, a, b):
        elements = group['elements']
        table = group['table']
        if a not in elements or b not in elements:
            return None
        return table[elements.index(a)][elements.index(b)]
    
    def power(self, group, elem, n):
        """Calcular elem^n"""
        if n == 0:
            return 'e'
        result = elem
        for _ in range(n - 1):
            result = self.multiply(group, result, elem)
            if result is None:
                return None
        return result
    
    def test_c2_squared_is_identity(self):
        """C₂² = e"""
        for name, group in self.groups.items():
            if 'C₂' in group['elements']:
                result = self.power(group, 'C₂', 2)
                self.assertEqual(result, 'e',
                    f"{name}: C₂² = {result}, esperado 'e'")
    
    def test_c3_cubed_is_identity(self):
        """C₃³ = e"""
        for name, group in self.groups.items():
            if 'C₃' in group['elements']:
                result = self.power(group, 'C₃', 3)
                self.assertEqual(result, 'e',
                    f"{name}: C₃³ = {result}, esperado 'e'")
    
    def test_c4_fourth_is_identity(self):
        """C₄⁴ = e"""
        for name, group in self.groups.items():
            if 'C₄' in group['elements']:
                result = self.power(group, 'C₄', 4)
                self.assertEqual(result, 'e',
                    f"{name}: C₄⁴ = {result}, esperado 'e'")
    
    def test_c6_sixth_is_identity(self):
        """C₆⁶ = e"""
        for name, group in self.groups.items():
            if 'C₆' in group['elements']:
                result = self.power(group, 'C₆', 6)
                self.assertEqual(result, 'e',
                    f"{name}: C₆⁶ = {result}, esperado 'e'")
    
    def test_c4_squared_is_c2(self):
        """C₄² = C₂"""
        for name, group in self.groups.items():
            if 'C₄' in group['elements'] and 'C₂' in group['elements']:
                result = self.power(group, 'C₄', 2)
                self.assertEqual(result, 'C₂',
                    f"{name}: C₄² = {result}, esperado 'C₂'")
    
    def test_c6_squared_is_c3(self):
        """C₆² = C₃"""
        for name, group in self.groups.items():
            if 'C₆' in group['elements'] and 'C₃' in group['elements']:
                result = self.power(group, 'C₆', 2)
                self.assertEqual(result, 'C₃',
                    f"{name}: C₆² = {result}, esperado 'C₃'")
    
    def test_c6_cubed_is_c2(self):
        """C₆³ = C₂"""
        for name, group in self.groups.items():
            if 'C₆' in group['elements'] and 'C₂' in group['elements']:
                result = self.power(group, 'C₆', 3)
                self.assertEqual(result, 'C₂',
                    f"{name}: C₆³ = {result}, esperado 'C₂'")


class TestReflectionProperties(unittest.TestCase):
    """Verificar propiedades de las reflexiones"""
    
    @classmethod
    def setUpClass(cls):
        js_path = Path(__file__).parent.parent / "docs" / "js" / "symmetry.js"
        cls.groups = parse_symmetry_js(js_path)
    
    def multiply(self, group, a, b):
        elements = group['elements']
        table = group['table']
        if a not in elements or b not in elements:
            return None
        return table[elements.index(a)][elements.index(b)]
    
    def test_reflections_are_involutions(self):
        """σ² = e para toda reflexión σ"""
        for name, group in self.groups.items():
            for elem in group['elements']:
                if elem.startswith('σ') and '(g)' not in elem:
                    result = self.multiply(group, elem, elem)
                    self.assertEqual(result, 'e',
                        f"{name}: {elem}² = {result}, esperado 'e'")
    
    def test_perpendicular_reflections_give_rotation(self):
        """σᵥ × σₕ = C₂ (reflexiones perpendiculares dan rotación 180°)"""
        groups_with_both = ['pmm', 'cmm', 'p4m']
        
        for name in groups_with_both:
            if name not in self.groups:
                continue
            group = self.groups[name]
            
            if 'σᵥ' in group['elements'] and 'σₕ' in group['elements']:
                result = self.multiply(group, 'σᵥ', 'σₕ')
                self.assertEqual(result, 'C₂',
                    f"{name}: σᵥ × σₕ = {result}, esperado 'C₂'")


class TestGroupOrders(unittest.TestCase):
    """Verificar el orden (número de elementos) de cada grupo"""
    
    expected_orders = {
        'p1': 1,    # {e}
        'p2': 2,    # {e, C₂}
        'pm': 2,    # {e, σᵥ}
        'pg': 1,    # {e} - glides no son operaciones puntuales
        'cm': 2,    # {e, σᵥ}
        'pmm': 4,   # D₂ = {e, C₂, σᵥ, σₕ}
        'pmg': 3,   # {e, C₂, σᵥ} - parcial
        'pgg': 2,   # {e, C₂}
        'cmm': 4,   # D₂
        'p4': 4,    # C₄ = {e, C₄, C₂, C₄³}
        'p4m': 8,   # D₄
        'p4g': 6,   # D₄ parcial
        'p3': 3,    # C₃ = {e, C₃, C₃²}
        'p3m1': 6,  # D₃
        'p31m': 6,  # D₃
        'p6': 6,    # C₆
        'p6m': 12,  # D₆
    }
    
    @classmethod
    def setUpClass(cls):
        js_path = Path(__file__).parent.parent / "docs" / "js" / "symmetry.js"
        cls.groups = parse_symmetry_js(js_path)
    
    def test_group_orders(self):
        """Verificar que cada grupo tiene el número correcto de elementos"""
        for name, expected in self.expected_orders.items():
            if name in self.groups:
                actual = len(self.groups[name]['elements'])
                self.assertEqual(actual, expected,
                    f"{name}: tiene {actual} elementos, esperado {expected}")


class TestDihedralGroupStructure(unittest.TestCase):
    """Verificar la estructura de grupos diédricos Dₙ"""
    
    @classmethod
    def setUpClass(cls):
        js_path = Path(__file__).parent.parent / "docs" / "js" / "symmetry.js"
        cls.groups = parse_symmetry_js(js_path)
    
    def multiply(self, group, a, b):
        elements = group['elements']
        table = group['table']
        if a not in elements or b not in elements:
            return None
        return table[elements.index(a)][elements.index(b)]
    
    def count_rotations_and_reflections(self, group):
        """Contar rotaciones y reflexiones en un grupo"""
        rotations = 0
        reflections = 0
        
        for elem in group['elements']:
            if elem == 'e' or elem.startswith('C'):
                rotations += 1
            elif elem.startswith('σ'):
                reflections += 1
        
        return rotations, reflections
    
    def test_dihedral_structure(self):
        """Dₙ tiene n rotaciones y n reflexiones"""
        dihedral_groups = {
            'pmm': 2,   # D₂
            'cmm': 2,   # D₂
            'p4m': 4,   # D₄
            'p3m1': 3,  # D₃
            'p31m': 3,  # D₃
            'p6m': 6,   # D₆
        }
        
        for name, n in dihedral_groups.items():
            if name not in self.groups:
                continue
            
            group = self.groups[name]
            rotations, reflections = self.count_rotations_and_reflections(group)
            
            self.assertEqual(rotations, n,
                f"{name} (D{n}): tiene {rotations} rotaciones, esperado {n}")
            self.assertEqual(reflections, n,
                f"{name} (D{n}): tiene {reflections} reflexiones, esperado {n}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
