# -*- coding: utf-8 -*-
# BioSTEAM: The Biorefinery Simulation and Techno-Economic Analysis Modules
# Copyright (C) 2020-2023, Yoel Cortes-Pena <yoelcortes@gmail.com>
#
# This module is under the UIUC open-source license. See
# github.com/BioSTEAMDevelopmentGroup/biosteam/blob/master/LICENSE.txt
# for license details.
"""
"""
import pytest
import thermosteam as tmo
import numpy as np
from numpy.testing import assert_allclose
from thermosteam import utils
from thermosteam.utils import misc, representation

def test_misc_helpers():
    assert misc.flattened([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]
    class A: pass
    a = A(); b = A()
    b.x = 1; b.y = 2
    misc.fill_like(a, b, ('x', 'y'))
    assert misc.getfields(a, ('x', 'y')) == [1, 2]
    misc.setfields(a, ('x', 'y'), (3, 4))
    assert (a.x, a.y) == (3, 4)
    arr = np.ones(2)
    copy = misc.copy_maybe(arr)
    assert copy is not arr and (copy == arr).all()
    assert misc.copy_maybe(1) == 1
    assert misc.get_instance([1, 'a', 2.0], str) == 'a'
    assert misc.factor('kg', 'kg') == 1
    assert misc.checkbounds(1, (0, 2))
    assert not misc.checkbounds(3, (0, 2))

def test_misc_rounding():
    assert misc.roundsigfigs(1234.5) == 1200
    assert_allclose(misc.roundsigfigs(0.012345, 3), 0.0123, rtol=1e-2)
    assert misc.roundsigfigs(0.) == 0
    assert misc.roundsigfigs('abc') == 'abc'
    value = misc.roundsigfigs([1.234, 1234.5], 2)
    assert (value == [0, 1200]).all()
    value = misc.roundsigfigs(np.array([0.1234, 0.5678]), 2)
    assert_allclose(value, [0.12, 0.57], rtol=1e-2)
    assert misc.docround([1.23456, 2.34567], 2) == [1.23, 2.35]
    assert_allclose(misc.docround(np.array([1.23456]), 2), [1.23])
    assert misc.docround(1.23456, 1) == 1.2
    arr = np.array([1234.5, 0.012345, 0.], dtype=object)
    out = misc.array_roundsigfigs(arr, 2)
    assert out[0] == 1200 and out[2] == 0
    assert_allclose(float(out[1]), 0.012, rtol=1e-2)
    arr = np.array(['1234.5', 'abc'], dtype=object)
    out = misc.array_roundsigfigs(arr, 2, inplace=True)
    assert out is arr
    assert out[0] == 1200 and out[1] == 'abc'

def test_misc_arrays_and_strings():
    array, keys = misc.dictionaries2array([{'a': 1, 'b': 2}, {'a': 3}])
    assert array.shape == (2, 2)
    assert array.sum() == 6
    assert misc.list_available_names(['a', 'b', 'c']) == "'a', 'b', and 'c'"
    assert misc.strtuple([1, misc.strtuple, 'x']) == '(1, strtuple, x)'
    assert misc.format_title('heat_utilityCost') == 'Heat utility cost'
    assert misc.format_title('a') == 'A'
    assert misc.format_unit_name('heat exchanger') == 'HeatExchanger'
    assert misc.subgroup([1, 2, 3], 2) == [[1, 2], [3]]
    assert misc.repr_subgroups([[1, 2], [3]]) == ['1, 2', '3']
    assert misc.repr_items('x=', [1, 2, 3], 2) == 'x=1, 2,\n  3'
    assert misc.repr_items('x=', [1, 2, 3], 2, brackets='[]') == 'x=[1, 2,\n   3]'

def test_misc_chemical_dicts():
    chemicals = tmo.Chemicals(['Water', 'O2,g', 'Glucose,s', 'Ethanol,l'])
    data = {'Water': 1, 'Methanol': 2}
    misc.remove_undefined_chemicals(data, chemicals)
    assert data == {'Water': 1}
    dct = {}
    misc.default_chemical_dict(dct, chemicals, 'g', 'l', 's')
    assert dct == {'Water': 'l', 'O2': 'g', 'Glucose': 's', 'Ethanol': 'l'}
    dct = {'Water': 0}
    misc.default_chemical_dict(dct, chemicals, 'g', 'l', 's', 'n')
    assert dct['Water'] == 0 and dct['O2'] == 'g'

def test_extended_signature():
    from inspect import signature, _empty as empty
    class DesignInput:
        def __init__(self, name, default=empty, return_type=None):
            self.__name__ = name
            self.default = default
            self._return_type = return_type
    def f(self, a, b=1, **kwargs): pass
    def g(self, c=2, d=3): pass
    design_inputs = [
        DesignInput('x', 1.0, float),
        DesignInput('y'),
        DesignInput('c'),
        DesignInput('a'),
    ]
    h = misc.extended_signature(f, g, design_inputs)
    params = signature(h).parameters
    assert list(params) == ['self', 'a', 'b', 'x', 'y', 'c', 'd']
    assert params['c'].default == 2
    assert params['x'].default == 1.0
    assert h.extension is g
    assert h.__wrapped__ is f
    assert g.parameters == frozenset(['c', 'd'])
    h2 = misc.extended_signature(f, None, [DesignInput('z', return_type=int)])
    assert 'z' in signature(h2).parameters
    def g_bad(self, x=2): pass
    with pytest.raises(RuntimeError):
        misc.extended_signature(f, g_bad, [DesignInput('x', 1.0)])

def test_representation():
    assert representation.repr_names([]) == 'none'
    assert representation.repr_names(['a']) == "'a' only"
    assert representation.repr_names(['a', 'b']) == 'a and b'
    assert representation.repr_names(['a', 'b', 'c']) == 'a, b and c'
    assert representation.repr_IDs_data(['a', 'b'], [1, 0]) == ', a=1'
    assert representation.repr_IDs_data(['a'], [0]) == ''
    assert representation.repr_IDs_data(['a'], [1], start='') == 'a=1'
    assert representation.repr_kwargs({}) == ''
    assert representation.repr_args([]) == ''
    assert representation.repr_kwarg('a', 0) == ''
    assert representation.repr_kwarg('a', 1) == ', a=1'
    assert representation.repr_couples(', ', ['a', 'b'], [1, 0]) == "('a', 1)"
    assert representation.repr_listed_values(['a']) == 'a'

def test_registered():
    from thermosteam.utils import registered, unregistered
    @registered('X')
    class Obj:
        def __init__(self, ID=''):
            self._register(ID)
    a = Obj()
    assert a.ID == 'X1'
    assert repr(a) == '<Obj: X1>' and str(a) == 'X1'
    b = Obj(5)
    assert b.ID == 'X6'
    c = Obj(['c', 'c_alias'])
    assert c.ID == 'c'
    assert Obj.registry.search('c_alias') is c
    c.register_alias('c_alias2', safe=False)
    assert Obj.registry.search('c_alias2') is c
    with pytest.raises(ValueError):
        Obj([1.5])
    with pytest.raises(ValueError):
        Obj(1.5)
    d = Obj('d')
    d.ID = '.d_unregistered'
    assert d.ID == 'd_unregistered'
    assert Obj.registry.search('d') is None
    d._register(None)
    assert d.ID == '' and repr(d) == '<Obj>' and str(d) == '-'
    Obj.autonumber = False
    try:
        e = Obj()
        assert e.ID == 'X'
    finally:
        Obj.autonumber = True

    @unregistered
    class Unreg:
        def __init__(self, ID):
            self._register(ID)
            self.register_alias('alias')
    assert Unreg('u')._ID == 'alias'

def test_registered_franchise():
    from thermosteam.utils import registered, registered_franchise
    @registered('P')
    class Parent: pass
    @registered_franchise(Parent)
    class Child:
        ticket_name = 'P'
        def __init__(self, ID=None):
            self._register(ID)
    a = Child()
    assert a.ID == '' and str(a) == '-'
    b = Child('b')
    assert Parent.registry.search('b') is b
    assert repr(b) == '<Child: b>'
    b.ID = '.b2'
    assert b.ID == 'b2' and Parent.registry.search('b') is None
    c = Child(3)
    assert c.ID == 'P4'
    d = Child('')
    assert d.ID.startswith('P')
    Parent.autonumber = False
    try:
        e = Child('')
        assert e.ID == 'P'
    finally:
        Parent.autonumber = True

def test_heat_and_power():
    Heat = tmo._heat_and_power.Heat
    Power = tmo._heat_and_power.Power
    h1 = Heat(None, heat=10.)
    h2 = Heat(None, heat=5., source='a', sink='b')
    p1 = Power(None, power=3., source='a')
    assert h2.source == 'a' and h2.sink == 'b'
    assert repr(h1) == '<Heat>' and str(h1) == '-'
    h = Heat.sum([h1, h2, p1])
    assert_allclose(h.heat, 18.)
    assert not h.isempty()
    h.empty()
    assert h.isempty()
    with pytest.raises(TypeError):
        h.mix_from([1])
    p2 = Power(None, power=2., sink='b')
    p3 = Power(None, power=1., source='a', sink='b')
    assert p1.heat == 3.
    assert p1.source == 'a' and p2.sink == 'b'
    assert repr(p3) == '<Power>'
    p = Power.sum([p1, p2, p3])
    assert_allclose(p.power, 6.)
    assert not p.isempty()
    p.empty()
    assert p.isempty()
    with pytest.raises(TypeError):
        p.mix_from([h1])

def test_plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from thermosteam.utils import plots
    rc = matplotlib.rcParams.copy()
    try:
        plots.set_font(size=9)
        plots.set_figure_size()
        plots.set_figure_size('half', 0.5)
        plots.set_figure_size(10, units='cm')
        assert_allclose(matplotlib.rcParams['figure.figsize'][0], 10 / 2.54, rtol=1e-2)
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        ticks = plots.set_ticks(ax, [0, 0.5, 1], 'x', ['a', 'b', 'c'], ha='center')
        assert len(ticks) == 3
        ticks = plots.set_ticks(ax, [0, 0.5, 1], 'y', ['a', 'b', 'c'], offset=True)
        assert len(ticks) == 2
        with pytest.raises(ValueError):
            plots.set_ticks(ax, [0, 1], 'z')
        plt.close(fig)
    finally:
        plt.close('all')
        matplotlib.rcParams.update(rc)

def test_stream_filters():
    class Unit:
        def __init__(self, **kwargs): self.__dict__.update(kwargs)
    from thermosteam.utils import stream_filters as sf
    tmo.settings.set_thermo(['Water', 'Ethanol'], cache=True)
    feed = tmo.Stream(None, Water=1)
    mid = tmo.Stream(None, Water=1)
    product = tmo.Stream(None, Ethanol=1)
    U1 = Unit(_ins=[feed], _outs=[mid], ins=[feed], outs=[mid], line='Mixer', _interaction=False, _universal=False)
    U2 = Unit(_ins=[mid], _outs=[product], ins=[mid], outs=[product], line='Processing', _interaction=True, _universal=False)
    feed._sink = U1
    mid._source = U1; mid._sink = U2
    product._source = U2
    units = [U1, U2]
    assert sf.inlets(units) == {feed, mid}
    assert sf.outlets(units) == {mid, product}
    assert sf.streams_from_units(units) == {feed, mid, product}
    system = Unit(converge=None, streams=[feed])
    assert sf.streams_from_path([system, U2]) == {feed, mid, product}
    assert sf.process_streams([feed, mid, product]) == {mid}
    assert sf.feeds([feed, mid, product]) == [feed]
    assert sf.products([feed, mid, product]) == [product]
    streams = {feed, mid}
    sf.filter_out_missing_streams(streams)
    assert streams == {feed, mid}
    assert sf.feeds_from_units(units) == [feed]
    assert sf.products_from_units(units) == [product]
    assert sf.get_inlet_origin(mid) is feed
    assert sf.get_inlet_origin(product) is feed
    U2.line = 'processing'
    U2._interaction = False
    assert sf.get_inlet_origin(product) is product
    assert sf.get_fresh_process_water_streams([feed, mid, product]) == [feed]
    product.price = 0.
    free = sf.FreeProductStreams([feed, mid, product], wastewater_keys=set())
    assert product in free.streams
    assert free.streams is free.streams
    assert product in free.combustibles
    assert not free.combustible_gases
    assert free.combustible_gases is free.combustible_gases
    assert product in free.combustible_slurries
    assert free.combustible_slurries is free.combustible_slurries
    assert not free.noncombustibles
    assert free.noncombustibles is free.noncombustibles
    assert not free.noncombustible_slurries
    assert free.noncombustible_slurries is free.noncombustible_slurries
    assert 'FreeProductStreams' in repr(free)
    for i in (feed, mid, product): i._source = i._sink = None

def test_variable_assignment_inference():
    from thermosteam.utils import infer_variable_assignment
    def caller():
        return infer_variable_assignment(caller)
    class Ctx:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __init__(self): self.name = infer_variable_assignment(Ctx)
    name = caller()
    assert name == 'name'
    obj = Ctx()
    assert obj.name == 'obj'
    with Ctx() as ctx:
        assert ctx.name == 'ctx'
    assert caller() == '-'
    ns = {'caller': caller}
    exec("x = caller()", ns)
    assert ns['x'] in ('x', '-')
    other = lambda: caller()
    y = other()
    assert y == '-'

def test_exceptions():
    from thermosteam import exceptions as exc
    error = exc.StampedKeyError('msg')
    assert str(error) == 'msg' and repr(error) == 'KeyError(msg)'
    assert exc.InfeasibleRegion('x').region == 'x'
    assert exc.UndefinedChemicalAlias('x').ID == 'x'
    exc.UndefinedPhase('x'); exc.InvalidMethod('x')
    assert exc.message_with_object_stamp('obj', "'obj' failed") == "'obj' failed"
    assert 'failed' in exc.message_with_object_stamp('obj', 'failed')
    def raise_key(): raise KeyError('k')
    def raise_stamped(): raise exc.StampedKeyError('k')
    def raise_value(): raise ValueError('v')
    def raise_no_args(): raise ValueError()
    assert exc.try_method_with_object_stamp('obj', lambda x: x, (1,)) == 1
    with pytest.raises(KeyError):
        exc.try_method_with_object_stamp('obj', raise_key)
    with pytest.raises(KeyError):
        exc.try_method_with_object_stamp('obj', raise_stamped)
    with pytest.raises(ValueError, match='v'):
        exc.try_method_with_object_stamp('obj', raise_value)
    with pytest.raises(ValueError):
        exc.try_method_with_object_stamp('obj', raise_no_args)

def test_functional():
    from thermosteam import functional as fn
    assert_allclose(fn.horner(2., [1., 2., 3.]), 11.)
    assert_allclose(fn.normalize(np.array([1., 3.])), [0.25, 0.75])
    assert_allclose(fn.normalize(np.zeros(2)), [0.5, 0.5])
    assert fn.first_true_index([0, 0, 1]) == 2
    from thermosteam.base import SparseVector
    material = SparseVector(np.array([1., 0.]))
    fn.remove_negligible_negative_values(material)
    assert_allclose(material.to_array(), [1., 0.])
    material = SparseVector(np.array([0., -1e-20]))
    fn.remove_negligible_negative_values(material)
    assert (material.to_array() >= 0).all()

def test_dictionary_view():
    from thermosteam.base.dictionary_view import MassFlowDict
    dct = {}
    view = MassFlowDict(dct, {'a': 2., 'b': 4.})
    assert not view and len(view) == 0
    view['a'] = 4.
    view.setdefault('b', 8.)
    view.setdefault('b', 100.)
    assert len(view) == 2 and 'a' in view
    assert_allclose(dct['a'], 2.)
    assert list(view) == ['a', 'b']
    assert list(view.keys()) == ['a', 'b']
    assert_allclose(list(view.values()), [4., 8.])
    assert_allclose(dict(view.items())['b'], 8.)
    assert_allclose(view.copy()['a'], 4.)
    assert_allclose(view.get('a'), 4.)
    assert view.get('c', 0) == 0
    view.update({'a': 2.})
    assert_allclose(view['a'], 2.)
    assert_allclose(view.pop('a'), 2.)
    assert_allclose(view.popitem(), 8.)
    view['a'] = 1.
    del view['a']
    view['a'] = 1.
    view.clear()
    assert not view

def test_t_dependent_property():
    Water = tmo.Chemical('Water')
    Psat = Water.Psat.copy()
    assert Psat
    assert Psat.method == Water.Psat.method
    method = Psat.method
    Psat.method = method
    with pytest.raises(ValueError):
        Psat.method = 'not a method'
    with pytest.warns(DeprecationWarning):
        Psat.set_model_priority(method)
    with pytest.raises(RuntimeError):
        Psat.set_model_priority(method, 1)
    with pytest.warns(DeprecationWarning):
        Psat.move_up_model_priority(method)
    with pytest.raises(RuntimeError):
        Psat.move_up_model_priority(method, 1)
    Psat.add_model(lambda T: 1e3, name='CONSTANT_TEST')
    @Psat.add_model(name='CONSTANT_TEST2')
    def f(T): return 1e3
    Psat.method = 'CONSTANT_TEST'
    assert_allclose(Psat(300.), 1e3)

def test_functor():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import sys
    fmod = sys.modules['thermosteam.base.functor']
    @tmo.functor(var='Psat')
    def AntoineCoverage(T, a, b, c):
        return 10.0**(a - b / (T + c))
    @tmo.functor(var='V.l')
    def LinearVCoverage(T, P, a, b):
        return a + b * P * 1e-12
    @tmo.functor
    def NoVarCoverage(T, a):
        return a
    f = AntoineCoverage.functor(10.116, 1687.5, -42.98)
    assert_allclose(f(373.15), 101157, rtol=1e-2)
    g = f.copy()
    assert g.__dict__ == f.__dict__ and g is not f
    for h in (type(f).from_other(f), type(f).from_args([10.116, 1687.5, -42.98]),
              type(f).from_kwargs(dict(a=10.116, b=1687.5, c=-42.98))):
        assert_allclose(h(373.15), 101157, rtol=1e-2)
    assert 'Psat' in str(f) and repr(f).startswith('<')
    f.show()
    f2 = AntoineCoverage.functor(10.116, 1687.5, -42.98)
    f2.__dict__['model'] = f
    f2.show()
    Ts, Ys = f.tabulate_vs_T(300, 350, 'degC', 'bar')
    assert_allclose(Ts[0], 26.85, rtol=1e-2)
    assert (Ys > 0).all()
    Ps, Ys = f.tabulate_vs_P(1e5, 2e5, 'bar', 'bar')
    assert len(Ys) == 2
    v = LinearVCoverage.functor(1e-5, 1.)
    assert_allclose(v(300, 101325), 1e-5, rtol=5e-2)
    Ts, Ys = v.tabulate_vs_T(300, 350, 'degC', 'm3/mol')
    Ps, Ys = v.tabulate_vs_P(1e5, 2e5, 'bar', 'm3/mol')
    assert (Ys > 0).all()
    try:
        f.plot_vs_T(300, 350, label='Antoine')
        v.plot_vs_T(300, 350, label='V')
        v.plot_vs_P(1e5, 2e5, label='V')
    finally:
        plt.close('all')
    n = NoVarCoverage.functor(1.)
    assert 'NoVarCoverage' in str(n)
    assert fmod.display_asfunctor(f, show_var=False).endswith('Pa')
    assert fmod.functor_name(1) == 'int'
    assert fmod.parse_var('V.l') == ['V', 'l']
    assert fmod.get_units('V.l', {'V': 'm3'}) == 'm3'
    assert 'K' in fmod.var_with_units('T')
    assert fmod.convert_var(1., None, None) == 1.
    assert 'Temperature' in fmod.describe_parameter('T', None)
    assert 'dummy_var' in fmod.describe_parameter('dummy_var', None)
    with pytest.raises(ValueError):
        fmod.functor_matching_params(('x',))
    @fmod.functor_lookalike
    class Lookalike:
        var = 'Psat'
        units = None
        def __call__(self, T): pass
    assert 'Psat' in str(Lookalike())

def test_units_of_measure():
    from thermosteam import units_of_measure as uom
    assert uom.parse_units_notation(None) == (None, None)
    assert uom.parse_units_notation(':.3g') == (None, '.3g')
    assert uom.parse_units_notation('kg:.3g') == ('kg', '.3g')
    assert uom.parse_units_notation('kg') == ('kg', None)
    assert 'kg' in uom.reformat_units('Flow [kg/hr]')
    u = uom.UnitsOfMeasure('kg/hr')
    assert uom.UnitsOfMeasure(u) is u
    assert u and str(u) == 'kg/hr' and 'kg/hr' in repr(u)
    assert_allclose(u.conversion_factor('g/hr'), 1000, rtol=1e-2)
    assert_allclose(u.unconvert(1000., 'g/hr'), 1., rtol=1e-2)
    notation = uom.DisplayNotation('.3g', '.3g', '.3g')
    assert 'DisplayNotation' in repr(notation)
    units = uom.DisplayUnits(T='K', P=['Pa', 'bar'], flow=('kmol/hr', 'kg/hr'))
    assert 'DisplayUnits' in repr(units)
    units.T = 'degC'
    units.flow = 'kg/hr'
    with pytest.raises(AttributeError):
        units.dummy = 'K'
    with pytest.raises(tmo.exceptions.DimensionError):
        units.T = 'kg'
    with pytest.raises(tmo.exceptions.DimensionError):
        units.flow = 'K'

def test_registry_utilities():
    from thermosteam.utils import registry
    assert registry.is_valid_ID('a') and not registry.is_valid_ID(1)
    with pytest.raises(RuntimeError):
        registry.check_valid_ID(1)
    with pytest.raises(RuntimeError):
        registry.check_valid_alias(1)
    with pytest.raises(RuntimeError):
        registry.check_valid_alias('1a')

def test_separations_errors():
    from thermosteam import separations
    with pytest.raises(tmo.exceptions.InfeasibleRegion):
        separations.check_partition_infeasibility([np.array([True])], True)
    with pytest.raises(ValueError):
        separations.material_balance(('Water',), [])

if __name__ == '__main__':
    test_misc_helpers()
    test_misc_rounding()
    test_misc_arrays_and_strings()
    test_misc_chemical_dicts()
    test_extended_signature()
    test_representation()
    test_registered()
    test_registered_franchise()
    test_heat_and_power()
    test_plots()
    test_stream_filters()
    test_variable_assignment_inference()
    test_exceptions()
    test_functional()
    test_dictionary_view()
    test_t_dependent_property()
    test_functor()
    test_units_of_measure()
    test_registry_utilities()
    test_separations_errors()
