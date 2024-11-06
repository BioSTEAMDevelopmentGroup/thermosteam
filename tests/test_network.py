# -*- coding: utf-8 -*-
"""
"""

def test_disconnect():
    import thermosteam as tmo
    class Mix(tmo.AbstractUnit):
        _N_ins = 2
        _N_outs = 1
        _ins_size_is_fixed = False
    
    class Split(tmo.AbstractUnit):
        _N_ins = 1
        _N_outs = 2
        _outs_size_is_fixed = False
    
    tmo.settings.set_thermo([])
    M = Mix(ins=())
    S = Split(ins=())
    
    # Test outs size fixed
    M.outs[0].disconnect_source()
    assert len(M.outs) == 1 and isinstance(M.outs[0], tmo.AbstractMissingStream)
    
    # Test ins size not fixed
    M.ins[0].disconnect_sink()
    assert len(M.ins) == 2 and isinstance(M.ins[0], tmo.AbstractMissingStream) and isinstance(M.ins[1], tmo.AbstractStream)
    
    # Test ins size fixed
    S.ins[0].disconnect_sink()
    assert len(S.ins) == 1 and isinstance(S.ins[0], tmo.AbstractMissingStream)
    
    # Test outs size not fixed
    S.outs[0].disconnect_source()
    assert len(S.outs) == 2 and isinstance(S.outs[0], tmo.AbstractMissingStream) and isinstance(S.outs[1], tmo.AbstractStream)
    
    
if __name__ == '__main__':
    test_disconnect()
