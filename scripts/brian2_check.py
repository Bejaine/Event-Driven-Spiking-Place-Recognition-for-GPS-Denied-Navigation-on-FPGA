"""Tiny Brian2 network to verify Cython code generation works (Phase B)."""
import sys, time, os
import brian2 as b2
target = sys.argv[1] if len(sys.argv) > 1 else 'cython'
b2.prefs.codegen.target = target
b2.prefs.codegen.runtime.cython.cache_dir = os.environ.get('TMPDIR', '/tmp') + '/b2cache_check'
G = b2.NeuronGroup(100, 'dv/dt = -v/(10*ms) : 1', threshold='v>1', reset='v=0', method='euler')
P = b2.PoissonGroup(100, 50*b2.Hz)
S = b2.Synapses(P, G, 'w:1', on_pre='v_post += w'); S.connect(); S.w = 0.05; S.delay = 'rand()*10*ms'
M = b2.SpikeMonitor(G)
t0 = time.time(); b2.run(1*b2.second); print(target, 'spikes', M.num_spikes, 'wall %.2fs' % (time.time()-t0), 'dt', b2.defaultclock.dt)
