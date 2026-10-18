"""Shared input semantics: source influence is distinct from target importance."""
def response(value):
    x=max(0.0,min(1.0,float(value)))
    return x*x*(3.0-2.0*x)

def target_priority(control):
    return max(0.1,min(10.0,float(control.get('bodymove_target_priority',1.0))))
