"""Data-only call return encoding; no pickle or executable deserialization."""
def encode(value):
    if value is None or type(value) in (bool,int,float,str): return value
    if type(value) is list: return [encode(x) for x in value]
    if type(value) is tuple: return {'$tuple':[encode(x) for x in value]}
    if type(value) is dict:
        if any(type(k) not in (str,int) for k in value): raise TypeError('non-JSON mapping keys')
        return {'$dict':[[encode(k),encode(v)] for k,v in value.items()]}
    raise TypeError('unsupported APPS return type')


def decode(value,depth=0):
    if depth>32: raise ValueError('return nesting limit')
    if value is None or type(value) in (bool,int,float,str): return value
    if isinstance(value,list): return [decode(x,depth+1) for x in value]
    if isinstance(value,dict) and set(value)=={'$tuple'}: return tuple(decode(x,depth+1) for x in value['$tuple'])
    if isinstance(value,dict) and set(value)=={'$dict'}:
        result={}
        for key,item in value['$dict']:
            key=decode(key,depth+1)
            if type(key) not in (str,int) or key in result: raise ValueError('invalid mapping key')
            result[key]=decode(item,depth+1)
        return result
    raise ValueError('unrecognized wire object')
