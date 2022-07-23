from functools import partial
class Config:
    def parse(self,raw):
        for k,v in raw.items():
            if type(v) == dict:
                curr_func = v.pop('FUNC')
                return_as_class = v.pop('as_class',False)
                assert curr_func in globals()
                for key,val in v.items():
                    if type(val) == str and val in globals():
                        v[key] = globals()[val]
                v = partial(globals()[curr_func],**v)
                if return_as_class:
                    v = v()
            elif v in globals():
                v = globals()[v]
            setattr(self,k,v)
    def __init__(self, raw):
        self._second_round = raw.pop('SECOND_ROUND') if 'SECOND_ROUND' in raw else {}
        self.parse(raw)

    def second_round(self):
        self.parse(self._second_round)