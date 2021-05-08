

def _filter_dict_by_not_none_keys(d, keys=None):
    return {k: d.get(k)
            for k in (keys if keys is not None else d)
            if d.get(k) is not None}

