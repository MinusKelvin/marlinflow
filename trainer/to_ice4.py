ACTIVATION_RANGE = 127
WEIGHT_SCALE = 64

def quantize(tensor, scale: int):
    q = []
    for elem in tensor:
        if type(elem) == list:
            q.append(quantize(elem, scale))
        else:
            q.append(round(elem * scale))
    return q

def transpose(matrix):
    t = [[0] * len(matrix) for _ in range(len(matrix[0]))]
    for i in range(len(matrix[0])):
        for j in range(len(matrix)):
            t[i][j] = matrix[j][i]
    return t

def to_ice4(params):
    return {
        # "pst": transpose(params["pst.weight"]),
        "ft.weight": transpose(params["ft.weight"]),
        "ft.bias": params["ft.bias"],
        "out.weight": transpose(params["out.weight"]),
        "out.bias": params["out.bias"],
    }
