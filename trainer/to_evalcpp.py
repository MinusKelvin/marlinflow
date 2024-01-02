#!/usr/bin/python3

import math
from contextlib import redirect_stdout

class DataStringer:
    def __init__(self):
        self.little = ""
        self.big = ""

    def add(self, data):
        smallest = min(data)
        for v in data:
            v = round(v - smallest)
            low = v % 95
            high = v // 95
            lc = chr(low + 32)
            hc = chr(high + 32)
            if lc == "\\" or lc == "\"":
                self.little += "\\"
            if hc == "\\" or hc == "\"":
                self.big += "\\"
            self.little += lc
            self.big += hc
        return smallest

def to_evalcpp(last_loss, train_id, param_map):
    print(f"// loss: {last_loss:.4g}    train id: {train_id}")
    print()
    print("#define S(a, b) (a + (b * 0x10000))")
    print()

    mg_scaled = [v * 160 for v in param_map["mg.weight"][0]]
    eg_scaled = [v * 160 for v in param_map["eg.weight"][0]]

    mg = []
    eg = []
    sizes = [48, 16, 3, 16, 3, 16, 3, 16, 3, 16, 48, 1, 8, 1, 1, 2, 1, 1, 4, 1, 1, 6, 1]
    acc = 0
    for s in sizes:
        mg.append(mg_scaled[acc:acc+s])
        eg.append(eg_scaled[acc:acc+s])
        acc += s

    data_string = ""

    mg_stringer = DataStringer()
    eg_stringer = DataStringer()

    defines = []

    def define_param(name, idx, *, sign=1):
        defines.append((
            name,
            round(mg[idx][0]) * sign,
            round(eg[idx][0]) * sign
        ))

    def array_param(name, idx, *, leading_zero=False, sign=1):
        print(f"int {name}[] = {{", end="")
        if leading_zero:
            print("0, ", end="")
        print(", ".join(
            f"S({sign * round(v1)}, {sign * round(v2)})"
            for v1, v2 in zip(mg[idx], eg[idx])
        ), end="")
        print("};")

    mg_off = round(mg_stringer.add(mg[0]))
    eg_off = round(eg_stringer.add(eg[0]))
    defines.append(("PAWN_OFFSET", mg_off, eg_off))
    mg_off = round(mg_stringer.add(mg[10]))
    eg_off = round(eg_stringer.add(eg[10]))
    defines.append(("PASSED_PAWN_OFFSET", mg_off, eg_off))

    mg_stringer.add(mg[9])
    eg_stringer.add(eg[9])

    print("int QUADRANTS[] = {", end="")
    for i in range(1, 9, 2):
        mg_off = mg_stringer.add(mg[i])
        eg_off = eg_stringer.add(eg[i])
        for j, (mg_q, eg_q) in enumerate(zip([0] + mg[i+1], [0] + eg[i+1])):
            if j % 4 == 0: print("\n   ", end="")
            print(f" S({round(mg_off + mg_q)}, {round(eg_off + eg_q)})", end=",")
    print("\n};")

    define_param("BISHOP_PAIR", 11)
    array_param("DOUBLED_PAWN", 12, sign=-1)
    define_param("TEMPO", 13)
    define_param("ISOLATED_PAWN", 14, sign=-1)
    array_param("PROTECTED_PAWN", 15, leading_zero=True)
    define_param("ROOK_OPEN", 16)
    define_param("ROOK_SEMIOPEN", 17)
    array_param("PAWN_SHIELD", 18)
    define_param("KING_OPEN", 19)
    define_param("KING_SEMIOPEN", 20)
    array_param("MOBILITY", 21, leading_zero=True)
    define_param("KING_RING_ATTACKS", 22)

    print()

    print(f"#define DATA_LOW \"{mg_stringer.little + eg_stringer.little}\"")
    print(f"#define DATA_HIGH \"{mg_stringer.big + eg_stringer.big}\"")

    print()
    for name, mg, eg in defines:
        print(f"#define {name} S({mg}, {eg})")

def dump_result(result):
    with open("eval.cpp", "w") as f:
        with redirect_stdout(f):
            to_evalcpp(result["loss"], result["train_id"], result["params"])
