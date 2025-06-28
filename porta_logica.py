def tabela_verdade_binaria(nome, func):
    print(f"--- {nome} ---")
    print("A B | Saída")
    print("------------")
    for a in [0, 1]:
        for b in [0, 1]:
            saida = func(a, b)
            print(f"{a} {b} |   {saida}")
    print()

def tabela_verdade_unaria(nome, func):
    print(f"--- {nome} ---")
    print("A | Saída")
    print("---------")
    for a in [0, 1]:
        saida = func(a)
        print(f"{a} |   {saida}")
    print()

# Funções das portas lógicas
def AND(a, b): return a & b
def OR(a, b): return a | b
def NAND(a, b): return int(not (a & b))
def NOR(a, b): return int(not (a | b))
def XOR(a, b): return a ^ b
def XNOR(a, b): return int(not (a ^ b))
def NOT(a): return int(not a)

# Mostrar tabelas
tabela_verdade_binaria("AND", AND)
tabela_verdade_binaria("OR", OR)
tabela_verdade_unaria("NOT", NOT)
tabela_verdade_binaria("NAND", NAND)
tabela_verdade_binaria("NOR", NOR)
tabela_verdade_binaria("XOR", XOR)
tabela_verdade_binaria("XNOR", XNOR)
