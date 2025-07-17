from typing import List, Optional

class Matrix:
    def _init_(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.data = [[0.0]*cols for _ in range(rows)]

    def _getitem_(self, pos):
        i, j = pos
        return self.data[i][j]

    def _setitem_(self, pos, value):
        i, j = pos
        self.data[i][j] = value

    def _str_(self):
        s = ""
        for row in self.data:
            s += " ".join(f"{v:.2f}" for v in row) + "\n"
        return s

    def determinant(self):
        raise NotImplementedError("Determinante não implementado para matriz geral")

    def transpose(self):
        data_t = []
        for j in range(self.cols):
            row = [self.data[i][j] for i in range(self.rows)]
            data_t.append(row)
        m = GeneralMatrix(self.cols, self.rows)
        m.data = data_t
        return m

    def _add_(self, other):
        if not (self.rows == other.rows and self.cols == other.cols):
            raise ValueError("Dimensões incompatíveis para soma")
        result = GeneralMatrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result[i,j] = self[i,j] + other[i,j]
        return result

    def _mul_(self, other):
        if isinstance(other, (int, float)):
            result = GeneralMatrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result[i,j] = self[i,j] * other
            return result
        elif isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError("Dimensões incompatíveis para multiplicação")
            result = GeneralMatrix(self.rows, other.cols)
            for i in range(self.rows):
                for j in range(other.cols):
                    s = 0.0
                    for k in range(self.cols):
                        s += self[i,k]*other[k,j]
                    result[i,j] = s
            return result
        else:
            raise TypeError("Multiplicação só suportada com escalar ou matriz")

class GeneralMatrix(Matrix):
    def _init_(self, rows:int, cols:int, data:Optional[List[List[float]]] = None):
        super()._init_(rows, cols)
        if data:
            if len(data) != rows or any(len(row) != cols for row in data):
                raise ValueError("Dados incompatíveis")
            self.data = data

    def determinant(self):
        if self.rows != self.cols:
            raise ValueError("Determinante só para matrizes quadradas")
        def det_recursive(m):
            n = len(m)
            if n == 1:
                return m[0][0]
            if n == 2:
                return m[0][0]*m[1][1] - m[0][1]*m[1][0]
            det = 0
            for c in range(n):
                minor = [row[:c]+row[c+1:] for row in m[1:]]
                det += ((-1)**c) * m[0][c] * det_recursive(minor)
            return det
        return det_recursive(self.data)

class DiagonalMatrix(Matrix):
    def _init_(self, diag: List[float]):
        n = len(diag)
        super()._init_(n, n)
        self.diag = diag

    def _getitem_(self, pos):
        i, j = pos
        if i == j:
            return self.diag[i]
        else:
            return 0.0

    def _setitem_(self, pos, value):
        i, j = pos
        if i == j:
            self.diag[i] = value
        elif value != 0.0:
            raise ValueError("Só zeros fora da diagonal")

    def _str_(self):
        s = ""
        n = len(self.diag)
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(f"{self.diag[i]:.2f}")
                else:
                    row.append("0.00")
            s += " ".join(row) + "\n"
        return s

    def determinant(self):
        prod = 1.0
        for v in self.diag:
            prod *= v
        return prod

class LowerTriangularMatrix(Matrix):
    def _init_(self, n:int, data:Optional[List[List[float]]]=None):
        super()._init_(n, n)
        if data is None:
            self.data = [ [0.0]*(i+1) for i in range(n) ]
        else:
            if len(data) != n or any(len(data[i]) != i+1 for i in range(n)):
                raise ValueError("Dados inválidos para triangular inferior")
            self.data = data

    def _getitem_(self, pos):
        i, j = pos
        if j > i:
            return 0.0
        return self.data[i][j]

    def _setitem_(self, pos, value):
        i, j = pos
        if j > i and value != 0.0:
            raise ValueError("Só zeros acima da diagonal em triangular inferior")
        if j > i and value == 0.0:
            return
        self.data[i][j] = value

    def _str_(self):
        n = self.rows
        s = ""
        for i in range(n):
            row = []
            for j in range(n):
                if j <= i:
                    row.append(f"{self.data[i][j]:.2f}")
                else:
                    row.append("0.00")
            s += " ".join(row) + "\n"
        return s

    def determinant(self):
        prod = 1.0
        for i in range(self.rows):
            prod *= self[i,i]
        return prod

class UpperTriangularMatrix(Matrix):
    def _init_(self, n:int, data:Optional[List[List[float]]]=None):
        super()._init_(n, n)
        if data is None:
            self.data = [ [0.0]*(n - i) for i in range(n) ]
        else:
            if len(data) != n or any(len(data[i]) != n - i for i in range(n)):
                raise ValueError("Dados inválidos para triangular superior")
            self.data = data

    def _getitem_(self, pos):
        i, j = pos
        if j < i:
            return 0.0
        return self.data[i][j - i]

    def _setitem_(self, pos, value):
        i, j = pos
        if j < i and value != 0.0:
            raise ValueError("Só zeros abaixo da diagonal em triangular superior")
        if j < i and value == 0.0:
            return
        self.data[i][j - i] = value

    def _str_(self):
        n = self.rows
        s = ""
        for i in range(n):
            row = []
            for j in range(n):
                if j >= i:
                    row.append(f"{self.data[i][j - i]:.2f}")
                else:
                    row.append("0.00")
            s += " ".join(row) + "\n"
        return s

    def determinant(self):
        prod = 1.0
        for i in range(self.rows):
            prod *= self[i,i]
        return prod

def ler_inteiro(msg:str, minimo:int = None, maximo:int = None) -> int:
    while True:
        try:
            val = int(input(msg))
            if minimo is not None and val < minimo:
                print(f"Valor deve ser >= {minimo}")
                continue
            if maximo is not None and val > maximo:
                print(f"Valor deve ser <= {maximo}")
                continue
            return val
        except:
            print("Entrada inválida, tente novamente.")

def criar_matriz():
    print("Escolha tipo de matriz:")
    print("1 - Geral")
    print("2 - Diagonal")
    print("3 - Triangular Inferior")
    print("4 - Triangular Superior")
    tipo = ler_inteiro("Opção: ",1,4)
    n = ler_inteiro("Número de linhas: ",1)
    m = n
    if tipo == 1:
        m = ler_inteiro("Número de colunas: ",1)
    if tipo == 1:
        print(f"Insira os elementos da matriz ({n}x{m}):")
        data = []
        for i in range(n):
            while True:
                linha = input(f"Linha {i+1} (separar por espaço): ").strip().split()
                if len(linha) != m:
                    print(f"Informe exatamente {m} valores.")
                    continue
                try:
                    nums = [float(x) for x in linha]
                    data.append(nums)
                    break
                except:
                    print("Entrada inválida, tente novamente.")
        return GeneralMatrix(n,m,data)

    elif tipo == 2:
        print(f"Insira os elementos da diagonal (tamanho {n}):")
        diag = []
        while len(diag) < n:
            linha = input(f"Valores (separar por espaço), faltam {n - len(diag)}: ").strip().split()
            try:
                nums = [float(x) for x in linha]
                diag.extend(nums)
            except:
                print("Entrada inválida, tente novamente.")
        diag = diag[:n]
        return DiagonalMatrix(diag)

    elif tipo == 3:
        print(f"Insira os elementos da matriz triangular inferior ({n} linhas):")
        data = []
        for i in range(n):
            while True:
                linha = input(f"Linha {i+1} (apenas {i+1} valores): ").strip().split()
                if len(linha) != i+1:
                    print(f"Informe exatamente {i+1} valores.")
                    continue
                try:
                    nums = [float(x) for x in linha]
                    data.append(nums)
                    break
                except:
                    print("Entrada inválida, tente novamente.")
        return LowerTriangularMatrix(n,data)

    elif tipo == 4:
        print(f"Insira os elementos da matriz triangular superior ({n} linhas):")
        data = []
        for i in range(n):
            while True:
                linha = input(f"Linha {i+1} (apenas {n - i} valores): ").strip().split()
                if len(linha) != n - i:
                    print(f"Informe exatamente {n - i} valores.")
                    continue
                try:
                    nums = [float(x) for x in linha]
                    data.append(nums)
                    break
                except:
                    print("Entrada inválida, tente novamente.")
        return UpperTriangularMatrix(n,data)
    return None


def main():
    matriz_atual = None
    while True:
        print("\nMENU:")
        print("1 - Criar matriz")
        print("2 - Mostrar matriz")
        print("3 - Calcular determinante")
        print("4 - Transpor matriz")
        print("5 - Somar com outra matriz")
        print("6 - Multiplicar por outra matriz")
        print("7 - Multiplicar por escalar")
        print("0 - Sair")
        opc = ler_inteiro("Opção: ",0,7)
        if opc == 0:
            print("Saindo...")
            break
        elif opc == 1:
            matriz_atual = criar_matriz()
            print("Matriz criada com sucesso.")
        elif matriz_atual is None:
            print("Crie uma matriz primeiro (opção 1).")
        elif opc == 2:
            print("Matriz atual:")
            print(matriz_atual)
        elif opc == 3:
            try:
                det = matriz_atual.determinant()
                print(f"Determinante: {det:.4f}")
            except Exception as e:
                print(f"Erro: {e}")
        elif opc == 4:
            matriz_atual = matriz_atual.transpose()
            print("Matriz transposta:")
            print(matriz_atual)
        elif opc == 5:
            print("Digite a segunda matriz para soma (mesmas dimensões):")
            try:
                outra = criar_matriz()
                if not (matriz_atual.rows == outra.rows and matriz_atual.cols == outra.cols):
                    print("As matrizes devem ter as mesmas dimensões para soma.")
                else:
                    resultado = matriz_atual + outra
                    print("Resultado da soma:")
                    print(resultado)
                    matriz_atual = resultado
            except Exception as e:
                print(f"Erro: {e}")
        elif opc == 6:
            print("Digite a segunda matriz para multiplicação (dimensões compatíveis):")
            try:
                outra = criar_matriz()
                if matriz_atual.cols != outra.rows:
                    print(f"As colunas da primeira matriz ({matriz_atual.cols}) devem ser iguais às linhas da segunda matriz ({outra.rows}) para multiplicação.")
                else:
                    resultado = matriz_atual * outra
                    print("Resultado da multiplicação:")
                    print(resultado)
                    matriz_atual = resultado
            except Exception as e:
                print(f"Erro: {e}")
        elif opc == 7:
            escalar = None
            while escalar is None:
                try:
                    escalar = float(input("Digite o escalar para multiplicar: "))
                except:
                    print("Entrada inválida.")
            matriz_atual = matriz_atual * escalar
            print("Matriz multiplicada por escalar:")
            print(matriz_atual)

if _name_ == "_main_":
    main()
