import numpy as np
import math

alphabet: list = list('abcdefg')


def entropy(l_p: list) -> np.float64:
    H: np.float64 = np.float64(0.0)
    for value in l_p:
        H += np.float64(value) * np.log2(np.float64(value))
    H = np.float64(-H)
    return np.float64(format(H, '.8g'))


def transpose_l_dict(lst: list) -> list:
    transposed_l_dict: list = list()
    for i in range(len(lst)):
        transposed_l_dict.append(dict())

    count = 0
    for l in lst:
        for key, value in l.items():
            transposed_l_dict[count][key] = value
            count += 1
        count = 0

    return transposed_l_dict


def find_full_conditional_p(list_ensembles_conditional_p: list, dict_ensemble_p: dict) -> list:
    list_ensembles_full_conditional_p: list = list()

    transposed_l_dict: list = transpose_l_dict(list_ensembles_conditional_p)

    for i in range(len(transposed_l_dict)):
        ensemble: dict = dict()

        for (key1, value1), (key2, value2) in zip(transposed_l_dict[i].items(), dict_ensemble_p.items()):

            p = np.float64(format(np.float64(value1) * np.float64(value2), '.5g'))
            ensemble[f'{key2}{key1}'] = p

            print(f'p(xi = {key2}, xi+1 = {alphabet[i]}) = p({key2}) * p({key1}) = {value2} * {value1} = {p}')

        print('\n')
        list_ensembles_full_conditional_p.append(ensemble)

    return list_ensembles_full_conditional_p


def make_matrix_from_l_dict(l_dic: list) -> np.array:
    matrix_zip: list = list()
    for i in range(len(l_dic)):

        l_i: list = list()
        for count, element in enumerate(l_dic[i].values()):
            if count == i:
                l_i.append(element - 1)
            else:
                l_i.append(element)
        matrix_zip.append(l_i)

    zipped_rows = zip(*matrix_zip)

    transpose_matrix: list = [list(row) for row in zipped_rows]
    for row in transpose_matrix:
        row.append(0)

    for j in range(len(transpose_matrix[0])):
        transpose_matrix[len(transpose_matrix) - 1][j] -= 1

    return np.array(transpose_matrix)


def gaussPivotFunc(matrix: np.array) -> np.array:
    copy_matrix = np.copy(matrix)
    for nrow in range(len(copy_matrix)):
        # nrow равен номеру строки
        # np.argmax возвращает номер строки с максимальным элементом в уменьшенной матрице
        # которая начинается со строки nrow. Поэтому нужно прибавить nrow к результату
        pivot = nrow + np.argmax(abs(copy_matrix[nrow:, nrow]))
        if pivot != nrow:
            # swap
            # matrix[nrow], matrix[pivot] = matrix[pivot], matrix[nrow] - не работает.
            # нужно переставлять строки именно так, как написано ниже
            copy_matrix[[nrow, pivot]] = copy_matrix[[pivot, nrow]]
        row = copy_matrix[nrow]
        divider = row[nrow]  # диагональный элемент
        if abs(divider) < 1e-10:
            # почти нуль на диагонали. Продолжать не имеет смысла, результат счёта неустойчив
            raise ValueError(f'Матрица несовместна. Максимальный элемент в столбце {nrow}: {divider:.3g}')
        # делим на диагональный элемент.
        row /= divider
        # теперь надо вычесть приведённую строку из всех нижележащих строчек
        for lower_row in copy_matrix[nrow + 1:]:
            factor = lower_row[nrow]  # элемент строки в колонке nrow
            lower_row -= factor * row  # вычитаем, чтобы получить ноль в колонке nrow

    # приводим к диагональному виду
    def make_identity(copy_matrix) -> np.array:
        # перебор строк в обратном порядке
        for nrow in range(len(copy_matrix) - 1, 0, -1):
            row = copy_matrix[nrow]
            for upper_row in copy_matrix[:nrow]:
                factor = upper_row[nrow]
                upper_row -= factor * row
        return copy_matrix
    make_identity(copy_matrix)

    return copy_matrix


def markov_chain(list_ensembles_conditional_p: list) -> list:
    print('Вход: \n', list_ensembles_conditional_p)
    print()

    matrix = make_matrix_from_l_dict(list_ensembles_conditional_p)
    print('Matrix: \n', matrix)
    print()

    I_matrix = gaussPivotFunc(matrix)
    print('I_matrix: \n', I_matrix)
    print()

    dict_ensemble_p: dict = dict()

    for i in range(len(I_matrix)):
        dict_ensemble_p[alphabet[i]] = format(I_matrix[i][len(I_matrix[0]) - 1], '.5g')

    print('p(a), p(b), p(c), ...\n', dict_ensemble_p)
    print()

    result = find_full_conditional_p(list_ensembles_conditional_p, dict_ensemble_p)

    p: list = list()
    for p_i in dict_ensemble_p.values():
        p.append(p_i)
    print(p)
    H_xi = entropy(p)
    s1: str = ''
    s2: str = ''
    for value in dict_ensemble_p.values():
        value = np.float64(value)
        s1 += f'- {value} * log2({value})'
        s2 += format(-value * np.log2(value), '.5g')
        s2 += ' + '
    s1 += '= \n'
    s2 += '= '
    print(s1, s2, H_xi)
    print()

    full_p: list = list()
    for l in result:
        full_p += list(l.values())
    H_xi_xi_1 = entropy(full_p)
    s3: str = ''
    s4: str = ''
    for value in full_p:
        value = np.float64(value)
        s3 += f'- {value} * log2({value})'
        s4 += format(-value * np.log2(value), '.5g')
        s4 += ' + '
    s3 += '= \n'
    s4 += '= '
    print(s3, s4, H_xi_xi_1)
    print()

    H_suffix_xi_xi_1 = np.float64(format(H_xi_xi_1 - H_xi, '.8g'))
    print(f'Hxi(xi+1) = H(xixi+1) - H(xi) = {H_xi_xi_1} - {H_xi} = {H_suffix_xi_xi_1}')
    return result


def main():
    suffix: int = 1
    # Читаем файл
    file_input = open(f'./input/input_{suffix}.txt', 'r')
    # Создаём массив непустных строк из файла
    lines = file_input.read().splitlines()
    lines = list(filter(None, lines))
    # Закрываем файл
    file_input.close()

    # Создаём лист, где будут храниться словари
    list_ensembles_conditional_p: list = list()

    # Добавляем ключ, значение в словарь
    ensemble: dict = dict()
    for i, line in enumerate(lines):
        line = line.split(':')
        for word in line:
            word.replace(' ', '')

        what_alpha = str(line[0].split('|')[1])
        if what_alpha != alphabet[i // math.floor(math.sqrt(len(lines)))]:
            raise ValueError(f'Ищите и исправляйте строку: {line}')

        key, value = line
        ensemble[key] = np.float64(value)

        if (i + 1) % math.floor(math.sqrt(len(lines))) == 0:
            list_ensembles_conditional_p.append(ensemble)
            ensemble = dict()

    # print(list_ensembles_conditional_p)
    def test_valid(l_ensemble: list) -> bool:
        for i in range(len(l_ensemble)):
            summary: np.float64 = np.float64(0)
            for value in l_ensemble[i].values():
                summary += value
            expr = (abs(1.0 - summary) < 1e-10)
            if not expr:
                return False
        return True

    # Сумма вероятностей должна быть равна 1.0
    if test_valid(list_ensembles_conditional_p):

        result = markov_chain(list_ensembles_conditional_p)

        file_output = open(f'./output/output_{suffix}.txt', 'w')
        for ensemble in result:
            for key, value in ensemble.items():
                file_output.write(f'{key}: {value}\n')
            file_output.write('\n')    
        file_output.close()

    else:
        raise ValueError('Неверный вход!')


if __name__ == '__main__':
    main()
