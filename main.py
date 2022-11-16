import numpy as np
import math

""" Общая структура и функции """
accurateness: str = '.6g'  # Точность до знака
alphabet: list = list('abcdefg')


def entropy(l_p: list) -> float:
    """ Энтропия """
    H: np.float64 = np.float64(0.0)

    s1: str = ''
    s2: str = ''
    for value in l_p:
        value = np.float64(value)

        H += value * np.log2(value)

        s1 += f' - {value} * log2({value})'
        s2 += format(-value * np.log2(value), accurateness) + ' + '

    H = np.float64(format(-H, accurateness))

    print(s1, ' =\n', s2, ' = ', H)
    print()

    return H


def conditional_entropy(full_conditional_p: list, ensemble_p: dict) -> float:
    """ Энтропия """
    H: float = 0.0

    s1: str = ''
    s2: str = ''
    for i, ensemble in enumerate(full_conditional_p):
        s1 += f'p({alphabet[i]}) * '
        s2 += f'{ensemble_p[alphabet[i]]} * '
        temp1 = ''
        temp2 = ''
        for key, value in ensemble.items():
            value = np.float64(value)

            temp1 += f' - p({key}) * log2(p({key}))'
            temp2 += format(-value * np.log2(value), accurateness) + ' + '
            H += float(format(-value * np.log2(value) * ensemble_p.get(alphabet[i]), accurateness))

        s1 += f'({temp1}) + '
        s2 += f'({temp2}) + '

    H = np.float64(format(H, accurateness))

    print(s1, ' =\n', s2, ' = ', H)
    print()

    return H


def transpose_l_dict(lst: list) -> list:
    """ Транспонирование списка словарей """
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


def make_matrix_from_l_dict(l_dic: list) -> np.array:
    """ Переход к шагу:
    p(x_i+1 = a) = p(a|a) * p(x_i = a) + p(a|b) * p(x_i = b)
    ...
    p(a) + p(b) + ... = 1

    ...
    Создаём матрицу коэффициентов размером: (n-1) x n.
    Для применения метода Гаусса потом. """
    matrix_zip: list = list()
    for i in range(len(l_dic)):

        l_i: list = list()
        for count, element in enumerate(l_dic[i].values()):
            if count == i:
                # Сразу переносим коэфы из левой части
                l_i.append(element - 1)
            else:
                l_i.append(element)
        matrix_zip.append(l_i)

    zipped_rows = zip(*matrix_zip)
    result_matrix: list = [list(row) for row in zipped_rows]

    for row in result_matrix:
        row.append(0)

    for j in range(len(result_matrix[0])):
        result_matrix[len(result_matrix) - 1][j] -= 1

    return np.array(result_matrix)


def gauss_pivot_func(matrix: np.array) -> np.array:
    result_matrix = np.copy(matrix)
    for nrow in range(len(result_matrix)):
        # nrow равен номеру строки
        # np.argmax возвращает номер строки с максимальным элементом в уменьшенной матрице
        # которая начинается со строки nrow. Поэтому нужно прибавить nrow к результату
        pivot = nrow + np.argmax(abs(result_matrix[nrow:, nrow]))
        if pivot != nrow:
            # swap
            # matrix[nrow], matrix[pivot] = matrix[pivot], matrix[nrow] - не работает.
            # нужно переставлять строки именно так, как написано ниже
            result_matrix[[nrow, pivot]] = result_matrix[[pivot, nrow]]
        row = result_matrix[nrow]
        divider = row[nrow]  # диагональный элемент
        if abs(divider) < 1e-10:
            # почти нуль на диагонали. Продолжать не имеет смысла, результат счёта неустойчив
            raise ValueError(f'Матрица несовместна. Максимальный элемент в столбце {nrow}: {divider:{accurateness}}')
        # делим на диагональный элемент.
        row /= divider
        # теперь надо вычесть приведённую строку из всех нижележащих строчек
        for lower_row in result_matrix[nrow + 1:]:
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

    make_identity(result_matrix)

    return result_matrix


""" Непосредственно цепь Маркова """


def find_full_conditional_p(list_ensembles_conditional_p: list, dict_ensemble_p: dict) -> list:
    list_ensembles_full_conditional_p: list = list()

    transposed_l_dict: list = transpose_l_dict(list_ensembles_conditional_p)

    for i in range(len(transposed_l_dict)):
        ensemble: dict = dict()

        for (key1, value1), (key2, value2) in zip(transposed_l_dict[i].items(), dict_ensemble_p.items()):
            p = np.float64(format(np.float64(value1) * np.float64(value2), accurateness))
            ensemble[f"{key2}{str(key1).split('|')[0]}"] = p

            print(f'p(xi = {key2}, xi+1 = {alphabet[i]}) = p({key2}) * p({key1}) = {value2} * {value1} = {p}')

        print('\n')
        list_ensembles_full_conditional_p.append(ensemble)

    return list_ensembles_full_conditional_p


def markov_chain(list_ensembles_conditional_p: list) -> list:
    print('Вход: \n', list_ensembles_conditional_p)
    print()

    # Матрица коэффов
    matrix = make_matrix_from_l_dict(list_ensembles_conditional_p)
    print('Matrix: \n', matrix)
    print()

    # Единичная матрица решений
    I_matrix = gauss_pivot_func(matrix)
    print('I_matrix: \n', I_matrix)
    print()

    # p(a), p(b), p(c), ...
    dict_ensemble_p: dict = dict()
    for i in range(len(I_matrix)):
        dict_ensemble_p[alphabet[i]] = float(format(I_matrix[i][len(I_matrix[0]) - 1], accurateness))

    print('p(a), p(b), p(c), ...\n', dict_ensemble_p)
    print()

    # [{p(x_i = a|x_i+1 = a), p(x_i = a|x_i+1 = b)},
    # {p(x_i = b|x_i+1 = a), p(x_i = b|x_i+1 = b)}]
    result: list = find_full_conditional_p(list_ensembles_conditional_p, dict_ensemble_p)

    print('H_xi = ')
    p: list = list()
    for p_i in dict_ensemble_p.values():
        p.append(p_i)
    H_xi = entropy(p)

    full_p: list = list()
    for lp_i in result:
        full_p += list(lp_i.values())
    print('H(x_i x_i+1) = ')
    H_xi_xi_1 = entropy(full_p)

    print(f'1. H_xi(x_i+1) = H(x_i x_i+1) - H(x_i) = {H_xi_xi_1} - {H_xi} = {format(H_xi_xi_1 - H_xi, accurateness)}')

    print(f'2. H_xi(x_i+1) = ')
    H_suffix_xi_xi_1 = conditional_entropy(list_ensembles_conditional_p, dict_ensemble_p)

    return result


def main():
    suffix: int = 1
    # Чтение файла
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
        # len(line) == 2
        line = line.split(':')
        for word in line:
            # В случае лишних пробелов
            word.replace(' ', '')

        # Если Вы заметили ошибку в этом блоке,
        # то у вас неправильно форматирован вход.
        # Пожалуйста, пишите так:
        # a|a: 0.5
        # b|a: 0.5
        #
        # a|b: 0.6
        # b|b: 0.4
        what_alpha = str(line[0].split('|')[1])
        if what_alpha != alphabet[i // math.floor(math.sqrt(len(lines)))]:
            raise ValueError(f'Ищите и исправляйте строку: {line}')

        # a|a, p(a|a)
        key, value = line
        ensemble[key] = float(value)

        # В list_ensembles_conditional_p входят словари длиной sqrt(кол-во строчек).
        # Весь ввод данных оставляется на совесть пользователя. :)
        if (i + 1) % math.floor(math.sqrt(len(lines))) == 0:
            list_ensembles_conditional_p.append(ensemble)
            ensemble = dict()

    # print(list_ensembles_conditional_p)

    # Однако я предусмотрел проверку на валидность сумм вероятностей.
    def test_valid(l_ensemble: list) -> bool:
        for i in range(len(l_ensemble)):
            summary: float = 0
            for value in l_ensemble[i].values():
                summary += value
            expr = (abs(1.0 - summary) < 1e-10)
            if not expr:
                return False
        return True

    # Сумма вероятностей для каждого из p(*|a), p(*|b), ... должна быть равна 1.0
    if test_valid(list_ensembles_conditional_p):

        # Возвращает список словарей:
        # [{p(x_i = a|x_i+1 = a), p(x_i = a|x_i+1 = b)},
        # {p(x_i = b|x_i+1 = a), p(x_i = b|x_i+1 = b)}]
        result = markov_chain(list_ensembles_conditional_p)

        # Запись в файл
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
