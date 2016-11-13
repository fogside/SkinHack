import numpy as np

def age_arrays(array_of_ages):
    # Принимает на вход numpy массив вида [ [age_1], [age_2], ... , [age_n] ], где age_i имеет тип int
    array_of_ages = np.ndarray.flatten(array_of_ages)
    #print(array_of_ages)

    array_0 = np.zeros(array_of_ages.shape[0], dtype=int)
    array_1 = np.zeros(array_of_ages.shape[0], dtype=int)
    array_2 = np.zeros(array_of_ages.shape[0], dtype=int)

    # Грубая разметка array_0

    indexes     = array_of_ages < 15
    array_0[indexes] = 0
    indexes     = ((array_of_ages >= 15) & (array_of_ages < 45))
    array_0[indexes] = 1
    indexes     = ((array_of_ages >= 45) & (array_of_ages < 100))
    array_0[indexes] = 2

    # Заполнения подмассива array_1

    indexes     = (array_of_ages < 5)
    array_1[indexes]     = 0
    indexes     = ((array_of_ages >= 5) & (array_of_ages < 10))
    array_1[indexes]     = 1
    indexes     = ((array_of_ages >= 10) & (array_of_ages < 15))
    array_1[indexes]     = 2

    indexes     = ((array_of_ages >= 15) & (array_of_ages < 20))
    array_1[indexes]     = 3
    indexes     = ((array_of_ages >= 20) & (array_of_ages < 30))
    array_1[indexes]     = 4
    indexes     = ((array_of_ages >= 30) & (array_of_ages < 45))
    array_1[indexes]     = 5

    indexes     = ((array_of_ages >= 45) & (array_of_ages < 55))
    array_1[indexes]     = 6
    indexes     = ((array_of_ages >= 55) & (array_of_ages < 75))
    array_1[indexes]     = 7
    indexes     = ((array_of_ages >= 75) & (array_of_ages < 100))
    array_1[indexes]     = 8

    # Автозаполнение array_2

    for i in range(9):
        indexes = ((array_of_ages >= (15 / 9.) * i) & (array_of_ages < (15 / 9.) * (i + 1)))
        array_2[indexes] = i

    for i in range(3):
        indexes = ((array_of_ages >= 15 + (5 / 3.) * i) & (array_of_ages < 15 + (5 / 3.) * (i + 1)))
        array_2[indexes] = i + 9

    for i in range(3):
        indexes = ((array_of_ages >= 20 + (10 / 3.) * i) & (array_of_ages < 20 + (10 / 3.) * (i + 1)))
        array_2[indexes] = i + 12

    for i in range(3):
        indexes = ((array_of_ages >= 30 + (15 / 3.) * i) & (array_of_ages < 30 + (15 / 3.) * (i + 1)))
        array_2[indexes] = i + 15

    for i in range(3):
        indexes = ((array_of_ages >= 45 + (10 / 3.) * i) & (array_of_ages < 45 + (10 / 3.) * (i + 1)))
        array_2[indexes] = i + 18

    for i in range(3):
        indexes = ((array_of_ages >= 55 + (20 / 3.) * i) & (array_of_ages < 55 + (20 / 3.) * (i + 1)))
        array_2[indexes] = i + 21

    for i in range(3):
        indexes = ((array_of_ages >= 75 + (25 / 3.) * i) & (array_of_ages < 75 + (25 / 3.) * (i + 1)))
        array_2[indexes] = i + 24

    """
    for i in range(array_0.shape[0]):
        print (array_0[i], array_1[i], array_2[i])
    """

    return [np.matrix(array_0).T, np.matrix(array_1).T, np.matrix(array_2).T]

#print     ( np.arange(0, 100) )
age_arrays( np.arange(0, 100) )
