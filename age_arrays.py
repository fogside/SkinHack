import numpy as np

def age_arrays(array_of_ages):
    # Принимает на вход numpy массив вида [ [age_1], [age_2], ... , [age_n] ], где age_i имеет тип int
    array_of_ages = np.ndarray.flatten(array_of_ages)
    #print(array_of_ages)

    array_rough = np.zeros(array_of_ages.shape[0], dtype=int)
    array_1     = np.zeros(array_of_ages.shape[0], dtype=int)
    array_2     = np.zeros(array_of_ages.shape[0], dtype=int)
    array_3     = np.zeros(array_of_ages.shape[0], dtype=int)

    # Грубая разметка и очевидные заполнения подмассивов
    indexes     = array_of_ages < 15
    array_rough[indexes] = 0
    array_2[indexes]     = 0
    array_3[indexes]     = 0

    indexes     = ((array_of_ages >= 15) & (array_of_ages < 45))
    array_rough[indexes] = 1
    array_1[indexes]     = 2
    array_3[indexes]     = 0

    indexes     = array_of_ages >= 45
    array_rough[indexes] = 2
    array_1[indexes]     = 2
    array_2[indexes]     = 3

    # Уточняющие заполнения подмассивов

    indexes     = (array_of_ages < 5)
    array_1[indexes]     = 0

    indexes     = ((array_of_ages >= 5) & (array_of_ages < 15))
    array_1[indexes]     = 1

    indexes     = ((array_of_ages >= 15) & (array_of_ages < 30))
    array_2[indexes]     = 1

    indexes     = ((array_of_ages >= 30) & (array_of_ages < 45))
    array_2[indexes]     = 2

    indexes     = ((array_of_ages >= 45) & (array_of_ages < 60))
    array_3[indexes]     = 1

    indexes     = (array_of_ages >= 60)
    array_3[indexes]     = 2

    #print (array_rough)
    #print (array_1)
    #print (array_2)
    #print (array_3)

    return [np.matrix(array_rough).T, np.matrix(array_1).T, np.matrix(array_2).T, np.matrix(array_3).T]

print(age_arrays(np.array([[0], [10], [15], [20], [25], [30], [35], [40], [70], [100]])))
