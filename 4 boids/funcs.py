import numpy as np
from scipy.spatial.distance import cdist
from numba import njit, prange


@njit
def mean_axis(boids):
    """
    Функция - аналог np.mean(axis = 0). Позволяет высчитать среднее по оси x

    Параметры:
    boids: np.ndarray;     массив боидов

    Возвращаемое значение:
    res : float;           среднее значение
    """
    n = boids.shape[1]
    res = np.empty(n, dtype=boids.dtype)
    for i in range(n):
        res[i] = boids[:, i].mean()
    return res


def init_boids(boids: np.ndarray, asp: float, vrange: tuple = (0., 1.)):
    """
    Функция, отвечающая за начальную инициализацию боидсов.

    Параметры:
    boids: np.ndarray;         изначально пустой массив с характеристиками для боидов. Функция его заполняет
    asp: float;                коотношение сторон экрана
    vrange: tuple;             ограничения на скорости

    Возвращаемое значение:
    Функция ничего не возвращает
    """
    n = boids.shape[0]
    rng = np.random.default_rng()

    boids[:, 0] = rng.uniform(0., asp, size=n)
    boids[:, 1] = rng.uniform(0., 1., size=n)

    alpha = rng.uniform(0, 2*np.pi, size=n)
    v = rng.uniform(*vrange, size=n)
    c, s = np.cos(alpha), np.sin(alpha)
    boids[:, 2] = v * c
    boids[:, 3] = v * s

def directions(boids: np.ndarray, dt: float):
    """
    Функция, отвечающая за создание массива из наситоящих и будущих координат птичек.

    Параметры:
    boids: np.ndarray;   Массив боидов
    dt: float;           временной шаг

    Возвращаемое значение:
    массив из настоящей и будущей координат птичек
    """
    pos = boids[:, :2]
    delta_pos = dt * boids[:, 2:4]
    pos0 = pos - delta_pos
    return np.hstack((pos0, pos))


@njit
def v_clip(boids: np.ndarray, lims):
    """
    Функция, отвечающая за ограничение скорости. Она "обрезает" вектора
    скорости для боидов, если те выходят за разрешенный диапазон

    Параметры:
    boids: np.ndarray;   Массив боидов
    lims: tuple;         Ограничения на скорость

    Возвращаемое значение:
    Функция ничего не возвращает
    """
    v = np.sum(boids * boids, axis=1)**0.5
    mask = v > 0
    v_clip = np.clip(v, *lims)
    boids[mask] *= (v_clip[mask] / v[mask]).reshape(-1, 1)


@njit
def propagate(boids: np.ndarray, dt: float, vrange: tuple):
    """
    Функция пересчитывает скорость птичек в зависимости от ускорений.

    Параметры:
    boids: np.ndarray;   Массив боидов
    dt: float;           временной шаг
    vrange: tuple;       ограничения на скорость

    Возвращаемое значение:
    Функция ничего не возвращает
    """
    boids[:, 2:4] += dt * boids[:, 4:6]
    v_clip(boids[:, 2:4], lims=vrange)
    boids[:, 0:2] += dt * boids[:, 2:4]


@njit
def walls(boids: np.ndarray, asp: float):
    """
    Функция, отвечающая за силовое взаимодействие со стенами.
    Чем ближе птичка к стене, тем сильнее она хочет от неё отлететь.

    Её можно подключить при желании, разкомментировав 1 строчку в floacking
    и прибавления компоненты ускорения, отвечающей за взаимодействие со стенами. Но для видео её не использовала.

    Параметры:
    boids: np.ndarray;   Массив боидов
    asp: float;          Соотношение стен

    Возвращаемое значение:
    Компоненты ускорения, отвечающие за силовое отталктивание от стен
    """
    c = 1
    x = boids[:, 0]
    y = boids[:, 1]

    a_left = 1 / (np.abs(x) + c)**2
    a_right = -1 / (np.abs(x - asp) + c)**2

    a_bottom = 1 / (np.abs(y) + c)**2
    a_top = -1 / (np.abs(y - 1.) + c)**2

    return np.column_stack((a_left + a_right, a_bottom + a_top))


@njit
def cohesion(boids: np.ndarray,
             idx: int,
             mask_visibility: np.ndarray,
             perception: float):
    """
    Функция подсчитывает составляющую ускорения, отвечающую за желание птичек сбиваться в группу
    и лететь стайкой, а не порознь.

    Параметры:
    boids: np.ndarray;             массив боидов
    i: int;                        номер рассматриваемого боида
    mask_visibility: np.ndarray;   маска для учета зоны видимости
    perception: float;             контанста, отвечающая за область видимости птички

    Возвращаемое значение:
    """
    center = mean_axis(boids[mask_visibility, :2])
    a = (center - boids[idx, :2]) / perception
    return a


@njit
def separation(boids: np.ndarray,
               i: int,
               mask_visibility: np.ndarray):
    """
    Функция, отвечающая за асчет компоненты ускорения, которое направлено на
    избежание столкновений с соседями.

    Параметры:
    boids: np.ndarray;             массив боидов
    i: int;                        номер рассматриваемого боида
    mask_visibility: np.ndarray;   маска для учета зоны видимости

    Возвращаемое значение:
    компонента ускорения
    """
    d = mean_axis(boids[mask_visibility, :2] - boids[i, :2])
    return -d / ((d[0]**2 + d[1]**2) + 1)


@njit
def alignment(boids: np.ndarray,
              i: int,
              mask_visibility: np.ndarray,
              vrange: tuple):
    """
    Функция, позволяющая высчитать компоненту ускорения, которая отвечает
    за выравнивание скорости по локальным соседям.

    Параметры:
    boids: np.ndarray;             массив боидов
    i: int;                        номер рассматриваемого боида
    mask_visibility: np.ndarray;   маска для учета зоны видимости
    vrange: tuple;                 ограничения на скорость

    Возвращаемое значение:
    Компонента ускорения
    """
    v_mean = mean_axis(boids[mask_visibility, 2:4])
    a = (v_mean - boids[i, 2:4]) / (2 * vrange[1])
    return a


def distance(boids: np.ndarray):
    """
    Функция, отвечающая за нахождения расстояний между боидами.

    Параметры:
    boids: np.ndarray; массив боидсов

    Возвращаемое значение:
    Массив попарных расстояний между птичками
    """
    return cdist(boids[:, :2], boids[:, :2])

@njit
def find_intersection(circle_center, radius, line_coefficients): # Функция для нахождения точек пересечения вектора скорости с окружностью
    """
    Функция, которая позволяет найти точки пересечения прямой и окружности
    по заданным коэффициентам уравнения прямой, координатам центра окружности и ей радиусу.

    Параметры:
    circle_center: np.ndarray;   цент окружнсоти, с которой боид взаимодействует в данный момент
    radius: float;               радиус окружности, с которой вступает в коллизию боид
    line_coefficients: tuple;    коэффициенты уравнения заданной прямой

    Возвращаемое значение:
    Координаты двух точек пересечения прямой и окружности
    """
    h, k = circle_center
    m, c = line_coefficients

    # Распаковываем координаты центра окружности
    x_center, y_center = circle_center

    # Находим координаты x точки пересечения путем решения квадратного уравнения
    A = 1 + m ** 2
    B = 2 * (m * (c - k) - h)
    C = h ** 2 + (c - k) ** 2 - radius ** 2
    discriminant = B ** 2 - 4 * A * C

    if discriminant < 0:
        print(discriminant)
        # return None  # Окружность и прямая не пересекаются

    x1 = (-B + np.sqrt(discriminant)) / (2 * A)
    x2 = (-B - np.sqrt(discriminant)) / (2 * A)

    # Находим соответствующие значения y
    y1 = m * x1 + c
    y2 = m * x2 + c

    return np.array([[x1, y1], [x2, y2]])


@njit
def find_normal_point(circle_center, radius, line_coefficients, boids):
    """
    Функция, которая находит, через какую точку прошла бы нормаль к окружности,
    имеющая тот же угловой коэффициент, что и заданная прямая. Так как таких точек две, выбирает ту точку, которая ближе к
    предыдущей координате боида.

    Параметры:
    circle_center: np.ndarray;   цент окружнсоти, с которой боид взаимодействует в данный момент
    radius: float;               радиус окружности, с которой вступает в коллизию боид
    line_coefficients: tuple;    коэффициенты уравнения заданной прямой
    boids: np.ndarray;           массив характеристик одного боида

    Возвращаемое значение:
    Координаты искомой точки
    """
    b1 = circle_center[1] - line_coefficients[0] * circle_center[0]
    normal_arr = find_intersection(circle_center, radius, (line_coefficients[0], b1))

    # берем ту точку, через которую проходит нормаль, которая ближе к предыдущему шагу  боида
    if ((normal_arr[0][0] - boids[0])**2 + (normal_arr[0][1] - boids[1])**2 < (normal_arr[1][0] - boids[0])**2 + (normal_arr[1][1] - boids[1])**2):
        normal_point = normal_arr[0]
    else:
        normal_point = normal_arr[1]

    return normal_point


@njit
def find_x(center, r, k, b1):
    """
    Вспомогательная функция, которая по заданным коэффициентам квадратного уравнения
    находит x

    Параметры:
    center: np.ndarray;        цент окружнсоти, с которой боид взаимодействует в данный момент
    radius: float;             радиус окружности, с которой вступает в коллизию боид
    k: float;                  угол наклона уравнения прямой
    b1: float;                 свободный коэффициент уравнения прямой

    Возвращаемое значение:
    Искомый х
    """
    sqr = np.sqrt(-(b1**2) + 2*b1*center[1] - (center[1]**2) - 2*b1*center[0]*k - ((center[0]**2)*(k**2)) + r**2 + (k**2)*(r**2))
    numerator = center[0] - b1*k + center[1]*k + sqr # -sqr +sqr будут совпадать
    denominator = 1 + k**2
    frac = numerator/denominator
    return frac


@njit
def angle(dir, c_center, radius, intersec_pos, k, b): #находит угол между вектором скорости и отрезком соединяющим
                                        # точку вхождения вектора в окружность и центром окружности
    """
    Функция, отвечающая за вычисление  угла вхождения боида в окружность.
    Также, она позволяет определить, следует в дальнейшем
    разворачивать вектор скорости по часовой или против часовой стрелки.

    Параметры:
    dir: np.ndarray;             массив с последней и предпоследней координатами боидсов
    c_center: np.ndarray;        цент окружнсоти, с которой боид взаимодействует в данный момент
    radius: float;               радиус окружности, с которой вступает в коллизию боид
    intersec_pos: np.array;      координаты точки, в которой боид вошел в окружность
    k: float;                    угол наклона уравнения прямой
    b: float;                    свободный коэффициент уравнения прямой

    Возвращаемое значение:
    Половину угла, на который нужно развернуть боид.
    А также флаг, который был мне нужен, когда я хотела дополниетльно отодвигать птичку
    немного в сторону от круга, чтобы она не застревала между разворотом влево и разворотом вправо
    Но я закомментировала это, так как иногда птички попадали в окружность на 1 кадр, что хуже,
    чем если они просто иногда в замешательстве перед окружностью, пока не появится новая птичка-сосед,
    за которой она захочет полететь))
    Для того, чтобы этого не происходило, у случае, когда поворот следует осуществлять против часовой стрелки,
    я поворачиваю сильнее чем по часовой. Это позволяет избежать случаев,
    когда птичка поворачивает туда-сюда много кадров подряд, пока не "ухватится" за пролетающих соседей.
    """
    k1 = (dir[3] - dir[1]) / (dir[2] - dir[0])
    k2 = (intersec_pos[1] - c_center[1]) / (intersec_pos[0] - c_center[0])
    tan = abs((k1 - k2)/(1 + k1*k2))
    ang = np.arctan(tan)  # НО нам необходимо определить, поворачивать по часовой или против часово стрелки

    # найти точку, через которую вошла бы "нормаль" к окружности с тем же углом наклона
    n_point = find_normal_point(c_center, radius, (k, b), dir)

    # Найти точку, через которую прошла бы касательная к окружности с тем же углом наклона
    # касательная будет находиться правее(и левее) нормали на радиус
    shifted_arr = np.array([[n_point[0] - radius, n_point[1]], [n_point[0] + radius, n_point[1]]]) # лева иправая
    # выбирать одну из них будем в зависимости от случая

    flag_shift = 0

    # x2 > x1
    if dir[2] > dir[0]:
        # y2 > y1
        if dir[3] > dir[1]:
            b1 = shifted_arr[1][1] - k*shifted_arr[1][0]  # свободный коэф нужной касательной
            x_t = find_x(c_center, radius, k, b1)
            # y_t = x_t*k + b1
            if dir[0] > n_point[0] and dir[0] < x_t:
                ang *= -1.3
                flag_shift = 1
        # y2 < y1
        else:
            b1 = shifted_arr[0][1] - k*shifted_arr[0][0]  # свободный коэф нужной касательной
            x_t = find_x(c_center, radius, k, b1)
            y_t = x_t*k + b1
            if dir[1] < n_point[1] and dir[1] > y_t:
                ang *= -1.3
                flag_shift = 2
    # x2 < x1
    else:
        # y2 > y1
        if dir[3] > dir[1]:
            b1 = shifted_arr[1][1] - k*shifted_arr[1][0]  # свободный коэф нужной касательной
            x_t = find_x(c_center, radius, k, b1)
            y_t = x_t*k + b1
            if dir[1] > n_point[1] and dir[1] < y_t:
                ang *= -1.3
                flag_shift = 3
        # y1 > y2
        else:
            b1 = shifted_arr[0][1] - k*shifted_arr[0][0]  # свободный коэф нужной касательной
            x_t = find_x(c_center, radius, k, b1)
            # y_t = x_t*k + b1
            if dir[0] < n_point[0] and dir[0] > x_t:
                ang *= -1.3
                flag_shift = 4
    return ang, flag_shift

@njit
def mask_circle_handler(dir: np.ndarray,
                        circle_center: np.ndarray,
                        r: float,
                        boids: np.ndarray):
    """
    Функция, отвечающая за обработку взаимодействия боидов с окружностями.
    Сначала вызывает функцию для нахождения точки, в которой боид прилетел в окружность,
    Затем вызывает функцию для вычисления угла, под которым боид блетел в окружность,
    В конце вектор скорости разворачивается на нужный угол (если ang > 0  - по часовой, еслиang < 0 - против часовой)

    Параметры:
    dir: np.ndarray;             массив с последней и предпоследней координатами боидсов
    circle_center: np.ndarray;   цент окружнсоти, с которой боид взаимодействует в данный момент
    r: float;                    радиус окружности, с которой вступает в коллизию боид
    boids: np.ndarray;           массив характеристик одного боида

    Возвращаемое значение:
    Функция ничего не возвращает
    """
    k = (dir[3] - dir[1]) / (dir[2] - dir[0])
    b = dir[1] - k * dir[0]

    intersec_pos_all = find_intersection(circle_center, r,(k, b))  # Ищем точки пересечения вектора направления с окружностью

    # Тк. у окружности и прямой 2 точки пересечения (в данном случае всегда, так как из-за особенностей округления точно по касательной прямая пройти не может)
    # Нужно выбрать ближайшую точку пересечения к предыдущей координате боида
    if ((intersec_pos_all[0][0] - boids[0])**2 + (intersec_pos_all[0][1] - boids[1])**2 < (intersec_pos_all[1][0] - boids[0])**2 + (intersec_pos_all[1][1] - boids[1])**2):
        inter_pos = intersec_pos_all[0]
    else:
        inter_pos = intersec_pos_all[1]

    ang = angle(dir, circle_center, r, inter_pos, k, b)

    vx, vy = boids[2], boids[3]  # создаю отедльную переменную, так как мне обе компоненты нужно будет менять
    boids[2] = vx * np.cos(2*ang[0]) - vy * np.sin(2*ang[0])  # х координата скорости
    boids[3] = vx * np.sin(2*ang[0]) + vy * np.cos(2*ang[0])  # y координата скорости

    # if(ang[1] == 1):
    #     boids[0] = inter_pos[0]
    #     boids[1] = inter_pos[1] - 0.01
    # else:
    #     if(ang[1] == 2):
    #         boids[0] = inter_pos[0] - 0.01
    #         boids[1] = inter_pos[1]
    #     else:
    #         if (ang[1] == 3):
    #             boids[0] = inter_pos[0] + 0.01
    #             boids[1] = inter_pos[1]
    #         else:
    #             if (ang[1] == 4):
    #                 boids[0] = inter_pos[0]
    #                 boids[1] = inter_pos[1] + 0.01
    #             else:
    #                 boids[0] = inter_pos[0]
    #                 boids[1] = inter_pos[1]

    boids[0] = inter_pos[0]
    boids[1] = inter_pos[1]

@njit(parallel=True)
def circles_collision(boids: np.ndarray,
                      dir: np.ndarray):
    """
    Функция, отвечающая за упругое отталкивание от стен.
    Функция сначала проверяет при помощи маски, какие боидсы столкнулись с какими окружностями.
    Вызывает функцию, которая высчитывает угол между вектором directions и касательной.
    Далее, в зависимости от направления входа вектора directions в окружность,
    функция определяет, развернуть скорость по часовой или против часовой стрелки.

    Параметры:
    boids: np.ndarray; массив боидсов
    dir: np.ndarray;   массив с последней и предпоследней координатами боидсов

    Возвращаемое значение:
    Функция ничего не возвращает
    """
    mask_circles = np.empty((4, boids.shape[0]))
    mask_circles[0] = (boids[:, 0] - 0.2) ** 2 + (boids[:, 1] - 0.2) ** 2 < 0.02
    mask_circles[1] = (boids[:, 0] - 0.4) ** 2 + (boids[:, 1] - 0.7) ** 2 < 0.01
    mask_circles[2] = (boids[:, 0] - 1.4) ** 2 + (boids[:, 1] - 0.8) ** 2 < 0.03
    mask_circles[3] = (boids[:, 0] - 1.4) ** 2 + (boids[:, 1] - 0.2) ** 2 < 0.005

    #проверяет для каждого боида
    for i in prange(boids.shape[0]):
        # окружность с координатами (относительными) (0.2, 0.2), r = sqrt(0.02)
        if mask_circles[0][i]:
            r = np.sqrt(0.02)
            circle_center = np.array([0.2, 0.2])
            mask_circle_handler(dir[i], circle_center, r, boids[i])

        # окружность с координатами (относительными) (0.4, 0.7), r = sqrt(0.01)
        if mask_circles[1][i]:
            r = np.sqrt(0.01)
            circle_center = (0.4, 0.7)
            mask_circle_handler(dir[i], circle_center, r, boids[i])

        # окружность с координатами (относительными) (1.4, 0.8), r = sqrt(0.03)
        if mask_circles[2][i]:
            r = np.sqrt(0.03)
            circle_center = (1.4, 0.8)
            mask_circle_handler(dir[i], circle_center, r, boids[i])

        # окружность с координатами (относительными) (1.4, 0.2), r = sqrt(0.005)
        if mask_circles[3][i]:
            r = np.sqrt(0.005)
            circle_center = (1.4, 0.2)
            mask_circle_handler(dir[i], circle_center, r, boids[i])


@njit(parallel=True)
def walls_collision(boids, field_size):
    """
    Функция, отвечающая за упругое отталкивание от стен.
    Если боид прилетает к стене, то, в зависимости от конкретной стены,
    его скорость отражается по оси x или по оси y

    Параметры:
    boids : np.nd.array;  массив боидов
    field_size : tuple;   относиетльные размеры экрана

    Возвращаемое значение:
    Функция ничего не возвращает
    """
    mask_walls = np.empty((4, boids.shape[0]))
    mask_walls[0] = boids[:, 1] > field_size[1]
    mask_walls[1] = boids[:, 0] > field_size[0]
    mask_walls[2] = boids[:, 1] < 0
    mask_walls[3] = boids[:, 0] < 0

    for i in prange(boids.shape[0]):
        if mask_walls[0][i]:
            boids[i][3] = -boids[i][3]
            boids[i][1] = field_size[1] - 0.001

        if mask_walls[1][i]:
            boids[i][2] = -boids[i][2]
            boids[i][0] = field_size[0] - 0.001

        if mask_walls[2][i]:
            boids[i][3] = -boids[i][3]
            boids[i][1] = 0.001

        if mask_walls[3][i]:
            boids[i][2] = -boids[i][2]
            boids[i][0] = 0.001


@njit()
def noise():
    """
    Функция генерирует случайное изменение вектора ускорения

    Параметры:
    Функция ничего не принимает

    Возвращаемое значение:
    Вектор из двух чисел, которые соответсвуют случайному изменению по x и по y
    """
    arr = np.random.uniform(low=-1, high=1, size=(2)) # генерируем 2 случайных числа от -1 до 1
    return arr


@njit(parallel=True)
def flocking(boids: np.ndarray,
             dist: np.ndarray,
             perception: float,
             coeffs: np.ndarray,
             field_size,
             vrange:tuple,
             directions: np.ndarray): # нужно для обработки отталкивания от окружностей
    """
    Функция, отвечающая за подсчет ускорений боидсов в каждом кадре.

    Параметры:
    boids: np.ndarray;       массив боидов
    dist: np.ndarray;        попарное расстояние между всеми боидами
    perception: float;       как далеко птички "видят"
    coeffs: np.ndarray;      коэффициенты взаимодействия
    field_size: tuple;       относительные размеры экрана
    vrange: tuple;           диапазон возможных скоростей
    directions: np.ndarray;  хранит текущую и предыдущую координату

    Возвращаемое значение:
    Функция ничего не возвращает
    """

    #walls_collision(boids, field_size)
    circles_collision(boids, directions)

    N = boids.shape[0]

    for i in prange(N):
        dist[i, i] = perception + 1
    mask_visibility = dist < perception # Маска, которая рассматривает только тех боидов для расчета, которые в области видимости

    wal = walls(boids, field_size[0])

    for i in prange(N): #prange
        if not np.any(mask_visibility[i]):
            coh = np.zeros(2)
            alg = np.zeros(2)
            sep = np.zeros(2)
            ns = noise()
        else:
            coh = cohesion(boids, i, mask_visibility[i], perception)
            alg = alignment(boids, i, mask_visibility[i], vrange)
            sep = separation(boids, i, mask_visibility[i])
            ns = noise()
        a = coeffs[0] * coh + coeffs[1] * alg + \
            coeffs[2] * sep + coeffs[4] * ns  + coeffs[3] * wal[i]
        boids[i, 4:6] = a