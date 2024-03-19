from vispy import app, scene
from vispy.geometry import Rect
from vispy.scene.visuals import Ellipse
from vispy.color import Color
from vispy.scene.visuals import Text

import imageio

from funcs import *

W, H = 1280, 720  # размеры экрана
N = 1000  # кол-во птиц
ratio = W / H
w, h = ratio, 1
field_size = (w, h)
dt = 0.1
asp = W / H
perception = 1/20
vrange=(0.05, 0.1)
#                  c      a    s      w     n
coeffs = np.array([0.08, 0.05, 2.5,  0.05, 0.02])


'''
cohesion - стремление в геометрический центр локальных соседей
separaion - избегание локального перенаселения
walls - взаимодействие агентов с границами 
noise - шум, символизирующий множество неучтенных факторов 
'''

                    # 0  1   2   3   4   5
                    # x, y, vx, vy, ax, ay
boids = np.zeros((N, 6), dtype=np.float64)


init_boids(boids, asp, vrange=vrange)

canvas = scene.SceneCanvas(show=True, size=(W, H))
view = canvas.central_widget.add_view()
view.camera = scene.PanZoomCamera(rect=Rect(0, 0, asp, 1))
arrows = scene.Arrow(arrows=directions(boids, dt),
                     arrow_color=(1, 1, 1, 1),
                     arrow_size=5,
                     connect='segments',
                     parent=view.scene)

white = Color("#ecf0f1")

# Задаю окружности
ellipse1 = Ellipse(center=(0.2, 0.2), radius=(np.sqrt(0.02), np.sqrt(0.02)),
                  color=None, border_width=4, border_color=white,
                  num_segments=1,
                  parent=view.scene)
ellipse2 = Ellipse(center=(0.4, 0.7), radius=(np.sqrt(0.01), np.sqrt(0.01)),
                  color=None, border_width=4, border_color=white,
                  num_segments=1,
                  parent=view.scene)
ellipse3 = Ellipse(center=(1.4, 0.8), radius=(np.sqrt(0.03), np.sqrt(0.03)),
                  color=None, border_width=4, border_color=white,
                  num_segments=1,
                  parent=view.scene)
ellipse4 = Ellipse(center=(1.4, 0.2), radius=(np.sqrt(0.005), np.sqrt(0.005)),
                  color=None, border_width=4, border_color=white,
                  num_segments=1,
                  parent=view.scene)

# Отрисовываю препятствия: окружности разного радиуса
ellipse1.num_segments = 50
ellipse2.num_segments = 50
ellipse3.num_segments = 50
ellipse4.num_segments = 50

main_txt = Text(f'{N = }'+ '\n' +
                f'c = {coeffs[0]}'+ '\n' +
                f'a = {coeffs[1]}'+ '\n' +
                f's = {coeffs[2]}'+ '\n' +
                # f'w = {coeffs[3]}' + '\n' +
                f'n = {coeffs[4]}' + '\n'
                ,
                parent=canvas.scene,
                color='blue')
main_txt.font_size = 9
main_txt.pos = canvas.size[0] // 8.7, canvas.size[1] // 7 * 5.6

fps_txt = Text(f'FPS = wait',
                parent=canvas.scene,
                color='blue')
fps_txt.font_size = 9
fps_txt.pos = canvas.size[0] // 8.7, canvas.size[1] // 7 * 6.3


frame_count = 0 # переменная, которая понадобится, чтобы не тратить время на обновление текста на экране в каждом кадре

writer = imageio.get_writer(f'boids_{N}.mp4', fps=60) # Сюда записываем видео
fr = 0


def make_video(event):
    """
    Функция для записи видео

    Возвращаемое значение:
    Ничего не возвращает
    """
    global frame_count
    # меняем текст только раз в 30 кадров
    if frame_count % 30 == 0:
        fps_txt.text = "FPS = " + f"{canvas.fps:0.1f}"
        canvas.measure_fps()
    frame_count += 1

    dist = distance(boids)
    dir = directions(boids, dt)
    arrows.set_data(arrows=dir)  # здесь мы высчитываем directions
    flocking(boids, dist, perception, coeffs, field_size, vrange, dir)  # рассчитываем ускорение на следующий шаг
    propagate(boids, dt, vrange)  # рассчитываем движение

    # 30+ секунд видео * 60 кадров в секунду
    if frame_count <= 2000:
        frame = canvas.render(alpha=False)
        writer.append_data(frame)
    else:
        writer.close()
        app.quit()

def update(event):
    """
    Функция, отвечающая за покадровое обновление состояния системы и отрисовку новой картинки.

    Возвращаемое значение:
    Ничего не возвращает
    """
    global frame_count
    # меняем текст только раз в 30 кадров
    if frame_count % 30 == 0:
        fps_txt.text= "FPS = " + f"{canvas.fps:0.1f}"
        canvas.measure_fps()
    frame_count += 1

    dist = distance(boids)
    dir = directions(boids, dt)
    arrows.set_data(arrows=dir) # здесь мы высчитываем directions
    flocking(boids, dist, perception, coeffs, field_size, vrange, dir) #рассчитываем ускорение на следующий шаг
    propagate(boids, dt, vrange) #рассчитываем движение
    canvas.update()


if __name__ == '__main__':
    # timer = app.Timer(interval=0, start=True, connect=make_video)
    timer = app.Timer(interval=0, start=True, connect=update)
    app.run()
